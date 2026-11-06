import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from gaze_memory.benchmark import (PILOT_TASKS, choose_frames, extract_selected_videos,
                                   import_samples, match_overlay_frames, timestamp_seconds)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_videollm", ROOT / "scripts/run_videollm.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class BenchmarkTests(unittest.TestCase):
    def test_timestamp(self):
        self.assertEqual(timestamp_seconds("02:22"), 142)
        self.assertEqual(timestamp_seconds("01:02:03.5"), 3723.5)
        for bad in ["[00:08 - 09:19]", "00:60", -1, float("nan")]:
            with self.assertRaises(ValueError):
                timestamp_seconds(bad)

    def test_causal_frame_pair_and_mismatch(self):
        timeline = [(i, i*.25) for i in range(12)]
        chosen = choose_frames(timeline, 1.85, 6)
        self.assertTrue(all(t <= 1.85 for _,t in chosen))
        self.assertEqual(match_overlay_frames(chosen, timeline), chosen)
        self.assertEqual(choose_frames(timeline, 1.85, 1), [(7, 1.75)])
        with self.assertRaises(ValueError):
            match_overlay_frames(chosen, [(i,t+.01) for i,t in timeline])
        with self.assertRaises(ValueError):
            choose_frames(timeline, 10, 6)

    def test_import_drops_answer_annotations_and_other_datasets(self):
        with tempfile.TemporaryDirectory() as folder:
            qa = Path(folder)
            question = {"question": "What is visible?", "time_stamp": "00:02", "answer": "A",
                        "options": ["A. cup", "B. box"], "gazed_objects": ["SECRET"],
                        "answer_object": "SECRET", "background_objects": ["SECRET"]}
            for task in PILOT_TASKS:
                (qa / (task + ".json")).write_text(json.dumps([
                    {"video_path": "OP01-R01-PastaSalad.mp4", "questions": [question]},
                    {"video_path": "R005-7July-GoPro.mp4", "questions": [question]}]))
            samples, hashes = import_samples(qa, max_videos=1)
            self.assertEqual(len(samples), 3)
            self.assertEqual(len(hashes), 3)
            self.assertNotIn("SECRET", json.dumps(samples))
            self.assertEqual(samples[0]["cutoff_s"], 2)

    def test_prompt_identical_except_pixels(self):
        sample = {"comparison": "publisher_overlay", "question": "What is visible?",
                  "options": ["A. cup", "B. box"], "answer": "B", "answer_object": "SECRET"}
        frames = [{"timestamp_s": .5}]
        baseline = runner.make_messages(sample, frames, "full")
        overlay = runner.make_messages(sample, frames, "publisher_overlay")
        self.assertEqual(baseline, overlay)
        self.assertNotIn("SECRET", json.dumps(baseline))

    def test_safe_selected_extraction(self):
        name = "OP01-R01-PastaSalad.mp4"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "videos.tar"
            with tarfile.open(archive, "w") as bundle:
                for filename in ["irrelevant.mp4", "nested/" + name]:
                    member = tarfile.TarInfo(filename)
                    member.size = 4
                    bundle.addfile(member, io.BytesIO(b"test"))
            result = extract_selected_videos(archive, {name}, root / "selected")
            self.assertEqual(set(result), {name})
            self.assertFalse((root / "selected/irrelevant.mp4").exists())
            self.assertEqual((root / "selected" / name).read_bytes(), b"test")

    def test_selected_traversal_and_symlink_rejected(self):
        name = "OP01-R01-PastaSalad.mp4"
        for filename, link in [("../" + name, False), (name, True)]:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                archive = root / "bad.tar"
                with tarfile.open(archive, "w") as bundle:
                    member = tarfile.TarInfo(filename)
                    if link:
                        member.type = tarfile.SYMTYPE
                        member.linkname = "/etc/passwd"
                        bundle.addfile(member)
                    else:
                        member.size = 4
                        bundle.addfile(member, io.BytesIO(b"test"))
                with self.assertRaises(ValueError):
                    extract_selected_videos(archive, {name}, root / "selected")
                self.assertFalse((root / "selected" / name).exists())


if __name__ == "__main__":
    unittest.main()
