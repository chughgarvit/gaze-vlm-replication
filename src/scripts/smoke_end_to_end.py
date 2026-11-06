#!/usr/bin/env python3
"""Exercise the entire import/extraction/evaluation wiring without real-data claims."""
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gaze_memory.benchmark import PILOT_TASKS
from gaze_memory.core import save_json


def main():
    source = ROOT / "results/preprocessing_smoke/source/test.mp4"
    if not source.is_file():
        raise SystemExit("Run scripts/smoke_videollm.py first to create the synthetic test clip")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = ROOT / "results/end_to_end_smoke" / timestamp
    dataset = folder / "fixture_dataset"
    (dataset / "qa").mkdir(parents=True)
    name = "OP01-R01-SyntheticFixture.mp4"
    for task, cutoff in zip(PILOT_TASKS, ["00:01.25", "00:01.50", "00:01.85"]):
        save_json(dataset / "qa" / (task + ".json"), [{"video_path": name, "questions": [{
            "question": "SYNTHETIC WIRING TEST: What color is the moving square?",
            "time_stamp": cutoff, "options": ["A. Blue", "B. Orange", "C. Green", "D. Black"],
            "answer": "A", "answer_object": "SYNTHETIC_SECRET_MUST_NOT_ENTER_PROMPT",
            "gazed_objects": ["SYNTHETIC_SECRET_MUST_NOT_ENTER_PROMPT"],
        }]}])
    for archive in ["videos_egtea_original.tar", "videos_egtea_viz.tar"]:
        with tarfile.open(dataset / archive, "w") as bundle:
            bundle.add(source, arcname="nested/" + name)
    # Both archives contain the same synthetic clip. This tests pairing only,
    # not the validity/effectiveness of a publisher gaze visualization.
    output = folder / "pipeline"
    subprocess.run([sys.executable, str(ROOT / "scripts/run_end_to_end.py"),
                    "--dataset-root", str(dataset), "--skip-download", "--dry-run",
                    "--synthetic-fixture", "--max-videos", "1", "--max-samples", "3",
                    "--output", str(output)], check=True)
    manifest = json.loads((output / "qa_manifest.json").read_text())
    rows = json.loads((output / "evaluation/predictions.json").read_text())
    summary = json.loads((output / "evaluation/summary.json").read_text())
    assert manifest["data_kind"] == "synthetic_test_fixture"
    assert len(rows) == 6
    assert "accuracy" not in summary and all("correct" not in row for row in rows)
    assert {row["mode"] for row in rows} == {"full", "publisher_overlay"}
    for sample in manifest["samples"]:
        prompts = []
        for mode in ["full", "publisher_overlay"]:
            path = output / "evaluation" / sample["id"] / mode / "prompt.json"
            prompts.append(json.loads(path.read_text()))
        assert prompts[0] == prompts[1]
        assert "SYNTHETIC_SECRET" not in json.dumps(prompts)
        for key in ["frames_manifest", "overlay_frames_manifest"]:
            frames = json.loads((output / sample[key]).read_text())["frames"]
            assert all(frame["timestamp_s"] <= sample["cutoff_s"] for frame in frames)
            assert all(frame["gaze"] is None for frame in frames)
    print("Passed safe extraction, official-format QA import, real ffmpeg timestamp pairing, "
          "identical paired prompts, and dry-run output checks. Synthetic fixtures only; no model accuracy.")
    print(f"Artifacts: {folder}")


if __name__ == "__main__":
    main()
