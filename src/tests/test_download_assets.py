import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_assets.py"
SPEC = importlib.util.spec_from_file_location("download_assets", SCRIPT)
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class DownloadTests(unittest.TestCase):
    def test_pilot_is_bounded_egtea_not_whole_dataset(self):
        groups = download.asset_groups("pilot")
        self.assertEqual(len(groups), 2)
        files = groups[0]["files"]
        archives = [f for f in files if f.endswith(".tar")]
        self.assertEqual(archives, download.VIDEO_FILES)
        self.assertFalse(any("*" in f or "holoassist" in f or "egoexolearn" in f for f in files))
        self.assertEqual(len([f for f in files if f.startswith("qa/")]), 10)

    def test_annotations_has_no_weights_or_videos(self):
        groups = download.asset_groups("annotations")
        self.assertEqual(len(groups), 1)
        self.assertFalse(any(f.endswith((".tar", ".safetensors")) for f in groups[0]["files"]))

    def test_revision_and_layout_required(self):
        group = {"files": ["example.txt"]}
        info = SimpleNamespace(sha="a" * 40, siblings=[SimpleNamespace(
            rfilename="example.txt", size=12, lfs=None)])
        self.assertEqual(download.resolve_group(group, info)["revision"], "a" * 40)
        info.sha = "main"
        with self.assertRaises(ValueError):
            download.resolve_group(group, info)
        info.sha = "a" * 40
        info.siblings = []
        with self.assertRaises(ValueError):
            download.resolve_group(group, info)

    def test_checksum_mismatch_fails_and_preserves_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "example.txt"
            path.write_text("test")
            record = {"filename": path.name, "size": 4,
                      "publisher_sha256": download.sha256(path)}
            self.assertEqual(download.check_file(path, record)["size"], 4)
            record["publisher_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                download.check_file(path, record)
            self.assertEqual(path.read_text(), "test")


if __name__ == "__main__":
    unittest.main()
