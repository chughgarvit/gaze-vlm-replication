import importlib.util
import io
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("download_egtea", Path(__file__).resolve().parents[1] / "scripts/download_egtea.py")
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class Response(io.BytesIO):
    def __init__(self, value, headers, status=200):
        super().__init__(value)
        self.headers, self.status = headers, status


class EgteaDownloadTests(unittest.TestCase):
    def page(self):
        return """<a href='https://www.dropbox.com/s/fixture/gaze_data.zip?dl=0'>Gaze Data</a>
        <a href='https://www.dropbox.com/s/fixture/action_annotation.zip'>Action Annotations</a>
        <a href='https://www.dropbox.com/s/fixture/video_links.txt'>Links to raw videos</a>
        <a href='readme.md'>Readme File</a>
        <a href='https://www.dropbox.com/s/fixture/video_clips.tar'>Trimmed Action Clips</a>"""

    def test_discovery_uses_publisher_anchors_not_guessed_urls(self):
        links = download.discover(self.page())
        self.assertEqual(links["gaze_data.zip"], "https://www.dropbox.com/s/fixture/gaze_data.zip?dl=1")
        self.assertEqual(links["readme.md"], download.PUBLISHER + "readme.md")

    def test_changed_or_ambiguous_page_fails(self):
        with self.assertRaises(ValueError):
            download.discover("<html>offline</html>")
        with self.assertRaises(ValueError):
            download.discover(self.page() + "<a href='other.zip'>Gaze Data</a>")

    def test_raw_manifest_rejects_nonvideo_and_duplicate_names(self):
        good = "https://www.dropbox.com/s/fixture/OP01-R01-PastaSalad.mp4?dl=0"
        self.assertEqual(download.raw_links(good)[0][0], "OP01-R01-PastaSalad.mp4")
        with self.assertRaises(ValueError):
            download.raw_links("https://example.org/malicious.sh")
        with self.assertRaises(ValueError):
            download.raw_links(good + "\n" + good)

    def test_link_schemes_and_credentials_rejected(self):
        for value in ("file:///etc/passwd", "ftp://example.org/video.mp4", "https://user:password@example.org/video.mp4"):
            with self.assertRaises(ValueError):
                download.direct_url(value)

    def test_bounded_transfer_hash_and_cached_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gaze.zip"
            budget = {"remaining": 10}
            response = Response(b"dataset", {"Content-Length": "7", "ETag": '"fixture"'})
            with patch.object(download.urllib.request, "urlopen", return_value=response) as call:
                first = download.download("https://example.org/gaze.zip", path, budget)
                cached = download.download("https://example.org/gaze.zip", path, budget)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(first["size"], 7)
            self.assertTrue(cached["cached"])
            self.assertEqual(budget["remaining"], 3)
            self.assertEqual(first["sha256"], cached["sha256"])

    def test_safe_resume_and_unhonored_range_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gaze.zip"
            partial = path.with_suffix(".zip.partial")
            partial.write_bytes(b"data")
            download.save(path.with_suffix(".zip.transfer.json"),
                          {"source_url": "https://example.org/gaze.zip", "size": 7, "etag": '"fixture"'})
            response = Response(b"dataset", {"Content-Length": "7", "ETag": '"fixture"'}, status=200)
            with patch.object(download.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    download.download("https://example.org/gaze.zip", path, {"remaining": 10})
            self.assertEqual(partial.read_bytes(), b"data")
            response = Response(b"set", {"Content-Length": "3", "Content-Range": "bytes 4-6/7", "ETag": '"fixture"'}, status=206)
            with patch.object(download.urllib.request, "urlopen", return_value=response):
                download.download("https://example.org/gaze.zip", path, {"remaining": 10})
            self.assertEqual(path.read_bytes(), b"dataset")

    def test_unknown_length_or_budget_excess_stops_before_transfer(self):
        for headers, budget in (({}, 10), ({"Content-Length": "20"}, 10)):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "gaze.zip"
                with patch.object(download.urllib.request, "urlopen", return_value=Response(b"dataset", headers)):
                    with self.assertRaises(ValueError):
                        download.download("https://example.org/gaze.zip", path, {"remaining": budget})
                self.assertFalse(path.exists())
                self.assertFalse(path.with_suffix(".zip.partial").exists())


if __name__ == "__main__":
    unittest.main()
