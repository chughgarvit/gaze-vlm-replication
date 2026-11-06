#!/usr/bin/env python3
"""Download original EGTEA Gaze+ assets discovered on the publisher's page.

No guessed Dropbox identifiers, mirrors, login, or downloaded code execution.
Archives remain unextracted. Transfers are bounded; partial files are retained.
"""
import argparse
import hashlib
import json
import re
import shutil
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = "https://cbs.ic.gatech.edu/fpv/"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.current, self.text = [], None, []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.current = dict(attrs).get("href")
            self.text = []

    def handle_data(self, data):
        if self.current is not None:
            self.text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.current is not None:
            self.links.append((" ".join(self.text), urllib.parse.urljoin(PUBLISHER, self.current)))
            self.current = None


def discover(html):
    parser = Links()
    parser.feed(html)
    rules = {
        "gaze_data.zip": r"gaze\s+data",
        "action_annotation.zip": r"action\s+annotations",
        "video_links.txt": r"links\s+to\s+raw\s+videos",
        "readme.md": r"readme\s+file",
        "video_clips.tar": r"trimmed\s+action\s+clips",
    }
    found = {}
    for name, pattern in rules.items():
        candidates = {url for label, url in parser.links if re.search(pattern, label, re.I)}
        if len(candidates) != 1:
            raise ValueError(f"Publisher layout changed: need one unambiguous {name} link, got {len(candidates)}")
        found[name] = direct_url(candidates.pop())
    return found


def direct_url(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Only public HTTP(S) asset links are allowed")
    if parsed.hostname in {"www.dropbox.com", "dropbox.com"}:
        query = dict(urllib.parse.parse_qsl(parsed.query))
        query["dl"] = "1"
        parsed = parsed._replace(query=urllib.parse.urlencode(query))
    return urllib.parse.urlunsplit(parsed)


def raw_links(text):
    urls = re.findall(r"https?://[^\s\"'<>]+", text)
    records = []
    for url in urls:
        url = direct_url(url)
        name = urllib.parse.unquote(Path(urllib.parse.urlsplit(url).path).name)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.(?:mp4|avi|mov)", name, re.I):
            raise ValueError("Unexpected raw-video filename; inspect publisher manifest before proceeding")
        records.append((name, url))
    if not records or len({n for n, _ in records}) != len(records):
        raise ValueError("Missing or duplicate raw-video links")
    return records


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def download(url, destination, budget, limit=None):
    """Resume only when server range, validator, and exact length agree."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    meta_path = destination.with_suffix(destination.suffix + ".transfer.json")
    partial = destination.with_suffix(destination.suffix + ".partial")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("source_url") != url and (partial.exists() or destination.exists()):
        raise ValueError("Existing transfer belongs to a different source; retain it for inspection")
    if destination.exists():
        with destination.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if meta.get("sha256") != digest or meta.get("size") != destination.stat().st_size:
            raise ValueError("Existing asset failed local integrity check")
        return {"path": str(destination), "size": destination.stat().st_size,
                "sha256": digest, "cached": True}
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "egtea-research-downloader/1.0", "Accept-Encoding": "identity"}
    validator = meta.get("etag") or meta.get("last_modified")
    if offset:
        if not validator:
            raise ValueError("Cannot safely resume a partial asset without a server validator")
        headers.update({"Range": f"bytes={offset}-", "If-Range": validator})
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        mime = response.headers.get("Content-Type", "").lower()
        if "text/html" in mime:
            raise ValueError("Asset URL returned a webpage instead of dataset bytes")
        length = response.headers.get("Content-Length")
        if length is None:
            raise ValueError("Server omitted length; refusing an unbounded transfer")
        remaining = int(length)
        if offset:
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or not content_range.startswith(f"bytes {offset}-"):
                raise ValueError("Server did not honor safe resume; partial asset was retained")
            total = int(content_range.rsplit("/", 1)[1])
            current_validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
            if current_validator != validator or total != meta.get("size") or total != offset + remaining:
                raise ValueError("Remote asset changed or has an inconsistent range")
        else:
            total = remaining
        if total > budget["remaining"] or (limit is not None and total > limit):
            raise ValueError("Asset exceeds remaining total/file byte budget")
        if shutil.disk_usage(destination.parent).free < remaining + 1_000_000_000:
            raise ValueError("Insufficient free disk space for this asset plus 1 GB headroom")
        save(meta_path, {"source_url": url, "size": total,
                         "etag": response.headers.get("ETag"),
                         "last_modified": response.headers.get("Last-Modified")})
        transferred = offset
        with partial.open("ab" if offset else "xb") as stream:
            while block := response.read(1024 * 1024):
                transferred += len(block)
                if transferred > total:
                    raise ValueError("Response exceeded its declared byte length")
                stream.write(block)
        if transferred != total:
            raise ValueError("Interrupted/incomplete asset; partial data retained")
        with partial.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        partial.replace(destination)
        budget["remaining"] -= total
        save(meta_path, {"source_url": url, "size": total, "sha256": digest,
                         "etag": response.headers.get("ETag"),
                         "last_modified": response.headers.get("Last-Modified"),
                         "checksum_note": "Local integrity hash, not a publisher-authenticated checksum"})
        return {"path": str(destination), "size": total, "sha256": digest, "cached": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--videos", choices=["raw", "trimmed", "none"], default="raw")
    parser.add_argument("--max-gb", type=float, default=35)
    parser.add_argument("--output", type=Path, default=ROOT / "data/external/egtea_gaze_plus")
    args = parser.parse_args()
    if args.max_gb <= 0:
        parser.error("--max-gb must be positive")
    plan = {"publisher": PUBLISHER, "videos": args.videos,
            "assets": ["readme.md", "gaze_data.zip", "action_annotation.zip", "video_links.txt"],
            "max_bytes": int(args.max_gb * 1e9),
            "note": "Raw videos preferred for pre-action anticipation. No extraction or gaze-clock assumptions."}
    if not args.execute:
        print(json.dumps({"status": "offline_plan_no_download", **plan}, indent=2))
        return 0
    status = {"status": "starting", "started_utc": datetime.now(timezone.utc).isoformat(),
              "files": [], "research_accuracy_available": False, **plan}
    path = args.output / "download_status.json"
    save(path, status)
    try:
        socket.getaddrinfo("cbs.ic.gatech.edu", 443, type=socket.SOCK_STREAM)
        with urllib.request.urlopen(PUBLISHER, timeout=15) as response:
            page = response.read(2_000_001)
        if len(page) > 2_000_000:
            raise ValueError("Publisher index unexpectedly large")
        links = discover(page.decode("utf-8"))
        save(args.output / "source_links.json", {"publisher": PUBLISHER, "links": links,
                                                "html_sha256": hashlib.sha256(page).hexdigest()})
        budget = {"remaining": int(args.max_gb * 1e9)}
        for name in plan["assets"]:
            cap = 2_000_000 if name in {"readme.md", "video_links.txt"} else 250_000_000
            record = download(links[name], args.output / name, budget, cap)
            if record["cached"]:
                budget["remaining"] -= record["size"]
            if budget["remaining"] < 0:
                raise ValueError("Existing selection exceeds byte budget")
            status["files"].append(record)
            save(path, status)
            print(f"Checked {name}: {record['size']} bytes", flush=True)
        videos = ([("video_clips.tar", links["video_clips.tar"])] if args.videos == "trimmed"
                  else raw_links((args.output / "video_links.txt").read_text()) if args.videos == "raw" else [])
        for name, url in videos:
            destination = args.output / ("raw_videos" if args.videos == "raw" else "archives") / name
            record = download(url, destination, budget)
            if record["cached"]:
                budget["remaining"] -= record["size"]
            if budget["remaining"] < 0:
                raise ValueError("Existing selection exceeds byte budget")
            status["files"].append(record)
            save(path, status)
            print(f"Checked {name}: {record['size']} bytes", flush=True)
        status["status"] = "downloaded_locally_hashed"
        status["note"] = "Read publisher terms/readme. Archives unextracted; alignment and participant audit still required."
        save(path, status)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        dns = isinstance(exc, socket.gaierror) or (isinstance(exc, urllib.error.URLError)
                                                  and isinstance(exc.reason, socket.gaierror))
        status.update(status="blocked_network" if dns else "incomplete", error_type=type(exc).__name__)
        status["note"] = "No research accuracy computed. Completed/partial transfers retained."
        save(path, status)
        print("EGTEA download blocked by DNS/network access." if dns
              else f"EGTEA transfer incomplete ({type(exc).__name__}); inspect source/disk/access.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
