"""Bounded, resumable HoloAssist HTTP-range subset download.

Reads uncompressed tar headers remotely and downloads selected regular files;
never extracts publisher paths directly. Network byte budget defaults to 20 GB.
"""
import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import tarfile
import threading
import time
import urllib.request
import requests

BASE = 'https://hl2data.z5.web.core.windows.net/holoassist-data-release/'
ROOT = Path(__file__).resolve().parents[1] / 'data/external/holoassist'
LOCK = threading.Lock()
TRANSFERRED = 0
BUDGET = 20_000_000_000
LOCAL = threading.local()
RANGE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=8)


def fetch(url, start, end):
    global TRANSFERRED
    size = end - start + 1
    with LOCK:
        if TRANSFERRED + size > BUDGET:
            raise RuntimeError('Network budget exceeded')
        TRANSFERRED += size
    for attempt in range(4):
        try:
            if not hasattr(LOCAL, 'session'):
                LOCAL.session = requests.Session()
            with LOCAL.session.get(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}, timeout=(8,60), stream=True) as response:
                if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                    raise RuntimeError('Server did not honor byte range; refusing full archive')
                data = response.raw.read(size + 1)
                if len(data) != size:
                    raise RuntimeError('Incomplete range')
                return data
        except Exception:
            if attempt == 3:
                raise
            with LOCK:
                if TRANSFERRED + size > BUDGET:
                    raise RuntimeError('Retry exceeds network budget')
                TRANSFERRED += size
            time.sleep(attempt + 1)


class RemoteTar(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as r:
            self.size = int(r.headers['Content-Length'])
            self.etag = r.headers.get('ETag')
        self.cache_start, self.cache = -1, b''

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b''
        if not (self.cache_start <= self.pos and self.pos + n <= self.cache_start + len(self.cache)):
            self.cache_start = self.pos
            self.cache = fetch(self.url, self.pos, min(self.size - 1, self.pos + max(n, 16384) - 1))
        data = self.cache[self.pos - self.cache_start:self.pos - self.cache_start + n]
        self.pos += len(data)
        return data


def inventory(archive, limit=0, wanted=None):
    dest = ROOT / (archive + '.index.json')
    rows = json.loads(dest.read_text())['members'] if dest.exists() else []
    # Re-scan headers (tiny requests) if the cached index is insufficient.
    complete = json.loads(dest.read_text()).get('complete', False) if dest.exists() else False
    if complete or (limit and len(rows) >= limit):
        return rows
    remote = RemoteTar(BASE + archive)
    old_rows = rows
    rows, found = [], set()
    if old_rows:
        rows = old_rows.copy()
        last = rows[-1]
        remote.seek(last['offset'] + ((last['size'] + 511) // 512) * 512)
        found = {recording(r['name']) for r in rows}
    with tarfile.open(fileobj=remote, mode='r:') as stream:
        for member in stream:
            if not member.isfile():
                continue
            rows.append({'name': member.name, 'size': member.size, 'offset': member.offset_data})
            found.add(recording(member.name))
            if len(rows) % 20 == 0:
                print(archive, len(rows), member.name, flush=True)
                dest.write_text(json.dumps({'url': remote.url, 'etag': remote.etag, 'members': rows}, indent=2))
            if (limit and len(rows) >= limit) or (wanted and wanted <= found and recording(member.name) not in wanted):
                break
        else:
            complete = True
    dest.write_text(json.dumps({'url': remote.url, 'etag': remote.etag, 'complete': complete, 'members': rows}, indent=2))
    return rows


def recording(name):
    parts = name.replace('\\', '/').split('/')
    for part in parts:
        if part.lower().startswith(('r0', 'r1', 'r2', 'z0', 'z1', 'z2')):
            return part
    return parts[0]


def download_member(archive, row):
    # Reject absolute paths, traversal, and link members (inventory is regular only).
    parts = Path(row['name']).parts
    if '..' in parts or Path(row['name']).is_absolute() or ':' in row['name']:
        raise ValueError('Unsafe archive member')
    target = ROOT / 'subset' / row['name']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == row['size']:
        sidecar = target.with_name(target.name + '.sha256')
        if not sidecar.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != sidecar.read_text().strip():
            raise RuntimeError(f'Cached payload integrity/provenance failed: {target}')
        return
    partial = target.with_name(target.name + '.partial')
    existing = partial.stat().st_size if partial.exists() else 0
    if existing > row['size']:
        raise RuntimeError('Oversized partial')
    with partial.open('ab') as out:
        chunk_size = 4 * 1024 * 1024
        # Bounded 24 MiB window also avoids a slow single final connection.
        for window in range(existing, row['size'], 6 * chunk_size):
            ranges = [(row['offset'] + offset, row['offset'] + min(row['size'], offset + chunk_size) - 1)
                      for offset in range(window, min(row['size'], window + 6 * chunk_size), chunk_size)]
            chunks = list(RANGE_POOL.map(lambda bounds: fetch(BASE + archive, *bounds), ranges))
            for chunk in chunks:
                out.write(chunk)
                out.flush()
    partial.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_name(target.name + '.sha256').write_text(digest + '\n')
    print('Downloaded', row['name'], row['size'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', choices=['video_compress.tar', 'eyes.tar', 'cam_info.tar'])
    parser.add_argument('--limit', type=int, default=30)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--recordings', type=int, default=24)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.inventory:
        print(json.dumps(inventory(args.inventory, args.limit)[:5], indent=2))
        return
    videos = inventory('video_compress.tar', 0)
    selected = [r for r in videos if r['name'].lower().endswith('.mp4')][:args.recordings]
    wanted = {recording(row['name']) for row in selected}
    (ROOT / 'selected_recordings.json').write_text(json.dumps(sorted(wanted), indent=2))
    if args.download:
        for archive in ['video_compress.tar', 'eyes.tar', 'cam_info.tar']:
            rows = videos if archive == 'video_compress.tar' else inventory(archive, 0, wanted=wanted)
            matched = [r for r in rows if recording(r['name']) in wanted]
            total = sum(r['size'] for r in matched)
            if total > BUDGET:
                raise RuntimeError('Subset exceeds budget')
            print('Downloading', archive, len(matched), total, flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda r: download_member(archive, r), matched))
    print('Requested transfer bytes including retries:', TRANSFERRED)


if __name__ == '__main__':
    main()
