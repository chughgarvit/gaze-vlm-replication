"""Pin and download the compact student, offline teacher, and reference sources."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
BASE = Path.home() / 'Datasets/HoloAssist/gaze_vlm_study'
ROOT = Path(__file__).resolve().parents[1]


def bounded_download(repo, revision, item, destination):
    """Bounded HTTP ranges avoid a stalled multi-GB streaming connection."""
    import requests
    target = destination/item.rfilename
    if target.exists() and target.stat().st_size == item.size:
        return
    chunks = BASE/'download_chunks'/repo.split('/')[-1]/item.rfilename
    chunks.mkdir(parents=True, exist_ok=True)
    block_size = 16*1024**2
    ranges = [(i, min(i+block_size, item.size)-1) for i in range(0, item.size, block_size)]
    def get(bounds):
        start, end = bounds
        path = chunks/str(start)
        if path.exists() and path.stat().st_size == end-start+1:
            return
        for attempt in range(5):
            try:
                url = f'https://huggingface.co/{repo}/resolve/{revision}/{item.rfilename}?range_start={start}&attempt={attempt}'
                with requests.get(url, headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(15, 30)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                        raise ValueError('Server did not honor exact range')
                    temporary = path.with_suffix('.tmp')
                    with temporary.open('wb') as output:
                        for block in response.iter_content(1024**2):
                            output.write(block)
                if temporary.stat().st_size != end-start+1:
                    raise ValueError('Incomplete range')
                temporary.replace(path)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2, attempt+1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(get, ranges), 1):
            if i % 8 == 0 or i == len(ranges):
                print('RANGES', item.rfilename, i, '/', len(ranges), flush=True)
                (BASE/'model_download_status.json').write_text(json.dumps({'file': item.rfilename,
                      'ranges_done_in_order': i, 'ranges_total': len(ranges)}))
    temporary = target.with_suffix('.assembling')
    digest = hashlib.sha256()
    with temporary.open('wb') as output:
        for start, end in ranges:
            with (chunks/str(start)).open('rb') as source:
                for block in iter(lambda: source.read(8*1024**2), b''):
                    digest.update(block); output.write(block)
    if item.lfs and digest.hexdigest() != item.lfs.sha256:
        raise ValueError('Publisher SHA-256 mismatch')
    temporary.replace(target)


def main():
    from huggingface_hub import HfApi, snapshot_download
    BASE.mkdir(parents=True, exist_ok=True)
    records = []
    for role, repo in [('student', 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'),
                       ('teacher', 'HuggingFaceTB/SmolVLM2-2.2B-Instruct')]:
        info = HfApi().model_info(repo, files_metadata=True)
        destination = BASE / 'models' / role
        print('MODEL', role, repo, info.sha, flush=True)
        snapshot_download(repo, revision=info.sha, local_dir=destination,
                          allow_patterns=['*.json', '*.txt', '*.model', '*.md', 'LICENSE'],
                          max_workers=4)
        for item in info.siblings:
            if item.rfilename.endswith('.safetensors'):
                bounded_download(repo, info.sha, item, destination)
        files = []
        for file in destination.iterdir():
            if file.is_file():
                h = hashlib.sha256()
                with file.open('rb') as stream:
                    for block in iter(lambda: stream.read(8 * 1024**2), b''):
                        h.update(block)
                files.append({'name': file.name, 'bytes': file.stat().st_size, 'sha256': h.hexdigest()})
        records.append({'role': role, 'repo': repo, 'revision': info.sha, 'files': files})
        (BASE / 'model_provenance.json').write_text(json.dumps(records, indent=2))
        print('VERIFIED', role, sum(f['bytes'] for f in files), flush=True)
    reference = ROOT / 'references/gaze_vlm'
    tree = json.loads((reference / 'tree.json').read_text())
    def fetch(entry):
        target = reference / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve('https://raw.githubusercontent.com/anupampani/Gaze-VLM/' +
                                   tree['sha'] + '/' + entry['path'], target)
    files = [x for x in tree['tree'] if x['type'] == 'blob' and
             (x['path'].endswith(('.py', '.md')) or x['path'] == 'LICENSE')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, files))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
