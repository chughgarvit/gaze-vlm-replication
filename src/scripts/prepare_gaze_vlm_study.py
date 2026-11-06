"""Extract all labeled train/val recordings and construct fixed causal windows."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gaze_transfer.data import Recording, json_write, stable_seed, window_targets

BASE = Path.home() / 'Datasets/HoloAssist'
OUT = BASE / 'gaze_vlm_study'


def recording(name):
    return next((p for p in Path(name).parts if p.startswith(('R', 'z'))), '')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    annotations = {r['video_name']: r for r in json.loads((BASE/'data-annotation-trainval-v1_1.json').read_text())}
    indices = {a: json.loads((BASE/'inventory'/(a+'.json')).read_text())
               for a in ['video_compress.tar', 'eyes.tar', 'cam_info.tar']}
    with zipfile.ZipFile(BASE/'data-splits-v1_2.zip') as z:
        official = {n: [v.strip() for v in z.read(n+'-v1_2.txt').decode().splitlines() if v.strip()]
                    for n in ['train', 'val']}
    required = ['Video_compress.mp4', 'Eyes/Eyes_sync.txt', 'Video/Pose_sync.txt',
                'Video/VideoMp4Timing.txt', 'Video/Intrinsics.txt']
    assets = {}
    for archive, entries in indices.items():
        for e in entries:
            name = recording(e['name'])
            if name in annotations and any(e['name'].replace('\\', '/').endswith('/'+p) for p in required):
                assets.setdefault(name, {}).setdefault(e['name'].lstrip('./'), {'archive': archive, **e})
    eligible = {n for n, a in assets.items() if all(any(e['name'].replace('\\', '/').endswith('/'+p)
                for e in a.values()) for p in required)}
    test = sorted(set(official['val']) & eligible)
    test_groups = {n.split('-')[0] for n in test}
    train_pool = sorted(n for n in set(official['train']) & eligible if n.split('-')[0] not in test_groups)
    groups = sorted({n.split('-')[0] for n in train_pool}, key=lambda s: stable_seed('dev20261018'+s))
    dev_groups = set(groups[:max(1, len(groups)//10)])
    dev = [n for n in train_pool if n.split('-')[0] in dev_groups]
    train = [n for n in train_pool if n.split('-')[0] not in dev_groups]
    splits = {'train': train, 'dev': dev, 'test': test}
    selected = set(train+dev+test)
    jobs = [e for n in sorted(selected) for e in assets[n].values()]
    policy = {'created': '2026-10-18', 'splits': splits, 'official_val_role': 'locked test',
              'development': 'deterministic 10% train-prefix groups; test-overlapping prefixes excluded',
              'identity_limit': 'Recording prefix is not verified wearer identity',
              'window_s': 5., 'stride_s': 5., 'anticipation_horizons_s': [2., 5.],
              'frames_per_window': 4, 'gaze_samples': 64, 'projection_depth_m': .5,
              'eligible': len(eligible), 'selected': len(selected),
              'excluded': sorted(set(annotations)-selected), 'extraction_bytes': sum(j['bytes'] for j in jobs)}
    json_write(OUT/'protocol.json', policy)
    shutil.copyfile(ROOT/'GAZE_VLM_STUDY.md', OUT/'GAZE_VLM_STUDY.md')
    print('SPLITS', {k: len(v) for k, v in splits.items()}, 'GB', policy['extraction_bytes']/1e9, flush=True)
    destination = BASE/'study_data/subset'
    def extract(e):
        relative = Path(e['name'].lstrip('./'))
        if relative.is_absolute() or '..' in relative.parts or ':' in str(relative):
            raise ValueError('Unsafe archive path')
        target = (destination/relative).resolve()
        # Windows may return the extended-length prefix transiently for an open file.
        checked = Path(str(target).removeprefix('\\\\?\\'))
        checked_root = Path(str(destination.resolve()).removeprefix('\\\\?\\'))
        if not checked.is_relative_to(checked_root):
            raise ValueError(f'Extraction escaped data root: {target!s} versus {destination.resolve()!s}')
        receipt = target.with_name(target.name+'.sha256')
        if target.exists() and target.stat().st_size == e['bytes'] and receipt.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name+'.extracting')
        digest = hashlib.sha256()
        with (BASE/e['archive']).open('rb') as source, temp.open('wb') as output:
            source.seek(e['offset']); remaining = e['bytes']
            while remaining:
                block = source.read(min(8*1024**2, remaining))
                if not block:
                    raise ValueError('Unexpected EOF')
                output.write(block); digest.update(block); remaining -= len(block)
        temp.replace(target); receipt.write_text(digest.hexdigest()+'\n')
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(extract, jobs), 1):
            if i % 100 == 0 or i == len(jobs):
                json_write(OUT/'prepare_status.json', {'stage': 'extract', 'done': i, 'total': len(jobs), 'elapsed_s': time.time()-started})
                print('EXTRACT', i, '/', len(jobs), flush=True)
    audits, total = [], 0
    for split, names in splits.items():
        manifest = OUT/(split+'.jsonl')
        temporary = manifest.with_suffix('.jsonl.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            for i, name in enumerate(names):
                try:
                    rec = Recording(destination/name/'Export_py')
                    count = 0
                    try:
                        for cutoff in range(5, int(rec.duration)-5+1, 5):
                            start = cutoff-5.
                            row = {'id': f'{name}@{cutoff:06d}', 'recording': name, 'group': name.split('-')[0],
                                   'split': split, 'start': start, 'cutoff': float(cutoff),
                                   'targets': {str(h): window_targets(annotations[name], start, cutoff, h)
                                               for h in [2, 5]}}
                            stream.write(json.dumps(row, ensure_ascii=False)+'\n'); count += 1
                        audits.append({'recording': name, 'split': split, 'accepted': True,
                                       'windows': count, 'duration': rec.duration})
                        total += count
                    finally:
                        rec.close()
                except Exception as exc:
                    audits.append({'recording': name, 'split': split, 'accepted': False, 'error': repr(exc)})
                if (i+1) % 20 == 0:
                    print('WINDOWS', split, i+1, '/', len(names), total, flush=True)
        temporary.replace(manifest)
    json_write(OUT/'audit.json', audits)
    json_write(OUT/'prepare_status.json', {'stage': 'complete', 'windows': total,
               'accepted_recordings': sum(a['accepted'] for a in audits), 'elapsed_s': time.time()-started})
    print('PREPARE COMPLETE', total, flush=True)


if __name__ == '__main__':
    main()
