"""Select a bounded convenience pilot before inspecting model outcomes."""
import concurrent.futures
import json
from download_holoassist import ROOT, recording, download_member


if __name__ == '__main__':
    annotations = {r['video_name']: r for r in json.loads((ROOT / 'data-annotation-trainval-v1_1.json').read_text())}
    videos = json.loads((ROOT / 'video_compress.tar.index.json').read_text())['members']
    prefix = json.loads((ROOT / 'video_compress.prefix.index.json').read_text())['members']
    eyes = json.loads((ROOT / 'eyes.tar.index.json').read_text())['members']
    available = {recording(r['name']) for r in eyes} & {recording(r['name']) for r in prefix}
    selected = [r for r in videos if recording(r['name']) in available and recording(r['name']) in annotations
                and ('ram' in recording(r['name']).lower() or 'graphicscard' in recording(r['name']).lower())
                and '&' not in r['name']][:20]
    wanted = {recording(r['name']) for r in selected}
    plan = [('video_compress.tar', selected + [r for r in prefix if recording(r['name']) in wanted]),
            ('eyes.tar', [r for r in eyes if recording(r['name']) in wanted])]
    manifest = {'selection': 'First 20 available RAM/graphics-card videos from bounded archive probe, without model-outcome selection',
                'recordings': sorted(wanted), 'bytes': sum(r['size'] for _, rows in plan for r in rows),
                'user_limit_bytes': 400_000_000_000, 'pilot_limit_bytes': 20_000_000_000,
                'archives': {archive: rows for archive, rows in plan}}
    if not wanted or manifest['bytes'] > manifest['pilot_limit_bytes']:
        raise RuntimeError('Empty or oversized plan')
    (ROOT / 'subset_plan.json').write_text(json.dumps(manifest, indent=2))
    print('Plan', len(wanted), manifest['bytes'], sorted(wanted), flush=True)
    for archive, rows in plan:
        # Small clock/pose files first so gaze audits can run during video transfer.
        rows.sort(key=lambda r: r['size'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda r: download_member(archive, r), rows))
