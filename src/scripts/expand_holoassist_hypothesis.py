"""Expand the pilot without selecting recordings on model outcomes or error counts."""
import concurrent.futures
import json
from download_holoassist import ROOT, recording, download_member


def main():
    annotations={r['video_name']:r for r in json.loads((ROOT/'data-annotation-trainval-v1_1.json').read_text())}
    indices={name:json.loads((ROOT/(name+'.index.json')).read_text())['members']
             for name in ['video_compress.tar','eyes.tar']}
    prefix=json.loads((ROOT/'video_compress.prefix.index.json').read_text())['members']
    old=json.loads((ROOT/'subset_plan.json').read_text())
    old_names=set(old['recordings']);old_groups={n.split('-')[0] for n in old_names}
    available={recording(r['name']) for r in indices['eyes.tar']} & {recording(r['name']) for r in prefix}
    extra=[r for r in indices['video_compress.tar'] if recording(r['name']) in annotations
           and recording(r['name']) in available and recording(r['name']).split('-')[0] not in old_groups]
    wanted=old_names|{recording(r['name']) for r in extra}
    video_rows=[r for r in old['archives']['video_compress.tar'] if r['name'].endswith('.mp4')]+extra
    plan={'selection':'Original 20 plus all 30 remaining annotation/gaze/clock-covered videos in the frozen bounded suffix inventory; exclude old recording-prefix groups from added set; no selection on model outcomes or error count',
          'recordings':sorted(wanted),'original_recordings':sorted(old_names),
          'new_recordings':sorted(wanted-old_names),'user_limit_bytes':400_000_000_000,
          'pilot_limit_bytes':20_000_000_000,
          'archives':{'video_compress.tar':video_rows+[r for r in prefix if recording(r['name']) in wanted],
                      'eyes.tar':[r for r in indices['eyes.tar'] if recording(r['name']) in wanted]}}
    plan['bytes']=sum(r['size'] for rows in plan['archives'].values() for r in rows)
    if len(wanted)!=50 or plan['bytes']>plan['pilot_limit_bytes']:
        raise ValueError('Unexpected candidate coverage or oversized expansion')
    (ROOT/'hypothesis_subset_plan.json').write_text(json.dumps(plan,indent=2))
    print('Combined plan:',len(wanted),'recordings;',plan['bytes'],'bytes',flush=True)
    for archive,rows in plan['archives'].items():
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda r:download_member(archive,r),sorted(rows,key=lambda r:r['size'])))


if __name__=='__main__':
    main()
