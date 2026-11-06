"""Select an outcome-independent fresh study and extract original local assets."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import zipfile

BASE=Path.home()/'Datasets/HoloAssist'
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/holoassist_full_study'


def recording(name):
    return next((p for p in Path(name).parts if p.startswith(('R','z'))),'')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    annotations={r['video_name']:r for r in json.loads((BASE/'data-annotation-trainval-v1_1.json').read_text())}
    indices={archive:json.loads((BASE/'inventory'/(archive+'.json')).read_text())
             for archive in ['video_compress.tar','eyes.tar','cam_info.tar']}
    videos={recording(r['name']):r for r in indices['video_compress.tar'] if r['name'].endswith('Video_compress.mp4')}
    eyes={recording(r['name']):r for r in indices['eyes.tar'] if r['name'].endswith('Eyes_sync.txt')}
    camera={recording(r['name']) for r in indices['cam_info.tar'] if r['name'].endswith('Video/Intrinsics.txt')}
    prior=json.loads((ROOT/'results/holoassist_hypothesis/selection.json').read_text())['selected']
    excluded={r.split('-')[0] for r in prior}
    with zipfile.ZipFile(BASE/'data-splits-v1_2.zip') as z:
        official={v.strip():n.split('-')[0] for n in ['train-v1_2.txt','val-v1_2.txt'] for v in z.read(n).decode().splitlines()}
    buckets={}
    for name,r in annotations.items():
        if name in videos and name in eyes and name in camera and name in official and name.split('-')[0] not in excluded:
            buckets.setdefault(r['taskType'],[]).append(name)
    rng=random.Random(20261017);selected=[];coverage={}
    for task,names in sorted(buckets.items()):
        names=sorted(names);rng.shuffle(names)
        chosen=names[:12];selected+=chosen;coverage[task]={'available':len(names),'selected':len(chosen)}
    selected=sorted(selected)
    wanted=set(selected)
    records=[]
    for archive,rows in indices.items():
        for row in rows:
            name=recording(row['name'])
            if name not in wanted:
                continue
            path=row['name'].replace('\\','/')
            if (path.endswith('Video_compress.mp4') or '/Eyes/' in path or
                '/Video/' in path and Path(path).name in ['Pose_sync.txt','VideoMp4Timing.txt','Intrinsics.txt']):
                records.append({'archive':archive,**row})
    # Duplicate pose/timing entries occur in the video and calibration archives;
    # prefer calibration for intrinsics, video archive for its own pose/clock.
    unique={}
    for row in records:
        key=row['name'].lstrip('./')
        if key not in unique:
            unique[key]=row
    records=list(unique.values())
    policy={'seed':20261017,'per_task':12,'selection':'Uniform seeded sample within task; availability and known train/val membership only; no selection on error labels, duration, or model outcome',
            'excluded_prior_recording_groups':sorted(excluded),'coverage':coverage,'recordings':selected,
            'bytes':sum(r['bytes'] for r in records),'groups':len({n.split('-')[0] for n in selected}),
            'source':'Full downloaded publisher archives','files':records}
    plan=BASE/'full_study_plan.json';plan.write_text(json.dumps(policy,indent=2))
    (OUT/'selection_plan.json').write_text(json.dumps(policy,indent=2))
    destination=BASE/'study_data';destination.mkdir(exist_ok=True)
    shutil.copyfile(BASE/'data-annotation-trainval-v1_1.json',destination/'data-annotation-trainval-v1_1.json')
    print('SELECTED',len(selected),'recordings',policy['groups'],'groups',len(coverage),'tasks',round(policy['bytes']/1e9,2),'GB',flush=True)
    def extract(row):
        parts=Path(row['name']).parts
        if '..' in parts or Path(row['name']).is_absolute() or ':' in row['name']:
            raise ValueError('Unsafe source path')
        target=destination/'subset'/row['name'].lstrip('./');target.parent.mkdir(parents=True,exist_ok=True)
        receipt=target.with_name(target.name+'.sha256')
        if target.exists() and target.stat().st_size==row['bytes'] and receipt.exists():
            return row['bytes']
        temporary=target.with_name(target.name+'.extracting');digest=hashlib.sha256()
        with (BASE/row['archive']).open('rb') as source,temporary.open('wb') as output:
            source.seek(row['offset']);remaining=row['bytes']
            while remaining:
                block=source.read(min(4*1024*1024,remaining))
                if not block:
                    raise ValueError('Unexpected archive EOF')
                digest.update(block);output.write(block);remaining-=len(block)
        temporary.replace(target);receipt.write_text(digest.hexdigest()+'\n')
        return row['bytes']
    done=0;start=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,size in enumerate(pool.map(extract,records),1):
            done+=size
            if i%25==0 or i==len(records):
                status={'stage':'extracting' if i<len(records) else 'complete','files':i,'total_files':len(records),
                        'bytes':done,'total_bytes':policy['bytes'],'elapsed_seconds':time.time()-start}
                (OUT/'extraction_status.json').write_text(json.dumps(status,indent=2))
                print(status,flush=True)
    (destination/'subset_plan.json').write_text(json.dumps({'recordings':selected},indent=2))
    print('EXTRACTION COMPLETE',flush=True)


if __name__=='__main__':
    main()
