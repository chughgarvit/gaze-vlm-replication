"""Audit fresh full-release inputs without selecting by downstream outcome."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np

import run_holoassist_pilot as pilot

BASE=Path.home()/'Datasets/HoloAssist'
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/holoassist_full_study'


def main():
    pilot.DATA=BASE/'study_data'
    plan=json.loads((BASE/'full_study_plan.json').read_text())
    accepted=[];rows=[]
    for name in plan['recordings']:
        try:
            base,gt,angles,valid,pt,pose,duration=pilot.load_recording(name)
            gaze_columns=len((base/'Eyes/Eyes_sync.txt').open().readline().strip().split('\t'))
            if gaze_columns!=9:
                raise ValueError(f'Unsupported gaze column count {gaze_columns}')
            if np.any(np.diff(gt)<=0) or np.any(np.diff(pt)<0):
                raise ValueError('Nonmonotonic source clock')
            if valid.mean()<.5:
                raise ValueError(f'Valid gaze coverage below .5: {valid.mean()}')
            error=pilot.verify_video_clock(base/'Video_compress.mp4')
            cap=cv2.VideoCapture(str(base/'Video_compress.mp4'))
            fps,count=cap.get(cv2.CAP_PROP_FPS),cap.get(cv2.CAP_PROP_FRAME_COUNT);cap.release()
            if fps<=0 or abs(count/fps-duration)>max(.25,2/fps):
                raise ValueError(f'Video/UTC span disagreement: {count/fps if fps else None} vs {duration}')
            accepted.append(name)
            rows.append({'recording':name,'accepted':True,'reason':'','gaze_coverage':float(valid.mean()),
                         'duration_s':duration,'cfr_error_s':error,'fps':fps})
        except Exception as exc:
            rows.append({'recording':name,'accepted':False,'reason':repr(exc)})
            print('AUDIT EXCLUDED',name,repr(exc),flush=True)
        if len(rows)%20==0:
            print('AUDIT',len(rows),'accepted',len(accepted),flush=True)
    pilot.write_csv(OUT/'input_audit.csv',rows)
    (BASE/'full_study_audited_plan.json').write_text(json.dumps({'recordings':accepted,'excluded':
        [r for r in rows if not r['accepted']]},indent=2))
    print('AUDIT COMPLETE',len(accepted),'of',len(rows),flush=True)


if __name__=='__main__':
    main()
