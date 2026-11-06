"""Exploratory recording-group-disjoint HoloAssist probes on measured gaze/RGB.

Completed annotated segments (oracle boundaries), frozen ImageNet ResNet18,
linear classifiers. This does not implement VLM transfer or live efficiency.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import sys
import zipfile

import cv2
import av
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/external/holoassist'
OUT = ROOT / 'results/holoassist_pilot'
AXIS = np.linalg.inv(np.array([[0,0,1],[-1,0,0],[0,-1,0]], float))
VOCAB = ['grab', 'place', 'withdraw', 'insert', 'lift', 'push', 'assemble', 'press']


def verify_video_clock(video):
    with av.open(str(video)) as container:
        stream=container.streams.video[0]
        pts=np.sort(np.array([float(packet.pts*stream.time_base) for packet in container.demux(stream)
                              if packet.pts is not None]))
        fps=float(stream.average_rate)
    if not len(pts):
        raise ValueError('Missing video presentation timestamps')
    deviation=float(np.max(np.abs(pts-np.arange(len(pts))/fps)))
    if deviation > .0001:
        raise ValueError(f'Non-CFR video requires a PTS-based extractor: {video}, deviation={deviation}')
    return deviation


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def load_recording(name):
    base = DATA / 'subset' / name / 'Export_py'
    gaze = np.loadtxt(base / 'Eyes/Eyes_sync.txt', delimiter='\t')
    pose = np.loadtxt(base / 'Video/Pose_sync.txt', delimiter='\t')
    start, end = [int(x) for x in (base / 'Video/VideoMp4Timing.txt').read_text().split()]
    # Subtract the .NET tick epoch before converting to seconds. Float64 input
    # precision at this epoch is ~13 us, much less than the 100 ms age threshold.
    gt = (gaze[:,1] - start) / 1e7
    pt = (pose[:,1] - start) / 1e7
    # Synced files repeat sensor readings: keep one row per source timestamp.
    _, index = np.unique(gt, return_index=True)
    gaze, gt = gaze[index], gt[index]
    pi = np.searchsorted(pt, gt, side='right') - 1
    safe = np.maximum(pi, 0)
    rotations = pose[safe, 2:].reshape(-1,4,4)[:,:3,:3]
    direction = np.einsum('nji,nj->ni', rotations, gaze[:,5:8]) @ AXIS.T
    norm = np.linalg.norm(direction, axis=1)
    valid = (gaze[:,-1] > 0) & (norm > .5) & np.isfinite(direction).all(1) & (pi >= 0) & (gt - pt[safe] <= .1)
    direction /= np.maximum(norm[:,None], 1e-8)
    angles = np.column_stack([np.arctan2(direction[:,0], direction[:,2]),
                              np.arctan2(direction[:,1], np.hypot(direction[:,0], direction[:,2]))])
    return base, gt, angles, valid, pt, pose[:,2:].reshape(-1,4,4), (end-start)/1e7


def gaze_summary(t, angles, valid, start, end, shuffled=False, seed=0):
    left,right=np.searchsorted(t,[start,end],side='left')
    right=np.searchsorted(t,end,side='right')
    t, angles, valid = t[left:right], angles[left:right].copy(), valid[left:right]
    quality = np.array([valid.mean() if len(t) else 0, np.log1p(len(t)),
                        (t[-1]-t[0]) / max(end-start,1e-6) if len(t)>1 else 0])
    static, dynamic = np.zeros(8), np.zeros(18)
    if valid.sum() < 3:
        return quality, static, dynamic
    if shuffled:
        rng = np.random.default_rng(seed)
        angles[valid] = angles[valid][rng.permutation(valid.sum())]
    a = angles[valid]
    quantiles=np.quantile(a,[.1,.9],axis=0)
    static[:] = np.r_[a.mean(0), a.std(0), quantiles[0], quantiles[1]]
    dt = np.diff(t)
    pair = valid[:-1] & valid[1:] & (dt > 0) & (dt <= .1)
    delta = np.arctan2(np.sin(np.diff(angles,axis=0)), np.cos(np.diff(angles,axis=0)))
    speed = np.linalg.norm(delta[pair],axis=1) / dt[pair]
    if len(speed):
        dynamic[:6] = [speed.mean(), speed.std(), *np.quantile(speed,[.25,.5,.75,.95])]
        # Angular low-speed proportion, NOT an inferred physiological fixation.
        dynamic[6:8] = [(speed < .1).mean(), np.linalg.norm(delta[pair],axis=1).sum()]
    for b in range(3):
        selected = valid & (t >= start+(end-start)*b/3) & (t < start+(end-start)*(b+1)/3)
        if selected.any():
            dynamic[8+b*2:10+b*2] = angles[selected].mean(0)
    dynamic[14:16] = a[-1]-a[0]
    dynamic[16:18] = np.mean(np.abs(np.diff(a,axis=0)),axis=0)
    return quality, static, np.nan_to_num(dynamic)


def build_features(gaze_only=False, plan=None, frame_budget_features=False, available_only=False, device='cpu'):
    selected = json.loads(Path(plan or DATA/'subset_plan.json').read_text())['recordings']
    planned=selected.copy()
    if available_only:
        selected=[name for name in selected if all((DATA/'subset'/name/'Export_py'/part).exists()
                  for part in ['Video_compress.mp4','Eyes/Eyes_sync.txt','Video/Pose_sync.txt','Video/VideoMp4Timing.txt'])]
    annotations = {r['video_name']:r for r in json.loads((DATA/'data-annotation-trainval-v1_1.json').read_text())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'selection.json').write_text(json.dumps({'available_only':available_only,'selected':selected,
        'planned':planned,'excluded':sorted(set(planned)-set(selected))},indent=2))
    cache = OUT/'features'
    cache.mkdir(exist_ok=True)
    encoder = transform = torch = None
    if not gaze_only:
        import torch
        from torchvision.models import resnet18, ResNet18_Weights
        torch.set_num_threads(4)
        weights = ResNet18_Weights.IMAGENET1K_V1
        encoder = resnet18(weights=weights)
        encoder.fc = torch.nn.Identity()
        encoder.eval().to(device)
        transform = weights.transforms()
    rows, audits = [], []
    arrays = {k:[] for k in ['visual','quality','static','dynamic','shuffled','motion','duration']}
    if frame_budget_features:
        arrays['visual_one']=[]
    if not gaze_only:
        selected.sort(key=lambda name:not (DATA/'subset'/name/'Export_py/Video_compress.mp4').exists())
    for name in selected:
        base, gt, angles, valid, pt, poses, duration = load_recording(name)
        events = [e for e in annotations[name]['events'] if e['label']=='Fine grained action'
                  and e['end'] > e['start'] and e['start'] >= 0 and e['end'] <= duration+.1]
        video = base/'Video_compress.mp4'
        frame_cache = cache/(name+'.npz')
        wait_start=time.time()
        while not gaze_only and not video.exists():
            if time.time()-wait_start>1800:
                raise FileNotFoundError(f'Video download incomplete after bounded wait: {video}')
            print('Waiting for video transfer:',name,flush=True)
            time.sleep(20)
        fps = count = 0
        if not gaze_only:
            cfr_error=verify_video_clock(video)
            cap = cv2.VideoCapture(str(video))
            fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Publisher timestamps and video must agree within 250 ms or 2 frames.
            if not fps or abs(count/fps-duration) > max(.25,2/fps):
                raise ValueError(f'Clock/video duration disagreement: {name}, {count/fps}, {duration}')
            cached = np.load(frame_cache) if frame_cache.exists() else None
            if cached is not None and (not frame_budget_features or 'frames' in cached):
                saved = cached
                event_features = saved['visual']
                event_frames = saved['frames'] if 'frames' in saved else None
                if saved['event_ids'].tolist() != [str(e['id']) for e in events]:
                    raise ValueError('Feature cache/annotation mismatch')
            else:
                frames = sorted({min(count-1,int((e['start']+(e['end']-e['start'])*f)*fps))
                                 for e in events for f in [.1,.35,.6,.85]})
                features = {}
                for offset in range(0,len(frames),32):
                    batch = frames[offset:offset+32]
                    images = []
                    for frame in batch:
                        cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
                        ok, image = cap.read()
                        if not ok:
                            raise RuntimeError(f'Decode failure {name}:{frame}')
                        images.append(transform(torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).permute(2,0,1)))
                    with torch.inference_mode():
                        embedding = encoder(torch.stack(images).to(device)).cpu().numpy()
                    features.update(zip(batch,embedding))
                event_features=[]
                event_frames=[]
                for e in events:
                    feats=np.array([features[min(count-1,int((e['start']+(e['end']-e['start'])*f)*fps))] for f in [.1,.35,.6,.85]])
                    event_features.append(np.r_[feats.mean(0),feats.std(0),feats[-1]-feats[0]])
                    event_frames.append(feats)
                event_features=np.array(event_features)
                event_frames=np.array(event_frames)
                np.savez_compressed(frame_cache,visual=event_features,frames=event_frames,event_ids=np.array([str(e['id']) for e in events]))
            cap.release()
        for i,e in enumerate(events):
            q,s,d = gaze_summary(gt,angles,valid,e['start'],e['end'])
            _,_,shuffle = gaze_summary(gt,angles,valid,e['start'],e['end'],True,seed=1000+i)
            p=(pt>=e['start'])&(pt<=e['end'])
            m=np.zeros(4)
            if p.sum()>1:
                pp=poses[p]
                movement=np.linalg.norm(np.diff(pp[:,:3,3],axis=0),axis=1)
                rotation=np.linalg.norm(np.diff(pp[:,:3,:3],axis=0),axis=(1,2))
                m[:]=[movement.mean(),movement.sum(),rotation.mean(),rotation.sum()]
            correctness=e['attributes'].get('Action Correctness','')
            row={'id':name+':'+str(e['id']),'recording':name,'group':name.split('-')[0],
                 'verb':e['attributes'].get('Verb',''),'noun':e['attributes'].get('Noun',''),
                 'task':annotations[name]['taskType'],'start':e['start'],'end':e['end'],
                 'mistake':1 if correctness.startswith('Wrong Action') else 0 if correctness=='Correct Action' else -1}
            rows.append(row)
            values=[np.zeros(1) if gaze_only else event_features[i],q,s,d,shuffle,m,[np.log1p(e['end']-e['start'])]]
            for key,value in zip(['visual','quality','static','dynamic','shuffled','motion','duration'],values):
                arrays[key].append(value)
            if frame_budget_features:
                if gaze_only:
                    raise ValueError('Frame budgets require RGB')
                arrays['visual_one'].append(np.r_[event_frames[i,1],np.zeros(1024)])
        audits.append({'recording':name,'events':len(events),'duration_s':duration,'gaze_samples':len(gt),
                       'valid_gaze_fraction':float(valid.mean()),'camera_clock_start_s':float(pt[0]),
                       'gaze_clock_start_s':float(gt[0]),'fps':fps,'video_frames':count,
                       'max_cfr_error_s':cfr_error if not gaze_only else None,
                       'video_duration_disagreement_s':abs(count/fps-duration) if fps else None})
        print('Features',name,len(events),'valid gaze',round(valid.mean(),3),flush=True)
    write_csv(OUT/('gaze_audit.csv' if gaze_only else 'alignment_audit.csv'),audits)
    write_csv(OUT/'segments.csv',rows)
    return rows,{k:np.nan_to_num(np.asarray(v,dtype=np.float64)) for k,v in arrays.items()}


def score(y, prediction, probability, task):
    values={'accuracy':accuracy_score(y,prediction),'macro_f1':f1_score(y,prediction,average='macro',zero_division=0),
            'balanced_accuracy':balanced_accuracy_score(y,prediction)}
    if task=='mistake':
        values['average_precision']=average_precision_score(y,probability)
    return {k:float(v) for k,v in values.items()}


def evaluate(rows, arrays, gaze_only=False):
    metrics, predictions, splits=[],[],[]
    for task in ['verb','mistake']:
        indices=np.array([i for i,r in enumerate(rows) if (r['verb'] in VOCAB if task=='verb' else r['mistake']>=0)])
        labels=np.array([VOCAB.index(rows[i]['verb']) if task=='verb' else rows[i]['mistake'] for i in indices])
        groups=np.array([rows[i]['group'] for i in indices])
        modes=['prior','gaze_static','gaze_temporal'] if gaze_only else ['prior','visual','visual_quality_motion','gaze_static','gaze_temporal','fusion_static','fusion_temporal','fusion_shuffled']
        for mode in modes:
            pred=np.full(len(indices),-1)
            prob=np.zeros(len(indices))
            for fold,(tr,te) in enumerate(GroupKFold(n_splits=4).split(indices,labels,groups)):
                train,test=indices[tr],indices[te]
                if mode==modes[0]:
                    splits.append({'task':task,'fold':fold,'train_groups':sorted(set(groups[tr])),
                                   'test_groups':sorted(set(groups[te])),'train_segments':len(tr),'test_segments':len(te),
                                   'train_mistakes':int(labels[tr].sum()) if task=='mistake' else None,
                                   'test_mistakes':int(labels[te].sum()) if task=='mistake' else None})
                if mode=='prior':
                    pred[te]=np.bincount(labels[tr]).argmax()
                    prob[te]=labels[tr].mean() if task=='mistake' else 0
                    continue
                parts=['duration']
                if mode.startswith(('visual','fusion')):
                    pca=PCA(n_components=min(32,len(tr)-1),random_state=0,svd_solver='randomized')
                    vistr=pca.fit_transform(arrays['visual'][train])
                    viste=pca.transform(arrays['visual'][test])
                else:
                    vistr=np.empty((len(tr),0));viste=np.empty((len(te),0))
                if mode!='visual':
                    parts+=['quality']
                if mode in ['visual_quality_motion','fusion_static','fusion_temporal','fusion_shuffled']:
                    parts+=['motion']
                if mode in ['gaze_static','gaze_temporal','fusion_static','fusion_temporal','fusion_shuffled']:
                    parts+=['static']
                if mode in ['gaze_temporal','fusion_temporal']:
                    parts+=['dynamic']
                if mode=='fusion_shuffled':
                    parts+=['shuffled']
                xtr=np.concatenate([vistr]+[arrays[k][train] for k in parts],axis=1)
                xte=np.concatenate([viste]+[arrays[k][test] for k in parts],axis=1)
                model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,class_weight='balanced',random_state=0))
                model.fit(xtr,labels[tr])
                pred[te]=model.predict(xte)
                if task=='mistake':
                    prob[te]=model.predict_proba(xte)[:,list(model[-1].classes_).index(1)]
            metrics.append({'task':task,'mode':mode,'segments':len(indices),'groups':len(set(groups)),**score(labels,pred,prob,task)})
            for j,index in enumerate(indices):
                predictions.append({'task':task,'mode':mode,'id':rows[index]['id'],'group':rows[index]['group'],
                                    'label':int(labels[j]),'prediction':int(pred[j]),'mistake_probability':float(prob[j])})
            print(metrics[-1],flush=True)
    write_csv(OUT/('gaze_metrics.csv' if gaze_only else 'metrics.csv'),metrics)
    write_csv(OUT/('gaze_predictions.csv' if gaze_only else 'predictions.csv'),predictions)
    (OUT/'splits.json').write_text(json.dumps(splits,indent=2))
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gaze-only',action='store_true')
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--frame-budget-features',action='store_true')
    parser.add_argument('--extract-only',action='store_true')
    parser.add_argument('--available-only',action='store_true')
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    args=parser.parse_args()
    if args.output:
        OUT=args.output.resolve()
    if args.data_root:
        DATA=args.data_root.resolve()
    started=time.time()
    rows,arrays=build_features(args.gaze_only,args.plan,args.frame_budget_features,args.available_only,args.device)
    np.savez_compressed(OUT/('gaze_features.npz' if args.gaze_only else 'features.npz'),**arrays)
    if args.extract_only:
        (OUT/'extraction.json').write_text(json.dumps({'synthetic':False,'recordings':len({r['recording'] for r in rows}),
            'segments':len(rows),'elapsed_seconds':time.time()-started,'frame_budget_features':args.frame_budget_features},indent=2))
        sys.exit(0)
    result=evaluate(rows,arrays,args.gaze_only)
    (OUT/('gaze_run.json' if args.gaze_only else 'run.json')).write_text(json.dumps({
        'elapsed_seconds':time.time()-started,'synthetic':False,'vocab':VOCAB,'metrics':result,
        'protocol':'4-fold recording-prefix-group-disjoint CV; completed annotated segments; fixed C=1; no tuning',
        'limitations':['Convenience subset of two task domains','Recording prefix is not verified participant identity',
                       'Only 26 errors before clock audit','Not official benchmark reproduction','No symbolic/VLM transfer',
                       'Frozen ImageNet scene features; camera-relative angular gaze, not fixation/object revisitation',
                       'Offline feature extraction time is not device efficiency','Oracle segment boundaries',
                       'Exploratory CV; hyperparameters not optimized; model capacities differ']},indent=2))
