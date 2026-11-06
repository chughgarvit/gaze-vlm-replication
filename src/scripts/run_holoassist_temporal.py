"""Capacity-matched scene/gaze GRU pilots with strictly held-out recording groups."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,f1_score
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

import run_holoassist_pilot as pilot

ROOT=Path(__file__).resolve().parents[1]
BASE=Path.home()/'Datasets/HoloAssist'
MODES=['visual_control','static_gaze','temporal_gaze','shuffled_gaze','gaze_only']


class JointEncoder(nn.Module):
    def __init__(self,dimensions):
        super().__init__()
        self.projection=nn.Sequential(nn.Linear(dimensions,64),nn.ReLU())
        self.gru=nn.GRU(64,64,batch_first=True)
        self.action=nn.Linear(64,8);self.error=nn.Linear(64,2)

    def forward(self,x):
        _,hidden=self.gru(self.projection(x));h=hidden[-1]
        return self.action(h),self.error(h)


def motion_summary(pt,poses,start,end):
    chosen=poses[(pt>=start)&(pt<=end)]
    if len(chosen)<2:
        return np.zeros(4)
    finite=np.isfinite(chosen).all(axis=(1,2));pair=finite[:-1]&finite[1:]
    if not pair.any():
        return np.zeros(4)
    movement=np.linalg.norm(np.diff(chosen[:,:3,3],axis=0),axis=1)[pair]
    rotation=np.linalg.norm(np.diff(chosen[:,:3,:3],axis=0),axis=(1,2))[pair]
    return np.array([movement.mean(),movement.sum(),rotation.mean(),rotation.sum()])


def sequence_data(rows,output):
    cache=output/'temporal_inputs.npz'
    ids=np.array([r['id'] for r in rows])
    if cache.exists():
        with np.load(cache) as data:
            if not np.array_equal(data['ids'],ids):
                raise ValueError('Temporal input cache ID mismatch')
            return {key:data[key] for key in data.files if key!='ids'}
    pilot.DATA=BASE/'study_data'
    common=np.zeros((len(rows),4,9),np.float32)
    static=np.zeros((len(rows),8),np.float32)
    temporal=np.zeros((len(rows),4,26),np.float32)
    shuffled=np.zeros_like(temporal)
    visual=np.zeros((len(rows),4,512),np.float32)
    groups={}
    for i,row in enumerate(rows):
        groups.setdefault(row['recording'],[]).append(i)
    for count,(name,indices) in enumerate(groups.items(),1):
        base,gt,angles,valid,pt,poses,duration=pilot.load_recording(name)
        with np.load(output/'features'/(name+'.npz')) as data:
            frames=data['frames'];lookup={str(k):i for i,k in enumerate(data['event_ids'])}
        for i in indices:
            row=rows[i];start,end=float(row['start']),float(row['end']);span=end-start
            fractions=[0,.1,.35,.6,.85];cutoff=start+.85*span
            _,static[i],_=pilot.gaze_summary(gt,angles,valid,start,cutoff)
            permuted=angles.copy();mask=(gt>=start)&(gt<=cutoff)&valid
            seed=int.from_bytes(hashlib.sha256(row['id'].encode()).digest()[:8],'little')
            rng=np.random.default_rng(seed);permuted[mask]=permuted[mask][rng.permutation(mask.sum())]
            for step in range(4):
                lo=start+fractions[step]*span;hi=start+fractions[step+1]*span
                q,s,d=pilot.gaze_summary(gt,angles,valid,lo,hi)
                _,ss,dd=pilot.gaze_summary(gt,permuted,valid,lo,hi)
                common[i,step]=np.r_[q,motion_summary(pt,poses,lo,hi),np.log1p(span),fractions[step+1]]
                temporal[i,step]=np.r_[s,d];shuffled[i,step]=np.r_[ss,dd]
            event_id=row['id'].rsplit(':',1)[1]
            visual[i]=frames[lookup[event_id]]
        if count%20==0:
            print('TEMPORAL INPUTS',count,'recordings',flush=True)
    result={'visual':visual,'common':common,'static':static,'temporal':temporal,'shuffled':shuffled}
    if any(not np.isfinite(x).all() for x in result.values()):
        raise ValueError('Nonfinite temporal inputs')
    np.savez_compressed(cache,ids=ids,**result)
    return result


def loss_function(outputs,action,error,action_weights,error_weights):
    total=outputs[0].sum()*0
    amask=action>=0;emask=error>=0
    if amask.any():
        total=total+nn.functional.cross_entropy(outputs[0][amask],action[amask],weight=action_weights)
    if emask.any():
        total=total+nn.functional.cross_entropy(outputs[1][emask],error[emask],weight=error_weights)
    return total


def evaluate(model,x,indices,device):
    action=[];error=[]
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(indices),256):
            a,e=model(x[indices[start:start+256]].to(device))
            action.append(a.softmax(1).cpu().numpy());error.append(e.softmax(1).cpu().numpy()[:,1])
    return np.concatenate(action),np.concatenate(error)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/holoassist_full_study')
    parser.add_argument('--epochs',type=int,default=25);parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    args=parser.parse_args();output=args.output;started=time.time();torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu';print('NEURAL DEVICE',device,flush=True)
    rows=list(csv.DictReader((output/'segments.csv').open()))
    inputs=sequence_data(rows,output)
    groups=np.array([r['group'] for r in rows]);all_indices=np.arange(len(rows))
    fit,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=20261017).split(all_indices,groups=groups))
    tr_rel,dev_rel=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=20261018).split(fit,groups=groups[fit]))
    train,dev=fit[tr_rel],fit[dev_rel]
    assert not (set(groups[train])&set(groups[dev]) or set(groups[train])&set(groups[test]) or set(groups[dev])&set(groups[test]))
    split={name:{'ids':[rows[i]['id'] for i in ix],'groups':sorted(set(groups[ix]))}
           for name,ix in [('train',train),('dev',dev),('test',test)]}
    (output/'neural_splits.json').write_text(json.dumps(split,indent=2))
    action=np.array([pilot.VOCAB.index(r['verb']) if r['verb'] in pilot.VOCAB else -1 for r in rows])
    error=np.array([int(r['mistake']) for r in rows])
    def weights(labels,classes):
        counts=np.bincount(labels[train][labels[train]>=0],minlength=classes)
        return torch.tensor(counts.sum()/np.maximum(counts*classes,1),dtype=torch.float32,device=device)
    aw,ew=weights(action,8),weights(error,2)
    pca=PCA(n_components=64,svd_solver='randomized',random_state=0)
    with threadpool_limits(limits=4):
        pca.fit(inputs['visual'][train].reshape(-1,512))
        visual=pca.transform(inputs['visual'].reshape(-1,512)).reshape(len(rows),4,64)
    y_action=torch.tensor(action,dtype=torch.long,device=device);y_error=torch.tensor(error,dtype=torch.long,device=device)
    metrics=[];predictions=[];histories={};parameter_count=None
    linear_metrics=[];linear_predictions=[]
    for mode in MODES:
        gaze=np.zeros((len(rows),4,26),np.float32)
        if mode=='static_gaze':gaze[:,:,:8]=inputs['static'][:,None,:]
        elif mode in ['temporal_gaze','gaze_only']:gaze=inputs['temporal']
        elif mode=='shuffled_gaze':gaze=inputs['shuffled']
        vis=np.zeros_like(visual) if mode=='gaze_only' else visual
        x=np.concatenate([vis,inputs['common'],gaze],axis=-1)
        mean=x[train].mean((0,1));std=np.maximum(x[train].std((0,1)),1e-4)
        x=torch.tensor((x-mean)/std,dtype=torch.float32)
        flat=x.numpy().reshape(len(rows),-1)
        with threadpool_limits(limits=4):
            amask=action[train]>=0;emask=error[train]>=0
            action_linear=LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=0)
            error_linear=LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=0)
            action_linear.fit(flat[train][amask],action[train][amask])
            error_linear.fit(flat[train][emask],error[train][emask])
            lap=action_linear.predict_proba(flat[test]);lep=error_linear.predict_proba(flat[test])[:,1]
        assert list(action_linear.classes_)==list(range(8))
        amask=action[test]>=0;emask=error[test]>=0
        linear_metrics.append({'mode':mode,'family':'sequence_linear',
            'verb_macro_f1':float(f1_score(action[test][amask],lap[amask].argmax(1),labels=list(range(8)),average='macro',zero_division=0)),
            'mistake_ap':float(average_precision_score(error[test][emask],lep[emask]))})
        for j,i in enumerate(test):
            linear_predictions.append({'id':rows[i]['id'],'group':rows[i]['group'],'mode':mode,'seed':0,
                'verb_label':int(action[i]),'error_label':int(error[i]),'error_probability':float(lep[j]),
                **{f'action_{k}':float(lap[j,k]) for k in range(8)}})
        pilot.write_csv(output/'sequence_linear_metrics.csv',linear_metrics)
        pilot.write_csv(output/'sequence_linear_predictions.csv',linear_predictions)
        print('MATCHED LINEAR',linear_metrics[-1],flush=True)
        for seed in args.seeds:
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=JointEncoder(x.shape[-1]).to(device)
            count=sum(p.numel() for p in model.parameters())
            parameter_count=count if parameter_count is None else parameter_count
            assert count==parameter_count
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
            best=None;best_loss=float('inf');stale=0;history=[]
            for epoch in range(args.epochs):
                model.train();order=rng.permutation(train);train_loss=0
                for offset in range(0,len(order),128):
                    ix=order[offset:offset+128];optimizer.zero_grad()
                    loss=loss_function(model(x[ix].to(device)),y_action[ix],y_error[ix],aw,ew)
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                    train_loss+=float(loss.detach())*len(ix)
                model.eval();dev_loss=0
                with torch.inference_mode():
                    for offset in range(0,len(dev),256):
                        ix=dev[offset:offset+256]
                        value=loss_function(model(x[ix].to(device)),y_action[ix],y_error[ix],aw,ew)
                        dev_loss+=float(value)*len(ix)
                dev_loss/=len(dev);history.append({'epoch':epoch+1,'train_loss':train_loss/len(train),'dev_loss':dev_loss})
                if dev_loss<best_loss-1e-6:
                    best_loss=dev_loss;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
                else:stale+=1
                if stale>=5:break
            model.load_state_dict(best)
            ap,ep=evaluate(model,x,test,device)
            amask=action[test]>=0;emask=error[test]>=0
            record={'mode':mode,'family':'gru','seed':seed,'epochs':len(history),'parameters':count,
                    'verb_macro_f1':float(f1_score(action[test][amask],ap[amask].argmax(1),labels=list(range(8)),average='macro',zero_division=0)),
                    'mistake_ap':float(average_precision_score(error[test][emask],ep[emask])),
                    'test_actions':int(amask.sum()),'test_correctness':int(emask.sum()),'test_mistakes':int(error[test][emask].sum())}
            metrics.append(record);histories[f'{mode}_{seed}']=history;print('NEURAL RESULT',record,flush=True)
            for j,i in enumerate(test):
                predictions.append({'id':rows[i]['id'],'group':rows[i]['group'],'mode':mode,'seed':seed,
                                    'verb_label':int(action[i]),'error_label':int(error[i]),'error_probability':float(ep[j]),
                                    **{f'action_{k}':float(ap[j,k]) for k in range(8)}})
            torch.save({k:v.cpu() for k,v in model.state_dict().items()},output/f'neural_{mode}_{seed}.pt')
            pilot.write_csv(output/'neural_metrics.csv',metrics);pilot.write_csv(output/'neural_predictions.csv',predictions)
    (output/'neural_training.json').write_text(json.dumps(histories,indent=2))
    (output/'neural_run.json').write_text(json.dumps({'synthetic':False,'device':device,'metrics':metrics,'parameters':parameter_count,
        'elapsed_seconds':time.time()-started,'protocol_sha256':hashlib.sha256((ROOT/'FULL_STUDY_PROTOCOL.md').read_bytes()).hexdigest()},indent=2))
    print('NEURAL COMPLETE',flush=True)


if __name__=='__main__':main()
