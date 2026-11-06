"""Fixed exploratory tests specified in HYPOTHESIS_PROTOCOL.md."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from run_holoassist_pilot import VOCAB, write_csv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/holoassist_hypothesis'


def confusion_f1(y,pred):
    matrix=np.bincount(y*8+pred,minlength=64).reshape(8,8)
    return float(np.mean(2*np.diag(matrix)/np.maximum(matrix.sum(0)+matrix.sum(1),1)))


def feature_parts(mode):
    parts=['duration','quality','motion']
    if mode.endswith(('static','temporal','shuffled')):
        parts+=['static']
    if mode.endswith('temporal'):
        parts+=['dynamic']
    if mode.endswith('shuffled'):
        parts+=['shuffled']
    return parts


def fit_probes(rows,arrays):
    metrics,predictions,splits=[],[],[]
    for task in ['verb','mistake']:
        ix=np.array([i for i,r in enumerate(rows) if r['verb'] in VOCAB] if task=='verb' else
                    [i for i,r in enumerate(rows) if int(r['mistake'])>=0])
        y=np.array([VOCAB.index(rows[i]['verb']) if task=='verb' else int(rows[i]['mistake']) for i in ix])
        groups=np.array([rows[i]['group'] for i in ix])
        folds=list(GroupKFold(n_splits=4).split(ix,y,groups))
        categorical=np.array([[rows[i]['verb'],rows[i]['noun'],rows[i]['task'],
                               rows[i]['verb']+'::'+rows[i]['noun']] for i in ix],dtype=object)
        for budget in [1,4]:
            modes=['control','static','temporal','shuffled']
            if task=='mistake' and budget==4:
                modes+=['oracle_control','oracle_static','oracle_temporal','oracle_shuffled']
            # Fit PCA once per budget/fold; every gaze condition shares it.
            visual=arrays['visual_one' if budget==1 else 'visual']
            fold_features=[]
            for fold,(tr,te) in enumerate(folds):
                train,test=ix[tr],ix[te]
                pca=PCA(n_components=32,svd_solver='randomized',random_state=0)
                vtr=pca.fit_transform(visual[train]);vte=pca.transform(visual[test])
                enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
                ctr=enc.fit_transform(categorical[tr]);cte=enc.transform(categorical[te])
                fold_features.append((vtr,vte,ctr,cte))
                if budget==1:
                    splits.append({'task':task,'fold':fold,'train_groups':sorted(set(groups[tr])),
                                   'test_groups':sorted(set(groups[te])),'train_segments':len(tr),'test_segments':len(te)})
            for mode in modes:
                pred=np.full(len(ix),-1);prob=np.zeros(len(ix));dim=None
                for fold,(tr,te) in enumerate(folds):
                    train,test=ix[tr],ix[te]
                    vtr,vte,ctr,cte=fold_features[fold]
                    parts=feature_parts(mode)
                    xtr=np.concatenate([vtr]+[arrays[k][train] for k in parts]+([ctr] if mode.startswith('oracle') else []),axis=1)
                    xte=np.concatenate([vte]+[arrays[k][test] for k in parts]+([cte] if mode.startswith('oracle') else []),axis=1)
                    dim=xtr.shape[1]
                    model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=0))
                    model.fit(xtr,y[tr]);pred[te]=model.predict(xte)
                    if task=='mistake':
                        prob[te]=model.predict_proba(xte)[:,list(model[-1].classes_).index(1)]
                metric=confusion_f1(y,pred) if task=='verb' else float(average_precision_score(y,prob))
                record={'task':task,'budget_frames':budget,'mode':mode,'segments':len(ix),'groups':len(set(groups)),
                        'metric':'macro_f1' if task=='verb' else 'average_precision','score':metric,
                        'accuracy':float(np.mean(y==pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
                        'positive_prevalence':float(y.mean()) if task=='mistake' else None,
                        'last_fold_features':dim}
                metrics.append(record);print(record,flush=True)
                for j,index in enumerate(ix):
                    predictions.append({'task':task,'budget_frames':budget,'mode':mode,'id':rows[index]['id'],
                                        'group':rows[index]['group'],'label':int(y[j]),'prediction':int(pred[j]),
                                        'probability':float(prob[j])})
    write_csv(OUT/'metrics.csv',metrics);write_csv(OUT/'predictions.csv',predictions)
    (OUT/'splits.json').write_text(json.dumps(splits,indent=2))
    return metrics,predictions


def bootstrap_contrasts(metrics,predictions):
    contrasts=[];intervals=[]
    rng=np.random.default_rng(42)
    for task in ['verb','mistake']:
        selected=[r for r in metrics if r['task']==task]
        mapped={(r['budget_frames'],r['mode']):{} for r in selected}
        for r in predictions:
            if r['task']==task:
                mapped[r['budget_frames'],r['mode']][r['id']]=r
        ids=sorted(next(iter(mapped.values())))
        first=mapped[1,'control']
        y=np.array([first[i]['label'] for i in ids]);groups=np.array([first[i]['group'] for i in ids])
        names=np.unique(groups);draws=[]
        for _ in range(1000):
            idx=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(names,len(names),replace=True)])
            if task=='mistake' and y[idx].sum()==0:
                continue
            draws.append(idx)
        boots={};points={}
        for r in selected:
            key=r['budget_frames'],r['mode']
            values=np.array([mapped[key][i]['prediction' if task=='verb' else 'probability'] for i in ids])
            boots[key]=np.array([confusion_f1(y[idx],values[idx]) if task=='verb' else
                                 average_precision_score(y[idx],values[idx]) for idx in draws])
            points[key]=r['score'];lo,hi=np.quantile(boots[key],[.025,.975])
            intervals.append({'task':task,'budget_frames':key[0],'mode':key[1],'low':float(lo),'high':float(hi)})
        comparisons=[('H1 interaction',[(1,'temporal',1),(1,'control',-1),(4,'temporal',-1),(4,'control',1)]),
                     ('H1 sparse fusion minus dense control',[(1,'temporal',1),(4,'control',-1)])]
        for budget in [1,4]:
            for baseline in ['control','static','shuffled']:
                comparisons.append((f'H2 {budget}-frame temporal minus {baseline}',[(budget,'temporal',1),(budget,baseline,-1)]))
        if task=='mistake':
            for baseline in ['oracle_control','oracle_static','oracle_shuffled']:
                comparisons.append((f'H3 temporal minus {baseline}',[(4,'oracle_temporal',1),(4,baseline,-1)]))
        for name,terms in comparisons:
            delta=sum(sign*boots[budget,mode] for budget,mode,sign in terms)
            point=sum(sign*points[budget,mode] for budget,mode,sign in terms)
            lo,hi=np.quantile(delta,[.025,.975])
            contrasts.append({'task':task,'contrast':name,'difference':float(point),'low':float(lo),'high':float(hi)})
    write_csv(OUT/'contrasts.csv',contrasts);write_csv(OUT/'intervals.csv',intervals)
    return contrasts


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--protocol',type=Path,default=ROOT/'HYPOTHESIS_PROTOCOL.md')
    args=parser.parse_args()
    OUT=args.output.resolve()
    started=time.time()
    rows=list(csv.DictReader((OUT/'segments.csv').open()))
    with np.load(OUT/'features.npz') as data:
        arrays={k:data[k] for k in data.files}
    if len(rows)!=len(arrays['visual']) or any(not np.isfinite(x).all() for x in arrays.values()):
        raise ValueError('Invalid feature data')
    with threadpool_limits(limits=4):
        metrics,predictions=fit_probes(rows,arrays)
        contrasts=bootstrap_contrasts(metrics,predictions)
    (OUT/'run.json').write_text(json.dumps({'synthetic':False,'elapsed_seconds':time.time()-started,
        'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        'protocol':'Fixed exploratory tests, four-fold recording-prefix-group CV, C=1, no tuning',
        'metrics':metrics,'contrasts':contrasts},indent=2))


if __name__=='__main__':
    main()
