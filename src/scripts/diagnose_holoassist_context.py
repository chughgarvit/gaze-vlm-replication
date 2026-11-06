"""Post-result diagnostic: context priors versus within-action correctness ranking.

This is explicitly exploratory after inspecting the fresh linear results.
It does not change primary models or their fixed protocol.
"""
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from threadpoolctl import threadpool_limits
from run_holoassist_pilot import write_csv

OUT=Path(__file__).resolve().parents[1]/'results/holoassist_full_study'


def main():
    rows=[r for r in csv.DictReader((OUT/'segments.csv').open()) if int(r['mistake'])>=0]
    y=np.array([int(r['mistake']) for r in rows]);groups=np.array([r['group'] for r in rows])
    categories=np.array([[r['verb'],r['noun'],r['task'],r['verb']+'::'+r['noun']] for r in rows],dtype=object)
    folds=np.full(len(rows),-1);scores=np.zeros(len(rows))
    with threadpool_limits(limits=4):
        for fold,(tr,te) in enumerate(GroupKFold(n_splits=4).split(y,groups=groups)):
            model=make_pipeline(OneHotEncoder(handle_unknown='ignore',sparse_output=False),StandardScaler(),
                                LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=0))
            model.fit(categories[tr],y[tr]);scores[te]=model.predict_proba(categories[te])[:,1];folds[te]=fold
    write_csv(OUT/'context_only_predictions.csv',[{'id':r['id'],'group':r['group'],'fold':int(folds[i]),
        'label':int(y[i]),'probability':float(scores[i])} for i,r in enumerate(rows)])
    models={'context_only':scores};lookup={r['id']:i for i,r in enumerate(rows)}
    for row in csv.DictReader((OUT/'predictions.csv').open()):
        if row['task']=='mistake' and int(row['budget_frames'])==4 and row['mode'].startswith('oracle_'):
            models.setdefault(row['mode'],np.zeros(len(rows)))[lookup[row['id']]]=float(row['probability'])
    strata=defaultdict(list)
    for i,r in enumerate(rows):strata[(int(folds[i]),r['task'],r['verb'],r['noun'])].append(i)
    eligible=[(key,np.array(ix)) for key,ix in strata.items() if y[ix].min()!=y[ix].max()]
    covered=np.unique(np.concatenate([ix for _,ix in eligible]))
    support=[{'fold':key[0],'task':key[1],'verb':key[2],'noun':key[3],'segments':len(ix),
              'mistakes':int(y[ix].sum()),'correct':int((y[ix]==0).sum())} for key,ix in eligible]
    write_csv(OUT/'within_context_support.csv',support)
    results=[]
    for mode,p in models.items():
        aucs=[];weights=[]
        for _,ix in eligible:
            aucs.append(roc_auc_score(y[ix],p[ix]));weights.append(int(y[ix].sum())*int((y[ix]==0).sum()))
        results.append({'mode':mode,'global_average_precision':float(average_precision_score(y,p)),
            'within_context_pair_weighted_auc':float(np.average(aucs,weights=weights)),
            'within_context_macro_auc':float(np.mean(aucs)),'eligible_fold_contexts':len(eligible),
            'covered_segments':len(covered),'covered_mistakes':int(y[covered].sum())})
    write_csv(OUT/'within_context_metrics.csv',results)
    (OUT/'context_diagnostic_note.json').write_text(json.dumps({'analysis':'Exploratory diagnostic added after fresh primary linear scores',
        'conditioning':'Within fold and exact task/verb/noun label stratum with both classes',
        'weighting':'Positive-negative pair count per stratum','uncertainty':'Point estimates only; no confirmatory inference',
        'results':results},indent=2))
    for row in results:print(row,flush=True)


if __name__=='__main__':main()
