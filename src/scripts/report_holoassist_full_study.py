"""Report fresh-release linear and learned temporal pilots without overstating claims."""
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score,f1_score

from run_holoassist_pilot import write_csv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/holoassist_full_study'


def metric(y,a,e,task):
    if task=='verb':
        valid=y[:,0]>=0
        return float(f1_score(y[valid,0],a[valid].argmax(1),labels=list(range(8)),average='macro',zero_division=0))
    valid=y[:,1]>=0
    return float(average_precision_score(y[valid,1],e[valid]))


def main():
    classical=json.loads((OUT/'run.json').read_text());neural=json.loads((OUT/'neural_run.json').read_text())
    expected=hashlib.sha256((ROOT/'FULL_STUDY_PROTOCOL.md').read_bytes()).hexdigest()
    if classical['protocol_sha256']!=expected or neural['protocol_sha256']!=expected:
        raise ValueError('Protocol changed after fitting')
    rows=list(csv.DictReader((OUT/'segments.csv').open()))
    audit=list(csv.DictReader((OUT/'alignment_audit.csv').open()))
    selection=json.loads((OUT/'selection_plan.json').read_text())
    splits=json.loads((OUT/'neural_splits.json').read_text())
    modes=['visual_control','static_gaze','temporal_gaze','shuffled_gaze','gaze_only']
    raw=list(csv.DictReader((OUT/'neural_predictions.csv').open()))
    linear=list(csv.DictReader((OUT/'sequence_linear_predictions.csv').open()))
    ids=sorted({r['id'] for r in raw});lookup={r['id']:r for r in raw}
    labels=np.array([[int(lookup[i]['verb_label']),int(lookup[i]['error_label'])] for i in ids])
    groups=np.array([lookup[i]['group'] for i in ids]);positions={key:i for i,key in enumerate(ids)}
    predictions={};summary=[]
    for family,data in [('gru',raw),('sequence_linear',linear)]:
        for mode in modes:
            selected=[r for r in data if r['mode']==mode]
            a=np.zeros((len(ids),8));e=np.zeros(len(ids));counts=np.zeros(len(ids))
            for row in selected:
                i=positions[row['id']];a[i]+=[float(row[f'action_{k}']) for k in range(8)]
                e[i]+=float(row['error_probability']);counts[i]+=1
            expected_count=len({r['seed'] for r in selected})
            if not np.all(counts==expected_count) or not expected_count:
                raise ValueError('Incomplete prediction grid')
            a/=counts[:,None];e/=counts;predictions[family,mode]=(a,e)
            summary.append({'family':family,'mode':mode,'verb_macro_f1':metric(labels,a,e,'verb'),
                            'mistake_ap':metric(labels,a,e,'mistake'),'seeds_averaged':expected_count})
    by_id={r['id']:r for r in rows}
    data_root=Path.home()/'Datasets/HoloAssist'
    annotations={r['video_name']:r for r in json.loads((data_root/'data-annotation-trainval-v1_1.json').read_text())}
    exposure=[]
    for event_id in ids:
        row=by_id[event_id];start,end=float(row['start']),float(row['end']);cutoff=start+.85*(end-start)
        correctives=[e for e in annotations[row['recording']]['events'] if e['label']=='Conversation'
            and 'correct' in str(e.get('attributes',{}).get('Conversation Purpose','')).lower()
            and 'wrong' in str(e.get('attributes',{}).get('Conversation Purpose','')).lower()
            and e['start']<=cutoff and e['end']>=start]
        exposure.append({'id':event_id,'group':row['group'],'corrective_speech_observed':bool(correctives),
                         'error_label':int(row['mistake'])})
    write_csv(OUT/'assistance_exposure.csv',exposure)
    no_correction=np.array([not r['corrective_speech_observed'] for r in exposure])
    subgroup=[]
    for name,mask in [('all',np.ones(len(ids),bool)),('no_annotated_corrective_speech',no_correction)]:
        eligible=mask&(labels[:,1]>=0)
        for mode in modes:
            a,e=predictions['gru',mode]
            subgroup.append({'stratum':name,'mode':mode,'segments':int(eligible.sum()),
                'mistakes':int((labels[eligible,1]==1).sum()),
                'prevalence':float((labels[eligible,1]==1).mean()),
                'average_precision':float(average_precision_score(labels[eligible,1],e[eligible]))
                    if (labels[eligible,1]==1).any() else None})
    write_csv(OUT/'assistance_subgroup_metrics.csv',subgroup)
    rng=np.random.default_rng(42);names=np.unique(groups);draws=[]
    for _ in range(1000):
        ix=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(names,len(names),replace=True)])
        if (labels[ix,1]==1).sum():draws.append(ix)
    bootstrap={}
    for key,(a,e) in predictions.items():
        for task in ['verb','mistake']:
            bootstrap[key+(task,)]=np.array([metric(labels[ix],a[ix],e[ix],task) for ix in draws])
    comparisons=[]
    for task in ['verb','mistake']:
        for baseline in ['visual_control','static_gaze','shuffled_gaze']:
            first=('gru','temporal_gaze');second=('gru',baseline)
            delta=bootstrap[first+(task,)]-bootstrap[second+(task,)]
            point=metric(labels,*predictions[first],task)-metric(labels,*predictions[second],task)
            lo,hi=np.quantile(delta,[.025,.975])
            comparisons.append({'task':task,'contrast':'GRU temporal minus '+baseline,'difference':point,'low':float(lo),'high':float(hi)})
        for mode in modes:
            first=('gru',mode);second=('sequence_linear',mode)
            delta=bootstrap[first+(task,)]-bootstrap[second+(task,)]
            point=metric(labels,*predictions[first],task)-metric(labels,*predictions[second],task)
            lo,hi=np.quantile(delta,[.025,.975])
            comparisons.append({'task':task,'contrast':'GRU minus matched linear: '+mode,'difference':point,'low':float(lo),'high':float(hi)})
    write_csv(OUT/'neural_summary.csv',summary);write_csv(OUT/'neural_contrasts.csv',comparisons)
    def classic(task,mode,budget=4):
        return next(r['score'] for r in classical['metrics'] if r['task']==task and r['mode']==mode and r['budget_frames']==budget)
    def contrast(task,name):
        return next(r for r in classical['contrasts'] if r['task']==task and r['contrast']==name)
    def fmt(c):
        return f"{100*c['difference']:+.2f} pp [{100*c['low']:+.2f}, {100*c['high']:+.2f}]"
    h3=contrast('mistake','H3 temporal minus oracle_control');order=contrast('mistake','H3 temporal minus oracle_shuffled')
    h1=contrast('verb','H1 interaction')
    support='positive fresh-recording association' if h3['low']>0 else 'inconclusive fresh-recording association' if h3['high']>=0 else 'negative fresh-recording contrast'
    parts=[]
    for task in ['verb','mistake']:
        for mode in modes:
            key=('gru',mode);a,e=predictions[key];parts.append({'task':task,'mode':mode,'score':metric(labels,a,e,task),
                'low':float(np.quantile(bootstrap[key+(task,)],.025)),
                'high':float(np.quantile(bootstrap[key+(task,)],.975))})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors=['#4c78a8','#59a14f','#f28e2b','#af7aa1','#929292']
    for col,task in enumerate(['verb','mistake']):
        ax=axes[0,col];cm=['control','static','temporal','shuffled']
        ax.bar(np.arange(4),[100*classic(task,m) for m in cm],color=colors[:4])
        ax.set_xticks(np.arange(4),cm);ax.set_title('Fresh-recording 4-fold linear probes')
        ax.set_ylabel('Verb macro-F1 (%)' if task=='verb' else 'Mistake AP (%)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax=axes[1,col];values=[r for r in parts if r['task']==task]
        ax.bar(np.arange(5),[100*r['score'] for r in values],color=colors)
        for i,r in enumerate(values):ax.vlines(i,100*r['low'],100*r['high'],color='black')
        ax.set_xticks(np.arange(5),[m.replace('_','\n') for m in modes],fontsize=8)
        ax.set_title('Held-out GRU, three-seed mean probabilities')
        ax.set_ylabel('Verb macro-F1 (%)' if task=='verb' else 'Mistake AP (%)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Fresh HoloAssist study: 240 recordings, 20 task categories\nTop and bottom panels use different evaluation splits; compare conditions within each panel')
    fig.savefig(OUT/'full_study_results.png',dpi=200);fig.savefig(OUT/'full_study_results.pdf')
    def neur(task,name):
        return next(c for c in comparisons if c['task']==task and c['contrast']==name)
    group_support={name:{'groups':len(split['groups']),'segments':len(split['ids'])} for name,split in splits.items()}
    lines=['# Fresh-recording HoloAssist pilot from the full release','',
        'Executed 2026-10-17. This report uses newly sampled recording groups, not a reanalysis of the earlier 34 recordings.',
        '', '## Findings for the hypothesis','',
        f"1. **Execution quality after oracle context controls:** {fmt(h3)} — {support}. "
        f"Mistake AP: {100*classic('mistake','oracle_control'):.2f}% control, {100*classic('mistake','oracle_temporal'):.2f}% temporal gaze, "
        f"{100*classic('mistake','oracle_shuffled'):.2f}% shuffled gaze. Temporal-minus-shuffled: {fmt(order)}.",
        f"2. **Sparse visual budget:** action interaction {fmt(h1)}. One-frame temporal fusion minus four-frame control: "
        f"{fmt(contrast('verb','H1 sparse fusion minus dense control'))}. This is not an energy measurement.",
        f"3. **Learned temporal encoder, held-out correctness:** {fmt(neur('mistake','GRU temporal minus visual_control'))} "
        f"versus RGB/quality/motion control; {fmt(neur('mistake','GRU temporal minus shuffled_gaze'))} versus shuffled gaze.",
        '', 'Positive contrasts with intervals crossing zero are inconclusive. Bootstrap intervals are conditional on fixed predictions, '
        'not corrected for multiple comparisons, and do not include training-data resampling uncertainty. '
        'Scene-grounded symbolic concepts, structured teacher transfer, wearer personalization, and device efficiency are not validated by these baselines.',
        '', '## Data and independence','',
        f"Selected {len(selection['recordings'])} recordings: 12 per task across 20 categories, {selection['groups']} recording-prefix groups. "
        f"All selected records passed schema/timing audits. Usable fine-action segments: {len(rows)}; explicit mistakes: "
        f"{sum(int(r['mistake'])==1 for r in rows)}. Actual video duration: {sum(float(r['duration_s']) for r in audit)/3600:.2f} hours.",
        '', f"Extracted {selection['bytes']/1e9:.2f} GB of RGB, gaze, video poses/timing and calibration from the complete 168.85 GB release. "
        'No prior 34-recording prefix group is reused. Prefix IDs are not verified wearer IDs. '
        'Sampling is task-balanced, not representative of the release\'s natural task mixture; absolute AP must be interpreted with its prevalence.',
        '', '## Fresh-data linear probes','', '| Frames | Condition | Verb macro-F1 (%) | Mistake AP (%) |', '|---:|---|---:|---:|']
    for budget in [1,4]:
        for mode in ['control','static','temporal','shuffled']:
            lines.append(f"| {budget} | {mode} | {100*classic('verb',mode,budget):.2f} | {100*classic('mistake',mode,budget):.2f} |")
    lines+=['', 'Control includes visual, duration, gaze-validity/sample-coverage and camera-motion features. '
        'The oracle diagnostic adds ground-truth verb, noun, task, and joint verb-noun categorical encodings. '
        'These labels are diagnostic inputs only. PCA/scalers/category encodings fit on training folds only; C=1 and fixed vocabulary.',
        '', '## Learned temporal encoding','',
        f"Unidirectional 64-hidden-unit GRU with frozen ResNet18 scene embeddings, joint action/correctness heads, "
        f"{neural['parameters']:,} trainable parameters per condition. Three seeds, development-only epoch selection, and group-disjoint splits: {group_support}.",
        '', 'The trainable parameter count excludes the frozen ResNet18 vision frontend and is not the total deployed model size or its processing cost.',
        '', '| Condition | GRU verb macro-F1 (%) | GRU mistake AP (%) | Matched-input linear verb macro-F1 (%) | Matched-input linear mistake AP (%) |',
        '|---|---:|---:|---:|---:|']
    for mode in modes:
        g=next(r for r in summary if r['family']=='gru' and r['mode']==mode)
        l=next(r for r in summary if r['family']=='sequence_linear' and r['mode']==mode)
        lines.append(f"| {mode} | {100*g['verb_macro_f1']:.2f} | {100*g['mistake_ap']:.2f} | {100*l['verb_macro_f1']:.2f} | {100*l['mistake_ap']:.2f} |")
    lines+=['', 'GRU values above average probabilities from three separately trained seeds; they are an ensemble diagnostic, '
        'not a single deployed model or its compute cost. Individual-seed metrics are in neural_metrics.csv. '
        'Matched-input linear heads use identical sequence inputs and train/test split; unlike the GRU they train heads separately. '
        'All GRU input-ablation conditions have exactly equal architecture and parameter count.',
        '', '### Corrective-speech exposure diagnostic','',
        '| Held-out stratum | Eligible segments | Mistakes | Prevalence (%) | Control AP (%) | Temporal AP (%) | Shuffled AP (%) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for stratum in ['all','no_annotated_corrective_speech']:
        ss={r['mode']:r for r in subgroup if r['stratum']==stratum};reference=ss['visual_control']
        values=[100*ss[m]['average_precision'] if ss[m]['average_precision'] is not None else float('nan')
                for m in ['visual_control','temporal_gaze','shuffled_gaze']]
        lines.append(f"| {stratum} | {reference['segments']} | {reference['mistakes']} | {100*reference['prevalence']:.2f} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} |")
    lines+=[
        '', '![Results](full_study_results.png)', '', '## Held-out neural contrasts','',
        '| Metric | Contrast | Difference / conditional 95% group interval |','|---|---|---:|']
    lines += [f"| {c['task']} | {c['contrast']} | {fmt(c)} |" for c in comparisons]
    lines += ['', '## What this does and does not test','',
        'The models observe completed annotated action segments with oracle boundaries/duration. '
        'Linear probes use gaze through the segment end; learned sequences use gaze only through the final observed frame at 85%. '
        'The linear-CV and neural-held-out panels differ in training split and observation window; their absolute scores are not directly comparable. '
        'The shared-input held-out linear comparison is provided for architecture diagnostics.',
        '', 'A corrective-speech timing diagnostic is saved in assistance_exposure.csv and assistance_subgroup_metrics.csv. '
        'The no-corrective-speech stratum excludes examples overlapping conversation purpose labels containing both "correct" and "wrong" '
        'before the final 85%-of-segment observation. This is not proof that no assistance occurred, and subgroup AP has its own prevalence.',
        '', 'Corrective speech may already have influenced the wearer during a completed wrong-action segment. '
        'No model receives speech, narration, error explanations or correctness labels as input, but this still does not establish '
        'early error detection, error onset, latent intent, or intervention necessity. Gaze is camera-relative angular measurement; '
        'statistics are not physiological fixation annotations or tracked-object revisits. Calibration extraction is not itself scene grounding.',
        '', 'No test-set tuning was performed. Multitask training is not held-out-task representation transfer. '
        'The study tests incremental information and a lightweight learned baseline, not the proposed mobile VLM, symbolic state tracker, '
        'knowledge-transfer method, or real-device controller.',
        '', '## Reproduce','', 'Use the dedicated runtime outside OneDrive:', '```powershell',
        '& "$HOME/Datasets/HoloAssist/.venv/Scripts/python.exe" scripts/prepare_holoassist_full_study.py',
        '& "$HOME/Datasets/HoloAssist/.venv/Scripts/python.exe" scripts/audit_holoassist_full_study.py',
        '& "$HOME/Datasets/HoloAssist/.venv/Scripts/python.exe" scripts/run_holoassist_pilot.py --data-root "$HOME/Datasets/HoloAssist/study_data" --plan "$HOME/Datasets/HoloAssist/full_study_audited_plan.json" --output results/holoassist_full_study --frame-budget-features --extract-only --device cuda',
        '& "$HOME/Datasets/HoloAssist/.venv/Scripts/python.exe" scripts/run_holoassist_hypotheses.py --output results/holoassist_full_study --protocol FULL_STUDY_PROTOCOL.md',
        '& "$HOME/Datasets/HoloAssist/.venv/Scripts/python.exe" scripts/run_holoassist_temporal.py',
        '& "$HOME/Datasets/HoloAssist/.venv/Scripts/python.exe" scripts/report_holoassist_full_study.py', '```',
        '', 'The source protocol hash matches FULL_STUDY_PROTOCOL.md for both fitted model suites. '
        'selection_plan.json, input_audit.csv, alignment_audit.csv, segments.csv, predictions.csv, splits.json, '
        'neural_splits.json, neural_predictions.csv and training histories preserve the audit trail. '
        'Original release archives have size/ETag/SHA-256 records under the data root; extracted files have SHA-256 sidecars.',
        '', 'Dataset source: [official HoloAssist release](https://holoassist.github.io/).']
    lines += ['', '## Exploratory context diagnostic', '',
        'A post-result diagnostic in within_context_metrics.csv finds context-only mistake AP of 18.26%, '
        'versus 18.37% for the visual/context control. In 501 fold-specific task/verb/noun strata '
        'containing both outcomes (5,907 segments, 783 mistakes), pair-weighted AUROC is 0.600 for '
        'the control and 0.577 for temporal gaze. This descriptive analysis was added after primary '
        'results; it is not a prespecified confirmatory test. Run scripts/diagnose_holoassist_context.py '
        'to regenerate it. See ../../FULL_STUDY_FINDINGS.md for the revised hypothesis.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT COMPLETE',OUT/'REPORT.md')


if __name__=='__main__':main()
