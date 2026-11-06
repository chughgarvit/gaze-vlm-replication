"""Summarize measured pilot outputs with recording-group bootstrap uncertainty."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, f1_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/holoassist_pilot'
DATA=ROOT/'data/external/holoassist'


def main():
    metrics=list(csv.DictReader((OUT/'metrics.csv').open()))
    rows=list(csv.DictReader((OUT/'predictions.csv').open()))
    intervals=[]
    contrasts=[]
    rng=np.random.default_rng(42)
    figure,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,task,metric in zip(axes,['verb','mistake'],['macro_f1','average_precision']):
        selected=[r for r in metrics if r['task']==task]
        modes=[r['mode'] for r in selected]
        by_mode={m:{r['id']:r for r in rows if r['task']==task and r['mode']==m} for m in modes}
        ids=sorted(by_mode[modes[0]])
        groups=np.array([by_mode[modes[0]][i]['group'] for i in ids])
        labels=np.array([int(by_mode[modes[0]][i]['label']) for i in ids])
        group_names=np.unique(groups)
        samples=[]
        for _ in range(1000):
            drawn=rng.choice(group_names,len(group_names),replace=True)
            idx=np.concatenate([np.flatnonzero(groups==g) for g in drawn])
            if task=='mistake' and labels[idx].sum()==0:
                continue
            samples.append(idx)
        boots={}
        for mode in modes:
            values=np.array([float(by_mode[mode][i]['mistake_probability']) if task=='mistake'
                             else int(by_mode[mode][i]['prediction']) for i in ids])
            func=(lambda ii:average_precision_score(labels[ii],values[ii])) if task=='mistake' else (
                lambda ii:f1_score(labels[ii],values[ii],labels=list(range(8)),average='macro',zero_division=0))
            boots[mode]=np.array([func(idx) for idx in samples])
            lo,hi=np.quantile(boots[mode],[.025,.975])
            intervals.append({'task':task,'mode':mode,'metric':metric,'low':float(lo),'high':float(hi),
                              'bootstrap_samples':len(samples)})
        for first,second in [('fusion_temporal','visual_quality_motion'),('fusion_temporal','fusion_static'),
                             ('fusion_temporal','fusion_shuffled'),('fusion_static','visual_quality_motion')]:
            delta=boots[first]-boots[second]
            point=float(next(r[metric] for r in selected if r['mode']==first))-float(next(r[metric] for r in selected if r['mode']==second))
            lo,hi=np.quantile(delta,[.025,.975])
            contrasts.append({'task':task,'contrast':first+' minus '+second,'metric':metric,
                              'difference':point,'low':float(lo),'high':float(hi)})
        y=np.array([float(r[metric])*100 for r in selected])
        lows=np.array([next(r['low'] for r in intervals if r['task']==task and r['mode']==m)*100 for m in modes])
        highs=np.array([next(r['high'] for r in intervals if r['task']==task and r['mode']==m)*100 for m in modes])
        x=np.arange(len(modes))
        ax.bar(x,y,color=['#8b939c' if m=='prior' else '#5185b8' if 'fusion' not in m else '#dc8c45' for m in modes])
        # Percentile intervals can exclude the point estimate for small/skewed samples.
        ax.vlines(x,lows,highs,color='black',linewidth=1)
        ax.scatter(x,lows,color='black',marker='_',s=30)
        ax.scatter(x,highs,color='black',marker='_',s=30)
        ax.set_xticks(x,[m.replace('_','\n') for m in modes],rotation=40,ha='right',fontsize=8)
        ax.set_ylabel(('Macro-F1' if task=='verb' else 'Average precision')+' (%)')
        ax.set_title('Eight-verb recognition' if task=='verb' else 'Completed-segment mistake classification')
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
        if task=='mistake':
            ax.axhline(labels.mean()*100,color='gray',linestyle='--',label='Mistake prevalence')
            ax.legend(fontsize=8)
    figure.suptitle('HoloAssist convenience pilot: 20 recordings, 15 recording-prefix groups\n95% group bootstrap intervals conditional on fitted out-of-fold predictions',fontsize=11)
    figure.savefig(OUT/'pilot_comparison.png',dpi=200)
    figure.savefig(OUT/'pilot_comparison.pdf')
    for name,records in [('intervals.csv',intervals),('contrasts.csv',contrasts)]:
        with (OUT/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    run=json.loads((OUT/'run.json').read_text())
    segments=list(csv.DictReader((OUT/'segments.csv').open()))
    audits=list(csv.DictReader((OUT/'alignment_audit.csv').open()))
    size=sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file())
    lines=['# HoloAssist real-data pilot','',
           'Executed on Windows, 2026-10-16. These are real downloaded sensor/video observations; no synthetic efficacy data.',
           '',f"Downloaded HoloAssist files occupy {size/1e9:.3f} GB (decimal), below the user\'s 400 GB ceiling. "
           'The selected payload plan is 2.128 GB; annotations, split files, and small remote index/probe transfers are additional.',
           '',f"## Data and protocol\n\n{len(audits)} recordings; {len(set(r['group'] for r in segments))} recording-prefix groups; "
           f"{len(segments)} fine-grained segments; {sum(int(r['mistake'])==1 for r in segments)} mistakes. "
           f"Total recorded duration: {sum(float(r['duration_s']) for r in audits)/60:.1f} minutes.",
           '', 'Subset: first 20 label-covered RAM/graphics-card recordings available from a bounded archive probe. '
           'This is a convenience sample of two task domains, not a representative HoloAssist sample.',
           '', 'Four-fold cross-validation keeps all videos sharing the R-number recording prefix in one fold. '
           'Participant identity has not been verified, so this is not a participant-disjoint or personalization result. '
           'This is a custom pilot, not the official benchmark protocol. No tuning on held-out folds was performed.',
           '', 'Completed ground-truth action segments provide oracle temporal boundaries. Eight-verb vocabulary: '+', '.join(run['vocab'])+'. '
           'Other verbs are excluded only from verb recognition; correctness classification uses all eligible segments. '
           'Mistake predictions use the entire annotated action, so they are retrospective classification, not online detection or anticipation.',
           '', 'Frozen torchvision ImageNet ResNet18 features use four RGB frames at 10/35/60/85% of each segment; '
           'mean, standard deviation, and last-minus-first descriptors are reduced to 32 PCA dimensions using training folds only. '
           'Compressed-video presentation timestamps are checked for constant frame rate before using frame indices; '
           'the video span must agree with the released UTC start/end span within 250 ms or two frames. '
           'This checks the publisher compressed timeline, not independent recovery of every original captured frame. '
           'Class-balanced logistic regression uses fixed C=1; scaling is fit on training folds. '
           'All learned modes receive log-duration. Prior predicts the training-fold majority class.',
           '', 'Gaze uses source UTC ticks relative to the video start; duplicate source samples are removed. '
           'Only the latest camera pose at or before a gaze sample is used, with a 100 ms age limit. '
           'World gaze directions are rotated into camera coordinates using the publisher axis transform. '
           'This uses angular ray direction, not projected pixel gaze: camera intrinsics and assumed fixation depth are unnecessary. '
           'The final released eye flag is CalibrationValid according to the PSI exporter source; this flag, finite direction, and pose coverage determine usable readings. '
           'This is a numerical alignment audit, not independent eye-tracker calibration validation.',
           '', 'Static gaze: angular mean/std/10th/90th percentiles. Temporal gaze: angular speed summaries, '
           'low-speed proportion, angular path length, three temporal-bin means, displacement, and mean angular changes. '
           'These are trajectory statistics, not validated physiological fixations or physical-object revisitations. '
           'Quality features include valid coverage/sample count; camera-motion controls summarize translation and rotation changes. '
           'Shuffling preserves within-segment angular positions and static summaries while changing temporal order; one fixed shuffle is used.',
           '', '## Measured results','', '| Method | Verb accuracy (%) | Verb macro-F1 (%) | Mistake AP (%) | Mistake balanced accuracy (%) |',
           '|---|---:|---:|---:|---:|']
    for mode in [r['mode'] for r in metrics if r['task']=='verb']:
        action=next(r for r in metrics if r['task']=='verb' and r['mode']==mode)
        mistake=next(r for r in metrics if r['task']=='mistake' and r['mode']==mode)
        lines.append(f"| {mode} | {100*float(action['accuracy']):.2f} | {100*float(action['macro_f1']):.2f} | {100*float(mistake['average_precision']):.2f} | {100*float(mistake['balanced_accuracy']):.2f} |")
    action_visual=next(r for r in metrics if r['task']=='verb' and r['mode']=='visual')
    action_fusion=next(r for r in metrics if r['task']=='verb' and r['mode']=='fusion_temporal')
    lines+=['',f"Temporal fusion scores {100*float(action_fusion['macro_f1']):.2f}% action macro-F1 versus "
            f"{100*float(action_visual['macro_f1']):.2f}% for visual-only. It does not improve action recognition in this pilot. "
            'The paired temporal-fusion contrasts against quality/motion controls, static fusion, and shuffled fusion '
            'all have intervals crossing zero for both tasks. Thus this pilot does not establish an incremental '
            'temporal-gaze benefit. Static gaze-only has the highest mistake AP, but with 26 errors this is an '
            'exploratory observation rather than evidence for the proposed fused representation.']
    lines+=['','![Measured comparison](pilot_comparison.png)','','## Paired contrasts','',
            '| Task | Contrast | Difference (percentage points) | Conditional 95% group-bootstrap interval |',
            '|---|---|---:|---:|']
    for c in contrasts:
        lines.append(f"| {c['task']} | {c['contrast']} | {100*c['difference']:+.2f} | [{100*c['low']:+.2f}, {100*c['high']:+.2f}] |")
    lines+=['','Intervals resample the 15 recording-prefix groups 1,000 times using fixed out-of-fold predictions. '
            'They do not include training/model-selection uncertainty, do not establish participant-level generalization, '
            'and are not corrected for multiple comparisons. With only 26 mistakes, error metrics are particularly unstable. '
            'The majority-correct classifier already has 97.25% accuracy, making raw mistake accuracy misleading.',
            '', '## Interpretation limits','',
            'Assess incremental gaze against visual_quality_motion, temporal information against fusion_static, '
            'and ordering against fusion_shuffled. Positive differences without supporting intervals are exploratory signals. '
            'A gain over the prior or gaze-only baseline does not establish incremental value beyond RGB. '
            'Different feature sets have different classifier parameter counts; this is not a capacity-matched learned-fusion evaluation.',
            '', 'This pilot does not evaluate teacher/student VLM transfer, symbolic constraints, frozen cross-task representations, '
            'unseen-user adaptation, gaze-driven cropping/token selection, or mobile energy/latency. CPU feature extraction '
            'runtime is not a mobile efficiency result. No inference about internal intentions is established.',
            '', '## Reproduction','', '```powershell',
            '.\\.tools\\uv\\uv.exe pip sync --python .venv/Scripts/python.exe requirements-holoassist-pilot.lock.txt',
            '.venv/Scripts/python.exe scripts/prepare_holoassist_subset.py',
            '.venv/Scripts/python.exe scripts/run_holoassist_pilot.py',
            '.venv/Scripts/python.exe scripts/report_holoassist_pilot.py',
            '.venv/Scripts/python.exe -m unittest discover -s tests -p test_holoassist_pilot.py -v',
            '```','', 'The subset downloader consumes the cached publisher archive indices and subset plan; '
            'initial discovery used download_holoassist.py and holoassist_video_probe.py. '
            'Do not infer full-archive availability from a partial index. Local payloads have SHA-256 sidecars; '
            'these are local integrity records, not publisher-authenticated hashes.',
            '', '## Sources','',
            '- [Official HoloAssist release and CDLA v2 license](https://holoassist.github.io/)',
            '- [Annotation and synchronization schema](https://holoassist.github.io/data_links/README.html)',
            '- [Publisher camera coordinate transform](https://github.com/taeinkwon/PyHoloAssist/blob/main/hand_eye_project.py)',
            '- [PSI source timestamp and pose formats](https://github.com/microsoft/psi/blob/master/Sources/MixedReality/HoloLensCapture/HoloLensCaptureExporter/Readme.md)',
            '- [PSI CalibrationValid export field](https://github.com/microsoft/psi/blob/master/Sources/MixedReality/HoloLensCapture/HoloLensCaptureInterop/Operators.cs)',
            '- [Torchvision ResNet18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)',
            '', f"Latest pipeline invocation wall time: {run['elapsed_seconds']:.1f} seconds, "
            'including cached feature reuse and any waiting for video downloads; see run.json. '
            'This is not a cold-start performance measurement. All 49 implementation tests passed.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Report written:',OUT/'REPORT.md')


if __name__=='__main__':
    main()
