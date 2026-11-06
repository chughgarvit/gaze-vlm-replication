"""Export measured hypothesis-development findings and scientific figures."""
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/holoassist_hypothesis'
DATA=ROOT/'data/external/holoassist'


def main():
    run=json.loads((OUT/'run.json').read_text())
    metrics=run['metrics'];contrasts=run['contrasts']
    intervals=list(csv.DictReader((OUT/'intervals.csv').open()))
    rows=list(csv.DictReader((OUT/'segments.csv').open()))
    audits=list(csv.DictReader((OUT/'alignment_audit.csv').open()))
    plan=json.loads((DATA/'hypothesis_subset_plan.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text())
    selected=set(selection['selected'])
    annotations={r['video_name']:r for r in json.loads((DATA/'data-annotation-trainval-v1_1.json').read_text())}
    domains=Counter(annotations[n]['taskType'] for n in selected)
    def result(task,budget,mode):
        return next(r for r in metrics if r['task']==task and r['budget_frames']==budget and r['mode']==mode)
    def contrast(task,name):
        return next(r for r in contrasts if r['task']==task and r['contrast']==name)
    def interval(task,budget,mode):
        r=next(r for r in intervals if r['task']==task and int(r['budget_frames'])==budget and r['mode']==mode)
        return float(r['low'])*100,float(r['high'])*100
    def fmt(c):
        return f"{100*c['difference']:+.2f} pp [95% interval {100*c['low']:+.2f}, {100*c['high']:+.2f}]"
    def verdict(c):
        if c['low']>0:
            return 'positive exploratory signal; requires independent validation'
        if c['high']<0:
            return 'negative contrast in this diagnostic'
        return 'inconclusive; interval crosses zero'
    h1=contrast('verb','H1 interaction')
    h3=contrast('mistake','H3 temporal minus oracle_control')
    fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
    colors={'control':'#4c78a8','static':'#59a14f','temporal':'#f28e2b','shuffled':'#af7aa1'}
    for ax,task in zip(axes[:2],['verb','mistake']):
        for mode,color in colors.items():
            yy=[100*result(task,b,mode)['score'] for b in [1,4]]
            ax.plot([1,4],yy,'o-',label=mode,color=color)
            for b,y in zip([1,4],yy):
                lo,hi=interval(task,b,mode)
                ax.vlines(b,lo,hi,color=color,alpha=.5)
        ax.set_xticks([1,4]);ax.set_xlabel('RGB frames per completed segment')
        ax.set_ylabel('Macro-F1 (%)' if task=='verb' else 'Average precision (%)')
        ax.set_title('Eight-verb recognition' if task=='verb' else 'Mistake classification')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    modes=['oracle_control','oracle_static','oracle_temporal','oracle_shuffled']
    axes[2].bar(np.arange(4),[100*result('mistake',4,m)['score'] for m in modes],color=list(colors.values()))
    for i,m in enumerate(modes):
        lo,hi=interval('mistake',4,m);axes[2].vlines(i,lo,hi,color='black')
    axes[2].set_xticks(np.arange(4),[m.replace('oracle_','') for m in modes],rotation=25)
    axes[2].set_title('Mistakes with oracle action/object/task controls')
    axes[2].set_ylabel('Average precision (%)');axes[2].grid(axis='y',alpha=.2);axes[2].set_axisbelow(True)
    fig.suptitle(f'HoloAssist exploratory hypothesis tests: {len(audits)}-recording convenience subset\nIntervals resample recording-prefix groups using fixed out-of-fold predictions',fontsize=11)
    fig.savefig(OUT/'hypothesis_tests.png',dpi=200);fig.savefig(OUT/'hypothesis_tests.pdf')
    files=[]
    for archive,members in plan['archives'].items():
        for m in members:
            name=next(part for part in Path(m['name']).parts if part.startswith(('R','z')))
            if name not in selected:
                continue
            p=DATA/'subset'/m['name']
            if not p.exists() or p.stat().st_size!=m['size']:
                raise ValueError('Incomplete expanded payload')
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            if digest!=p.with_name(p.name+'.sha256').read_text().strip():
                raise ValueError('Payload hash mismatch')
            files.append({'archive':archive,'member':m['name'],'bytes':m['size'],'offset':m['offset'],'sha256':digest})
    actual_bytes=sum(f['bytes'] for f in files)
    (OUT/'provenance.json').write_text(json.dumps({'selected_payload_bytes':actual_bytes,
        'source_base':'https://hl2data.z5.web.core.windows.net/holoassist-data-release/',
        'files':files,'protocol_sha256':run['protocol_sha256']},indent=2))
    stored=sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file())
    lines=['# Expanded HoloAssist hypothesis-development study','',
           'Executed 2026-10-16. This is an exploratory follow-up to the initial pilot, not an independent confirmatory replication.',
           '', '## Main findings','',
           f"- **H1: does gaze help more with sparse RGB?** {fmt(h1)}; {verdict(h1)}. "
           f"One-frame temporal fusion versus four-frame control: {fmt(contrast('verb','H1 sparse fusion minus dense control'))}.",
           f"- **H2: does order add information?** Four-frame action temporal-minus-static: {fmt(contrast('verb','H2 4-frame temporal minus static'))}; "
           f"temporal-minus-shuffled: {fmt(contrast('verb','H2 4-frame temporal minus shuffled'))}. "
           'Assess both controls; a gain over static summaries alone cannot establish useful temporal ordering.',
           f"- **H3: does gaze add beyond oracle action/object/task context for correctness?** {fmt(h3)}; {verdict(h3)}. "
           'This is a diagnostic with ground-truth context, not an input condition available to an ordinary deployed assistant.',
           '', 'The central scene-grounded/structured-transfer hypothesis remains proposed. These tests evaluate angular gaze statistics '
           'with frozen ImageNet features and linear classifiers. They cannot validate symbolic reasoning, teacher transfer, wearer '
           'personalization, a causal controller, or mobile resource savings.',
           '', '## Data','',
           f"{len(audits)} recordings, {len(set(r['group'] for r in rows))} recording-prefix groups, {len(rows)} usable fine-action segments, "
           f"{sum(int(r['mistake'])==1 for r in rows)} mistakes. Total duration: {sum(float(r['duration_s']) for r in audits)/60:.1f} minutes.",
           '', f"Added {len(selected-set(plan['original_recordings']))} fully downloaded recordings to the initial 20. "
           f"Verified selected sensor/video payload: {actual_bytes/1e9:.3f} GB; total HoloAssist folder including preserved partials: {stored/1e9:.3f} GB, below the user's 400 GB cap.",
           '', f"The original expansion planned {len(plan['recordings'])} recordings ({plan['bytes']/1e9:.3f} GB payload). "
           f"Repeated Azure connection timeouts left {len(selection['excluded'])} recordings unavailable. "
           'The availability amendment was written before expanded model fitting. Completed transfers define this diagnostic sample; '
           'selection may favor shorter videos. Missing files and resumable partials remain listed in selection.json and the original plan.',
           '', '| Task domain | Recordings |','|---|---:|']
    lines += [f'| {domain} | {count} |' for domain,count in sorted(domains.items())]
    lines += ['', 'The added set excludes the original recording-prefix groups, but all available recordings are pooled for exploratory group CV. '
              'No claim of a fresh held-out replication follows. Recording prefixes are not verified wearer IDs. '
              'Task domains remain imbalanced and several new domains have very few recordings.',
              '', '## Budget probes','',
              '| Frames | Method | Verb macro-F1 (%) | Mistake AP (%) |','|---:|---|---:|---:|']
    for budget in [1,4]:
        for mode in colors:
            lines.append(f"| {budget} | {mode} | {100*result('verb',budget,mode)['score']:.2f} | {100*result('mistake',budget,mode)['score']:.2f} |")
    lines += ['', 'Control = RGB plus duration, gaze quality/sample coverage, and camera motion. Static adds angular position summaries; '
              'temporal adds speed/path/time-bin statistics; shuffled replaces those temporal statistics using one fixed within-segment permutation. '
              'No model receives correctness explanations, narration, speech, or future action labels.',
              '', '## Oracle context diagnostic','', '| Method | Mistake AP (%) |','|---|---:|']
    lines += [f"| {mode} | {100*result('mistake',4,mode)['score']:.2f} |" for mode in modes]
    lines += ['', 'Oracle controls one-hot encode verb, noun, task, and joint verb-noun identity using training-fold categories only. '
              'They control some action/class-prior shortcuts, but are not exact matched-pair or causal analyses. '
              'Unequal model feature counts and imperfect linear conditional modeling remain limitations.',
              '', '![Measured hypothesis tests](hypothesis_tests.png)',
              '', '## All paired contrasts','',
              '| Task | Contrast | Difference / conditional 95% interval |','|---|---|---:|']
    lines += [f"| {r['task']} | {r['contrast']} | {fmt(r)} |" for r in contrasts]
    lines += ['', 'All intervals use 1,000 recording-group bootstrap draws conditional on fixed out-of-fold predictions. '
              'They exclude refitting and hyperparameter-selection uncertainty and have no multiple-comparison correction. '
              'Treat positive findings as exploratory; task/participant generalization and independent replication remain necessary.',
              '', '## Reproducibility and limits','',
              'Fixed comparisons were written in [HYPOTHESIS_PROTOCOL.md](../../HYPOTHESIS_PROTOCOL.md) before the expanded models ran. '
              'Four-fold GroupKFold separates R-number prefixes. Training-fold-only PCA has 32 visual dimensions for each budget. '
              'Scaling and oracle encodings use training folds only; logistic regression has C=1, balanced class weights, and no tuning.',
              '', 'One-frame uses the 35%-of-segment image. Four-frame uses 10/35/60/85%, pooled as mean/std/last-minus-first. '
              'All four embeddings were precomputed offline for this comparison. Source presentation timestamps are checked for constant '
              'frame rate and video span is checked against released UTC timing. The eye CalibrationValid flag and causal pose association '
              'filter gaze rays; angular statistics are not validated fixations or tracked-object revisits.',
              '', 'Complete annotated boundaries and segment duration are oracle inputs. Mistake classification may observe the wearer after '
              'assistance; it is not early detection or intervention prediction. Actual error onset, intervention necessity, and latent intent '
              'are not established by these labels. One/four-frame differences are not token, energy, or causal scheduling results.',
              '', '```powershell',
              '.venv/Scripts/python.exe scripts/expand_holoassist_hypothesis.py',
              '.venv/Scripts/python.exe scripts/run_holoassist_pilot.py --plan data/external/holoassist/hypothesis_subset_plan.json --output results/holoassist_hypothesis --frame-budget-features --extract-only --available-only',
              '.venv/Scripts/python.exe scripts/run_holoassist_hypotheses.py',
              '.venv/Scripts/python.exe scripts/report_holoassist_hypotheses.py','```',
              '', 'Raw files have SHA-256 sidecars; provenance.json records checked hashes, byte offsets, and the source base URL. '
              'The full dataset was not downloaded. The original 20-recording report is preserved. '
              'Re-running extraction after new downloads will change the available set; preserve selection.json and use a new output folder for a larger follow-up.',
              '', 'Framework proposal: [HYPOTHESIS_AND_METHOD.md](../../HYPOTHESIS_AND_METHOD.md). '
              'Dataset and schema: [official HoloAssist release](https://holoassist.github.io/), '
              '[publisher annotation format](https://holoassist.github.io/data_links/README.html).']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Report written:',OUT/'REPORT.md')


if __name__=='__main__':
    main()
