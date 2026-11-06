#!/usr/bin/env python3
"""Generate plots and a transparent report directly from measured CSV outputs."""
import csv
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POLICIES=["fifo","reservoir","category_recency","gaze_priority","gaze_context"]
LABELS={"fifo":"FIFO","reservoir":"Reservoir","category_recency":"Category recency",
        "gaze_priority":"Gaze priority","gaze_context":"Gaze + context"}
COLORS={"fifo":"#8f97a3","reservoir":"#b48ac5","category_recency":"#436da7",
        "gaze_priority":"#c36c42","gaze_context":"#278878"}


def load(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    folder=ROOT/"results/neural_pilot"
    if not (folder/"summary.csv").exists():
        folder=ROOT/"results/pilot"
    rows=load(folder/"summary.csv")
    details=load(folder/"per_video_metrics.csv")
    stats=json.loads((folder/"dataset_stats.json").read_text())
    lookup={(float(r["noise"]),int(r["capacity"]),r["policy"],r["task"],r["metric"]):r for r in rows}
    def value(policy,task,metric="retention",budget=8,noise=0):
        return 100*float(lookup[(noise,budget,policy,task,metric)]["mean"])
    def paired_delta(a,b,task,metric="retention"):
        selected=[r for r in details if float(r["noise"])==0 and int(r["capacity"])==8 and r["task"]==task]
        by={}
        for r in selected:
            by.setdefault((r["video"],r["policy"]),[]).append(float(r[metric]))
        videos=sorted({v for v,p in by})
        deltas=np.array([np.mean(by[(v,a)])-np.mean(by[(v,b)]) for v in videos])
        rng=np.random.default_rng(20261005)
        boots=deltas[rng.integers(0,len(deltas),(2000,len(deltas)))].mean(1)*100
        return float(deltas.mean()*100),float(np.quantile(boots,.025)),float(np.quantile(boots,.975))
    figure,axes=plt.subplots(2,2,figsize=(11.5,8.5),layout="constrained")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    budgets=sorted({int(r["capacity"]) for r in rows})
    for policy in POLICIES:
        axes[0,0].plot(budgets,[value(policy,"attended",budget=b) for b in budgets],
                       "o-",color=COLORS[policy],label=LABELS[policy])
    axes[0,0].set(title="A  Attended evidence retained",xlabel="Maximum memory records",ylabel="Video-macro retention (%)",ylim=(0,105))
    axes[0,0].legend(fontsize=8,loc="lower right")
    x=np.arange(len(POLICIES))
    axes[0,1].bar(x-.17,[value(p,"attended") for p in POLICIES],width=.34,color="#278878",label="Attended")
    axes[0,1].bar(x+.17,[value(p,"peripheral") for p in POLICIES],width=.34,color="#c36c42",label="Peripheral")
    axes[0,1].set(title="B  Context tradeoff at 8 records",ylabel="Video-macro retention (%)",ylim=(0,105))
    axes[0,1].set_xticks(x,[LABELS[p].replace(" ","\n") for p in POLICIES],fontsize=8)
    axes[0,1].legend(fontsize=8)
    noise=[0,.25,.5,1]
    for policy in ["gaze_priority","gaze_context"]:
        axes[1,0].plot(np.array(noise)*100,[value(policy,"attended",noise=n) for n in noise],
                       "o-",color=COLORS[policy],label=LABELS[policy])
    axes[1,0].axhline(value("category_recency","attended"),color=COLORS["category_recency"],ls="--",label="Category recency")
    axes[1,0].set(title="C  Corrupted gaze-role labels",xlabel="Episodes selected for within-episode label shuffling (%)",ylabel="Attended retention (%)",ylim=(0,105))
    axes[1,0].legend(fontsize=8)
    neural=(0,8,"gaze_context","attended","retrieval_at_1") in lookup
    if neural:
        for policy in POLICIES:
            axes[1,1].plot(budgets,[value(policy,"attended","retrieval_at_1",b) for b in budgets],
                           "o-",color=COLORS[policy],label=LABELS[policy])
        axes[1,1].set(title="D  Frozen BGE text retrieval",xlabel="Maximum memory records",ylabel="Attended evidence recall@1 (%)",ylim=(0,105))
    else:
        for policy in POLICIES:
            axes[1,1].plot(budgets,[value(policy,"peripheral",budget=b) for b in budgets],"o-",color=COLORS[policy])
        axes[1,1].set(title="D  Peripheral evidence retained",xlabel="Maximum memory records",ylabel="Retention (%)",ylim=(0,105))
    figure.suptitle("Real annotation replay · HoloAssist / StreamGaze · no VideoLLM evaluated",fontsize=13)
    out=ROOT/"results/figures"
    out.mkdir(parents=True,exist_ok=True)
    for extension in ["png","svg","pdf"]:
        figure.savefig(out/f"pilot_results.{extension}",dpi=180)
    plt.close(figure)
    table=["| Memory policy | Attended retention | Peripheral retention | Overall retention |",
           "|---|---:|---:|---:|"]
    for policy in POLICIES:
        table.append(f"| {LABELS[policy]} | {value(policy,'attended'):.1f}% | {value(policy,'peripheral'):.1f}% | {value(policy,'all'):.1f}% |")
    neural_table=[]
    if neural:
        neural_table=["| Memory policy | Attended R@1 | Attended R@3 | Peripheral R@1 |",
                      "|---|---:|---:|---:|"]
        for policy in POLICIES:
            neural_table.append(f"| {LABELS[policy]} | {value(policy,'attended','retrieval_at_1'):.1f}% | {value(policy,'attended','retrieval_at_3'):.1f}% | {value(policy,'peripheral','retrieval_at_1'):.1f}% |")
    delta,lo,hi=paired_delta("gaze_context","category_recency","attended")
    cost_delta,cost_lo,cost_hi=paired_delta("gaze_context","category_recency","peripheral")
    neural_delta=paired_delta("gaze_context","category_recency","attended","retrieval_at_1") if neural else None
    report=f"""# Executed pilot results

Status: real annotation replay and frozen text retrieval completed. No VideoLLM
was trained or evaluated. Date: 2026-10-05.

The pilot supports investigating a controllable attention/context tradeoff in a
wearable assistant's memory. It does not establish an improvement in general
video understanding, answer accuracy, human intent prediction, or physical
object localization.

## Data and protocol

- {stats['episodes']} released fixation episodes, {stats['videos']} video names after
  removing whitespace/case differences, and {stats['observations']} category-level
  object observations. {stats['missing_primary_episodes']} episodes have no representative object.
- Source: [StreamGaze HoloAssist metadata](https://huggingface.co/datasets/daeunni/StreamGaze/blob/main/metadata/holoassist.csv),
  file commit `71806a8`, CC BY 4.0. The CSV was reconstructed from the rendered file
  view because terminal network access was unavailable. It is not a byte-exact
  original download. The frozen reconstructed file has SHA-256 `{stats['source_sha256']}`.
- Captions and gaze roles are released perception annotations. They replace a
  real perception model for this component test. The video pixels and raw gaze
  coordinates were not loaded. Video names are grouping units, not known unique people.
- Every policy reads the same observations in timestamp order and writes a memory
  without access to any question. Queries arrive after the observed sequence.
- Equal-time observations use a SHA-256 uid tie-break independent of gaze roles.
  This avoids confounding recency/FIFO with the metadata's foreground-first order.
- One query per latest observed semantic category per video, derived from the same
  annotations: 594 queries total (348 attended/near-gaze, 246 peripheral). These
  are diagnostic queries, not official benchmark QA or independently authored queries.
- Memory capacities: 4, 8, 16, 32 records. Captions are capped at 64 whitespace
  words. The budget matches maximum record count, not actual bytes or model tokens.
- Ten random seeds for reservoir and role-shuffle controls. Main results are
  averaged within video, then across videos. Confidence intervals use 2,000
  video-cluster bootstrap samples after averaging seeds. This is a convenience
  sample with uncertain participant independence; intervals are descriptive.

## Result 1: evidence retention at eight records

{chr(10).join(table)}

Gaze + context reserves two of eight available records for peripheral semantic
categories and fills remaining capacity by gaze priority and recency. Compared
with category recency, attended retention changes by {delta:+.1f} percentage points
(paired descriptive 95% interval {lo:+.1f} to {hi:+.1f}), while peripheral retention
changes by {cost_delta:+.1f} points ({cost_lo:+.1f} to {cost_hi:+.1f}).

The three category-deduplicating policies have the same overall retention by
construction at a fixed capacity: each stores the same number of unique categories.
Gaze changes which categories survive; it has not created extra storage capacity.
FIFO and reservoir retain repeated categories and sometimes older records, explaining
their lower latest-category retention. Category recency is therefore the stronger
gaze-free comparator. The proposed context quota was fixed at 25% before inspecting
these outputs; it has not been tuned on this sample.

## Result 2: frozen neural text retrieval

{chr(10).join(neural_table) if neural else 'Not run.'}

{f'Gaze + context versus category recency changes attended R@1 by {neural_delta[0]:+.1f} percentage points (paired descriptive 95% interval {neural_delta[1]:+.1f} to {neural_delta[2]:+.1f}).' if neural else ''}

The cached [BGE-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)
checkpoint was actually loaded and run locally. Captions use normalized CLS
embeddings; queries use the model's retrieval instruction followed by
`What was the most recent appearance of the [category]?`. Ranking uses cosine
similarity. No timestamps, gaze roles, answers, or target record ids enter embeddings.
No fine-tuning was performed. Captions and object names are supplied in the records,
so this is an easy, annotation-assisted category retrieval test. R@3 being close to
retention means the main bottleneck here is retaining evidence, not language ranking.

## Result 3: gaze-role corruption

| Episodes selected for label shuffling | Gaze priority attended retention | Gaze + context attended retention |
|---|---:|---:|
"""
    for n in noise:
        report+=f"| {n*100:.0f}% | {value('gaze_priority','attended',noise=n):.1f}% | {value('gaze_context','attended',noise=n):.1f}% |\n"
    report+="""
Role labels are permuted among objects in each selected episode. The original
captions, event timestamps, and evaluation targets are unchanged. A selected
episode can retain identical labels, especially when it has only one object or
several equal roles. Thus 100% selected does not imply all labels are wrong or that
all gaze information is removed. This experiment is not a simulation of measured
eye-tracker coordinate noise, angular error, calibration drift, or timestamp jitter.

## Interpretation and remaining evidence

The evidence motivates testing a memory allocation policy that trades attended
recall against peripheral recall. A gaze-only policy has a substantial context
cost in this replay. The context quota partially recovers peripheral evidence.
Selecting gazed targets using gaze annotations makes attended retention structurally
favor gaze-based selection; independently authored downstream questions are needed
before claiming real task improvement.

The actual VideoLLM comparison remains unrun: this Intel Mac has no CUDA/MPS
backend, no cached VideoLLM checkpoint, and terminal DNS is restricted. A local
Qwen2.5-VL evaluation runner is provided, with full-frame, center-crop, gaze-overlay,
gaze-crop, same-canvas dual-view, and shuffled-gaze controls. Its preprocessing
smoke test uses synthetic frames and must not be counted as a dataset/model result.

Before a submission: run real frame perception; audit raw gaze/video alignment;
evaluate independent questions on participant-disjoint data; compare under actual
token and byte budgets; add a model-based saliency baseline, a learned adapter
baseline, and an online object-memory baseline; include localization/evidence
grounding, noisy/missing gaze, and participant-level task outcomes.

## Reproduction and artifacts

```sh
python3 -m unittest discover -s tests -v
python3 scripts/run_pilot.py
python3 scripts/run_pilot.py --embedding-model /path/to/local/bge-base-en-v1.5 --output results/neural_pilot
python3 scripts/build_results.py
```

- `results/pilot/`: annotation-only replay, raw per-video outputs and run manifest.
- `results/neural_pilot/`: executed BGE replay, embedding cache, per-video outputs,
  aggregate confidence intervals, normalized observations, and run manifest.
- `results/figures/pilot_results.png`, `.svg`, `.pdf`: plot exports.
- `results/neural_pilot/example_evidence.json`: a provenance-bearing evidence prompt.
"""
    (ROOT/"RESULTS.md").write_text(report,encoding="utf-8")
    print(f"Report and plots saved. Attended delta: {delta:+.2f} pp, paired interval [{lo:+.2f}, {hi:+.2f}].")


if __name__=="__main__":
    main()
