#!/usr/bin/env python3
"""Real-annotation component replay. No VideoLLM or QA benchmark is evaluated."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from gaze_memory.core import (BoundedMemory, Observation, corrupt_roles,
                              evidence_prompt, load_streamgaze_metadata, save_json)

POLICIES = ["fifo", "reservoir", "category_recency", "gaze_priority", "gaze_context"]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, metric):
    """Video-cluster bootstrap; seeds are averaged before resampling videos."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["video"], []).append(row[metric])
    values = np.array([np.mean(v) for v in grouped.values()])
    rng = np.random.default_rng(20261005)
    boots = values[rng.integers(0, len(values), (2000, len(values)))].mean(1)
    return float(values.mean()), float(np.quantile(boots, .025)), float(np.quantile(boots, .975))


def neural_embeddings(observations, model_path, cache):
    # Both query and caption embeddings are frozen; no fine-tuning or external call.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("USE_TF", "0")
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4)
    captions = [o.text for o in observations]
    queries = [f"Represent this sentence for searching relevant passages: What was the most recent appearance of the {o.object_name}?" for o in observations]
    fingerprint = hashlib.sha256(json.dumps([captions, queries, str(model_path)]).encode()).hexdigest()
    if cache.exists():
        existing = np.load(cache, allow_pickle=False)
        if str(existing["fingerprint"]) == fingerprint:
            return existing["captions"], existing["queries"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    def encode(texts):
        result = []
        for i in range(0, len(texts), 16):
            batch = tokenizer(texts[i:i+16], padding=True, truncation=True,
                              max_length=96, return_tensors="pt")
            with torch.inference_mode():
                vectors = model(**batch).last_hidden_state[:, 0]
                vectors = torch.nn.functional.normalize(vectors, dim=-1)
            result.append(vectors.numpy())
            if i % 160 == 0:
                print(f"Embedding {i}/{len(texts)}", flush=True)
        return np.concatenate(result)
    c, q = encode(captions), encode(queries)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, captions=c, queries=q, fingerprint=fingerprint)
    return c, q


def evaluate(observations, capacities, seeds, context_fraction, embedding_pair=None):
    groups = {}
    for obs in observations:
        groups.setdefault(obs.video, []).append(obs)
    index = {o.uid: i for i, o in enumerate(observations)}
    details = []
    for noise in [0.0, .25, .5, 1.0]:
        for seed in range(seeds):
            input_groups = {}
            for obs in corrupt_roles(observations, noise, seed + 1000):
                input_groups.setdefault(obs.video, []).append(obs)
            for video, truth in groups.items():
                # All queries arrive after the prefix has been compressed without
                # question access. End-of-sequence replay, not a streaming QA score.
                query_time = max(o.end for o in truth) + 1
                latest = {}
                for obs in truth:
                    latest[obs.object_name] = obs
                targets = list(latest.values())
                for capacity in capacities:
                    for policy in POLICIES:
                        # Gaze-independent policies do not change under role noise.
                        if noise and policy not in {"gaze_priority", "gaze_context"}:
                            continue
                        memory = BoundedMemory(capacity, policy, seed, context_fraction)
                        for obs in input_groups[video]:
                            memory.add(obs)
                        records = memory.snapshot(query_time)
                        kept = {o.uid for o in records}
                        for task, task_targets in [
                            ("all", targets),
                            ("attended", [o for o in targets if o.role != "peripheral"]),
                            ("peripheral", [o for o in targets if o.role == "peripheral"]),
                        ]:
                            if not task_targets:
                                continue
                            hits = [o.uid in kept for o in task_targets]
                            hit1, hit3 = [], []
                            if embedding_pair is not None:
                                caption_vecs, query_vecs = embedding_pair
                                candidate_ids = [index[o.uid] for o in records]
                                for target in task_targets:
                                    scores = caption_vecs[candidate_ids] @ query_vecs[index[target.uid]]
                                    order = np.argsort(-scores, kind="stable")
                                    ranked = [records[i].uid for i in order]
                                    hit1.append(target.uid in ranked[:1])
                                    hit3.append(target.uid in ranked[:3])
                            details.append({
                                "video": video, "seed": seed, "noise": noise,
                                "capacity": capacity, "policy": policy, "task": task,
                                "n_queries": len(task_targets),
                                "retention": float(np.mean(hits)),
                                "retrieval_at_1": float(np.mean(hit1)) if hit1 else "",
                                "retrieval_at_3": float(np.mean(hit3)) if hit3 else "",
                                "stored_records": len(records),
                                "stored_words": sum(len(o.text.split()) for o in records),
                                "input_records": len(truth),
                            })
    return details


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, default=ROOT/"data/raw/streamgaze_holoassist_web.csv")
    p.add_argument("--output", type=Path, default=ROOT/"results/pilot")
    p.add_argument("--capacities", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--context-fraction", type=float, default=.25)
    p.add_argument("--embedding-model", type=Path)
    args = p.parse_args()
    observations, stats = load_streamgaze_metadata(args.metadata)
    save_json(args.output/"dataset_stats.json", stats)
    write_csv(args.output/"observations.csv", [asdict(o) for o in observations])
    embeddings = None
    if args.embedding_model:
        embeddings = neural_embeddings(observations, args.embedding_model,
                                       args.output/"embeddings.npz")
    details = evaluate(observations, args.capacities, args.seeds,
                       args.context_fraction, embeddings)
    write_csv(args.output/"per_video_metrics.csv", details)
    groups = {}
    for row in details:
        groups.setdefault((row["noise"], row["capacity"], row["policy"], row["task"]), []).append(row)
    summary = []
    for (noise, capacity, policy, task), rows in sorted(groups.items()):
        metrics = ["retention"] + (["retrieval_at_1", "retrieval_at_3"] if embeddings else [])
        for metric in metrics:
            mean, lo, hi = summarize(rows, metric)
            summary.append({"noise": noise, "capacity": capacity, "policy": policy,
                            "task": task, "metric": metric, "mean": mean,
                            "ci_low": lo, "ci_high": hi,
                            "videos": len({r["video"] for r in rows}),
                            "queries_per_seed": sum(r["n_queries"] for r in rows if r["seed"] == 0)})
    write_csv(args.output/"summary.csv", summary)
    save_json(args.output/"run_manifest.json", {
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "config": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "data": stats,
        "evaluation": "annotation replay; derived category queries; no VideoLLM accuracy",
        "causality": "write memory before questions; only observations ending before query_time",
        "budget": "equal maximum record capacity; captions capped at 64 words; NOT equal measured tokens/bytes",
        "equal_timestamp_order": "SHA-256 of observation uid, independent of role, question, and answer",
        "confidence_interval": "2000 video-cluster bootstrap samples after seed averaging; descriptive convenience sample",
        "packages": {name: importlib.metadata.version(name) for name in
                     (["numpy", "torch", "transformers"] if embeddings is not None else ["numpy"])},
        "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in [ROOT/"gaze_memory/core.py", Path(__file__).resolve()]},
        "embedding_protocol": "BGE normalized CLS, 96-token limit, cosine ranking" if embeddings is not None else None,
    })
    first_video = observations[0].video
    example = [o for o in observations if o.video == first_video]
    memory = BoundedMemory(8, "gaze_context")
    for o in example:
        memory.add(o)
    save_json(args.output/"example_evidence.json", {
        "prompt": evidence_prompt("What did the battery look like when I last saw it?",
                                  memory.snapshot(max(o.end for o in example) + 1))
    })
    print(json.dumps(stats, indent=2), flush=True)
    print(f"Saved {len(summary)} aggregate metric rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
