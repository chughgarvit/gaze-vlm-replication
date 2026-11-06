#!/usr/bin/env python3
"""Train subject-disjoint gaze fusion baselines and later-session personalization.

Consumes audited, frozen visual features plus timestamped observed gaze. This is
a supervised recognition/anticipation prototype, not a completed VideoLLM,
self-supervised representation-transfer, or edge-energy experiment.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gaze_behavior.data import (MODES, build_sequence, class_matched_support, load_manifest,
                                personalization_split, subject_split)
from gaze_behavior.model import (adapt_head, embeddings, metrics, normalization,
                                 paired_interval, prepare, train_model)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(args, output):
    manifest, examples = load_manifest(args.manifest, args.synthetic_fixture)
    subjects = sorted({e["subject_id"] for e in examples})
    labels = np.array([e["label"] for e in examples])
    classes = len(manifest["classes"])
    rows, predictions, adaptation, split_records, training = [], [], [], [], []
    folds = subjects[:args.max_folds] if args.max_folds else subjects
    parameter_count = None
    for test_subject in folds:
        dev_subject = subjects[(subjects.index(test_subject) + 1) % len(subjects)]
        train, dev, target = subject_split(examples, test_subject, dev_subject)
        support, query = personalization_split(examples, target)
        wrong_support, _ = personalization_split(examples, dev)
        query = query if support else target
        split_records.append({"test_subject": test_subject, "dev_subject": dev_subject,
                              "train_subjects": sorted({examples[i]["subject_id"] for i in train}),
                              "support_ids": [examples[i]["id"] for i in support],
                              "query_ids": [examples[i]["id"] for i in query],
                              "personalization_eligible": bool(support),
                              "wrong_user_support_ids": [examples[i]["id"] for i in wrong_support]})
        for seed in args.seeds:
            for mode in args.modes:
                sequences = [build_sequence(e["arrays"], mode, e["id"], seed) for e in examples]
                mean, std = normalization(sequences, train)
                sequences = prepare(sequences, mean, std)
                started = time.perf_counter()
                model, log = train_model(sequences, labels, train, dev, classes, hidden=args.hidden,
                                         epochs=args.epochs, batch_size=args.batch_size, seed=seed)
                parameter_count = sum(p.numel() for p in model.parameters())
                code = f"fold{subjects.index(test_subject):02d}_{mode}_seed{seed}"
                write_json(output / (code + "_training.json"), log)
                # Save safe, tensor-only weights; no pickle-based model artifact.
                from safetensors.torch import save_file
                save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(output / (code + ".safetensors")))
                np.savez_compressed(output / (code + "_normalization.npz"), mean=mean, std=std)
                embedding = embeddings(model, sequences)
                with torch.no_grad():
                    pred = model.head(embedding[query]).argmax(1).numpy()
                result = metrics(labels[query], pred, classes)
                train_classes = set(labels[train])
                coverage = {"training_classes": len(train_classes),
                            "query_unseen_class_samples": int(sum(labels[i] not in train_classes for i in query))}
                rows.append({"subject": test_subject, "seed": seed, "mode": mode, **result, **coverage})
                for i, p in zip(query, pred):
                    predictions.append({"id": examples[i]["id"], "subject": test_subject,
                                        "seed": seed, "mode": mode, "label": int(labels[i]), "prediction": int(p)})
                training.append({"subject": test_subject, "seed": seed, "mode": mode,
                                 "epochs": len(log), "parameters": parameter_count,
                                 "train_seconds": time.perf_counter() - started,
                                 "input_dimension": sequences[0].shape[1]})
                for shots in args.shots:
                    if shots == 0:
                        if not support:
                            continue
                        adaptation.append({"subject": test_subject, "seed": seed, "mode": mode,
                                           "shots": 0, "adaptation": "generic", "support_classes": 0,
                                           **result})
                        continue
                    if len(support) < shots:
                        continue
                    controls = [("personal", support[:shots]), ("personal_bias_only", support[:shots])]
                    if len(wrong_support) >= shots:
                        controls.append(("wrong_user", wrong_support[:shots]))
                    matched = class_matched_support(labels, support[:shots], wrong_support)
                    if matched:
                        controls.append(("wrong_user_classmatched", matched))
                    # Equal number of gradient updates/data points; already-seen
                    # population examples are a rehearsal control, not new labels.
                    if len(train) >= shots:
                        controls.append(("population_rehearsal", train[:shots]))
                    for name, selected in controls:
                        head = adapt_head(model.head, embedding[selected], torch.from_numpy(labels[selected]),
                                          steps=args.adapt_steps, lr=args.adapt_lr, anchor=args.adapt_anchor,
                                          bias_only=name == "personal_bias_only")
                        with torch.no_grad():
                            p = head(embedding[query]).argmax(1).numpy()
                        adaptation.append({"subject": test_subject, "seed": seed, "mode": mode,
                                           "shots": shots, "adaptation": name,
                                           "support_classes": int(len(set(labels[selected]))),
                                           **metrics(labels[query], p, classes)})
                print(f"Finished {test_subject}/{mode}/seed{seed}; {len(query)} later-session queries", flush=True)
    contrasts = []
    for baseline in ("visual", "quality_control", "static_gaze", "shuffled_gaze"):
        if baseline in args.modes and "temporal_gaze" in args.modes:
            contrasts.append({"comparison": f"temporal_gaze - {baseline}",
                              **paired_interval(rows, "temporal_gaze", baseline)})
    personal_contrasts = []
    for mode in args.modes:
        for shots in (s for s in args.shots if s):
            selected = [dict(r, mode=r["adaptation"]) for r in adaptation
                        if r["mode"] == mode and (r["shots"] == shots or r["shots"] == 0)]
            for control in ("generic", "wrong_user_classmatched", "personal_bias_only", "wrong_user", "population_rehearsal"):
                personal_contrasts.append({"mode": mode, "shots": shots, "control": control,
                                           **paired_interval(selected, "personal", control)})
    write_csv(output / "recognition.csv", rows)
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "personalization.csv", adaptation)
    write_json(output / "splits.json", split_records)
    write_json(output / "training.json", training)
    interactions = []
    # Does gaze improve the gain FROM personalization, not merely generic score?
    for shots in (s for s in args.shots if s):
        table = {}
        for r in adaptation:
            if r["mode"] in {"visual", "temporal_gaze"} and r["adaptation"] in {"generic", "personal"}:
                if r["shots"] in {0, shots}:
                    key = (r["subject"], r["seed"], r["mode"], r["adaptation"])
                    table[key] = r["macro_f1_present"]
        gains = []
        for subject in folds:
            for seed in args.seeds:
                for mode in ("visual", "temporal_gaze"):
                    generic, personal = (subject, seed, mode, "generic"), (subject, seed, mode, "personal")
                    if generic in table and personal in table:
                        gains.append({"subject": subject, "seed": seed, "mode": mode,
                                      "macro_f1_present": table[personal] - table[generic]})
        interactions.append({"shots": shots, "comparison": "personalization_gain(temporal_gaze) - personalization_gain(visual)",
                             **paired_interval(gains, "temporal_gaze", "visual")})
    write_json(output / "contrasts.json", {"representation": contrasts, "personalization": personal_contrasts,
                                          "gaze_personalization_interaction": interactions})
    with args.manifest.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = {"manifest_sha256": digest, "dataset": manifest.get("dataset"), "task": manifest["task"],
                  "synthetic": manifest["synthetic"], "feature_provenance": manifest.get("provenance"),
                  "classes": manifest["classes"], "all_subjects": subjects, "tested_subjects": folds,
                  "feature_sha256": {e["id"]: e["feature_sha256"] for e in examples},
                  "hyperparameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
                  "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), *sorted((ROOT / "gaze_behavior").glob("*.py"))]},
                  "metrics_note": "Present-ground-truth-class macro metrics within wearer; average seeds then wearers. "
                                  "CIs cluster wearers, not clips. Class-unmatched wrong-user control is exploratory."}
    write_json(output / "run_manifest.json", provenance)
    return {"status": "synthetic_training_smoke_passed" if args.synthetic_fixture else "real_feature_experiment_completed",
            "research_accuracy_available": not args.synthetic_fixture,
            "tested_subjects": len(folds), "model_parameters": parameter_count,
            "trained_models": len(rows), "adaptation_rows": len(adaptation), "contrasts": contrasts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/external/egtea_gaze_plus/features/manifest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--synthetic-fixture", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 1, 5, 10, 20])
    parser.add_argument("--max-folds", type=int, default=0, help="0 means all wearers; otherwise a convenience pilot")
    parser.add_argument("--adapt-steps", type=int, default=20)
    parser.add_argument("--adapt-lr", type=float, default=0.01)
    parser.add_argument("--adapt-anchor", type=float, default=1.0)
    args = parser.parse_args()
    if min(args.epochs, args.hidden, args.batch_size, args.adapt_steps) <= 0 or min(args.shots) < 0 or args.max_folds < 0:
        parser.error("Invalid count/budget")
    if args.adapt_lr <= 0 or args.adapt_anchor < 0:
        parser.error("Invalid adaptation hyperparameters")
    if len(set(args.modes)) != len(args.modes) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate modes/seeds would repeat the same experiment")
    args.shots = sorted(set(args.shots) | {0})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "results/behavior_study" / stamp
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    try:
        if not args.manifest.exists():
            status = {"status": "blocked_missing_audited_features", "research_accuracy_available": False,
                      "required_manifest": str(args.manifest),
                      "note": "Download EGTEA, audit gaze alignment/identity/chronology, and export real scene features first."}
            code = 1
        else:
            status, code = execute(args, output), 0
    except Exception as exc:
        status = {"status": "incomplete", "error_type": type(exc).__name__,
                  "error": str(exc), "research_accuracy_available": False}
        code = 1
    write_json(output / "execution_status.json", status)
    fixture = args.synthetic_fixture
    report = ["# SYNTHETIC IMPLEMENTATION TEST — NOT RESEARCH RESULTS" if fixture else "# Behavior study execution",
              "", f"Status: `{status['status']}`.", "",
              "Research accuracy available: " + str(status["research_accuracy_available"]) + ".", "",
              "The frozen scene-feature / unidirectional GRU prototype and labeled head adaptation are implemented. "
              "This is not a completed VideoLLM, cross-task representation-transfer, or systems experiment.", "",
              "See execution_status.json, run_manifest.json, splits.json, and contrasts.json when present.", "",
              "Synthetic CSV metrics, if present, are fixture diagnostics only and must never be cited as EGTEA findings."
              if fixture else "Do not infer gaze benefits from implementation success; interpret paired real-data contrasts."]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    print(f"{status['status']}: {output}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
