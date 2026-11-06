#!/usr/bin/env python3
"""Plot real behavior results only; refuses synthetic/incomplete executions."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        r["macro_f1_present"] = float(r["macro_f1_present"])
        if "shots" in r:
            r["shots"] = int(r["shots"])
    return rows


def wearer_summary(rows):
    table = {}
    for row in rows:
        table.setdefault(row["subject"], []).append(row["macro_f1_present"])
    values = np.array([np.mean(v) for v in table.values()])
    if not len(values):
        return None
    rng = np.random.default_rng(0)
    boot = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
    return values.mean(), *np.quantile(boot, [0.025, 0.975]), len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.run / "execution_status.json").read_text())
    if status.get("research_accuracy_available") is not True or status.get("status") != "real_feature_experiment_completed":
        parser.error("No completed real-data experiment: synthetic or blocked outputs cannot become research plots")
    manifest = json.loads((args.run / "run_manifest.json").read_text())
    if manifest.get("synthetic") is not False:
        parser.error("Research plots require explicitly nonsynthetic provenance")
    recognition = read_csv(args.run / "recognition.csv")
    personal = read_csv(args.run / "personalization.csv") if (args.run / "personalization.csv").exists() else []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    methods = [m for m in ("visual", "quality_control", "static_gaze", "shuffled_gaze", "temporal_gaze", "gaze_only")
               if any(r["mode"] == m for r in recognition)]
    labels = {"visual": "Vision", "quality_control": "Quality only", "static_gaze": "Static gaze",
              "shuffled_gaze": "Shuffled gaze", "temporal_gaze": "Temporal gaze", "gaze_only": "Gaze only"}
    summaries = [wearer_summary([r for r in recognition if r["mode"] == m]) for m in methods]
    axes[0].bar(range(len(methods)), [100 * s[0] for s in summaries], color="#376a86")
    axes[0].errorbar(range(len(methods)), [100 * s[0] for s in summaries],
                     yerr=np.array([[100 * (s[0] - s[1]) for s in summaries],
                                    [100 * (s[2] - s[0]) for s in summaries]]), fmt="none", color="black", capsize=3)
    axes[0].set_xticks(range(len(methods)), [labels[m] for m in methods], rotation=25, ha="right")
    axes[0].set_title(f"Held-out-wearer {manifest['task']}")
    axes[0].set_ylabel("Wearer-macro F1, present classes (%)")
    axes[0].set_ylim(0, 100)
    # Use a fixed common cohort across both modes and every requested shot.
    max_shots = max((r["shots"] for r in personal if r["adaptation"] == "personal"), default=0)
    cohort = {r["subject"] for r in personal if r["mode"] == "visual" and r["shots"] == max_shots and r["adaptation"] == "personal"}
    cohort &= {r["subject"] for r in personal if r["mode"] == "temporal_gaze" and r["shots"] == max_shots and r["adaptation"] == "personal"}
    for mode, color in (("visual", "#777777"), ("temporal_gaze", "#c16e32")):
        shots = sorted({r["shots"] for r in personal if r["mode"] == mode and r["subject"] in cohort
                        and r["adaptation"] in {"generic", "personal"}})
        summaries = [wearer_summary([r for r in personal if r["mode"] == mode and r["shots"] == k
                                    and r["subject"] in cohort and r["adaptation"] in {"generic", "personal"}]) for k in shots]
        if shots:
            axes[1].plot(shots, [100 * s[0] for s in summaries], "o-", color=color, label=labels[mode])
            axes[1].fill_between(shots, [100 * s[1] for s in summaries], [100 * s[2] for s in summaries], color=color, alpha=0.15)
    if cohort:
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No common eligible personalization cohort", ha="center", transform=axes[1].transAxes)
    axes[1].set_title(f"Later-session personalization (n={len(cohort)})")
    axes[1].set_xlabel("Labeled, nonoverlapping early-recording examples")
    axes[1].set_ylabel("Wearer-macro F1, present classes (%)")
    axes[1].set_ylim(0, 100)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{manifest['dataset']}: real frozen-feature prototype; not VideoLLM or energy results", fontsize=10)
    fig.tight_layout()
    folder = args.run / "figures"
    folder.mkdir(exist_ok=False)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(folder / ("behavior_personalization." + extension), dpi=200, bbox_inches="tight")
    print(f"Real-data figures: {folder}")


if __name__ == "__main__":
    main()
