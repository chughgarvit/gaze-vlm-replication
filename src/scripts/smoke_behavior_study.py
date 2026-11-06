#!/usr/bin/env python3
"""Explicit synthetic gradient-training/adaptation test; no dataset inference."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "results/behavior_smoke" / stamp
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    sessions, samples = [], []
    for subject in range(4):
        for order in range(2):
            session = f"SYNTHETIC-person{subject}-recording{order}"
            sessions.append({"subject_id": f"synthetic{subject}", "session_id": session, "order": order})
            for clip in range(8):
                label = clip % 2
                start, cutoff = clip * 3.0, clip * 3.0 + 2.0
                frame_time = np.linspace(start, cutoff, 8)
                gaze_time = np.linspace(start, cutoff, 61)
                # Fixture embeddings and trajectories are invented, not CLIP or EGTEA.
                visual = rng.normal(0, 0.2, (8, 16)).astype(np.float32)
                visual[:, label] += 0.3
                xy = np.column_stack([np.linspace(0.2, 0.8, 61) if label else np.linspace(0.8, 0.2, 61),
                                      np.full(61, 0.4 + subject * 0.03)]).astype(np.float32)
                filename = f"fixture-{subject}-{order}-{clip}.npz"
                np.savez_compressed(output / filename, visual=visual, frame_time=frame_time,
                                    gaze_time=gaze_time, gaze_xy=xy, gaze_valid=np.ones(61, bool))
                samples.append({"id": f"SYNTHETIC-{subject}-{order}-{clip}", "subject_id": f"synthetic{subject}",
                                "session_id": session, "start_s": start, "cutoff_s": cutoff,
                                "label": label, "features": filename})
    manifest = {"dataset": "SYNTHETIC IMPLEMENTATION FIXTURE — NOT EGTEA", "synthetic": True,
                "task": "recognition", "classes": ["fixture_class_0", "fixture_class_1"],
                "audit": {k: True for k in ("alignment_verified", "subject_identity_verified",
                                           "session_order_verified", "labels_independent_of_gaze")},
                "provenance": "Artificial arrays for wiring/gradient tests; not biological or camera observations.",
                "sessions": sessions, "samples": samples}
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run([sys.executable, str(ROOT / "scripts/run_behavior_study.py"),
                    "--manifest", str(path), "--output", str(output / "training"), "--synthetic-fixture",
                    "--epochs", "4", "--hidden", "16", "--seeds", "0", "--shots", "0", "1", "5",
                    "--adapt-steps", "3"], check=True)
    status = json.loads((output / "training/execution_status.json").read_text())
    assert status["status"] == "synthetic_training_smoke_passed"
    assert status["research_accuracy_available"] is False
    assert status["trained_models"] == 24
    splits = json.loads((output / "training/splits.json").read_text())
    for fold in splits:
        assert set(fold["support_ids"]).isdisjoint(fold["query_ids"])
        assert fold["test_subject"] not in fold["train_subjects"]
        assert fold["dev_subject"] not in fold["train_subjects"]
    print(f"Synthetic training/adaptation checks passed: {output}")


if __name__ == "__main__":
    main()
