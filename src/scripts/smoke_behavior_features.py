#!/usr/bin/env python3
"""Actual frozen-CLIP/ffmpeg test on an explicitly synthetic existing clip."""
import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    video = ROOT / "results/preprocessing_smoke/source/test.mp4"
    if not video.is_file():
        parser.error("Prior explicitly synthetic preprocessing clip is required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "results/behavior_feature_smoke" / stamp
    output.mkdir(parents=True, exist_ok=False)
    gaze = output / "synthetic_gaze.csv"
    with gaze.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp_s", "x", "y", "valid"])
        for t in np.linspace(0, 2, 61):
            writer.writerow([t, 0.3 + 0.05 * t, 0.6, 1])
    manifest = {"dataset": "SYNTHETIC CLIP EXTRACTION FIXTURE — NOT EGTEA", "synthetic": True,
                "task": "recognition", "classes": ["fixture0", "fixture1"],
                "audit": {k: True for k in ("alignment_verified", "subject_identity_verified",
                                           "session_order_verified", "labels_independent_of_gaze")},
                "provenance": "Test-only artificial clip/coordinates/identities, not a dataset sample.",
                "sessions": [{"subject_id": f"synthetic{s}", "session_id": f"fixture{s}", "order": 0}
                             for s in range(3)],
                "samples": [{"id": f"SYNTHETIC{s}", "subject_id": f"synthetic{s}", "session_id": f"fixture{s}",
                             "start_s": 0, "cutoff_s": 1.85, "label": s % 2,
                             "video": str(video), "gaze_csv": str(gaze)} for s in range(3)]}
    path = output / "windows.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    subprocess.run([sys.executable, str(ROOT / "scripts/extract_behavior_features.py"),
                    "--windows", str(path), "--checkpoint", str(args.checkpoint), "--frames", "3",
                    "--output", str(output / "features"), "--synthetic-fixture"], check=True)
    for file in (output / "features").glob("*.npz"):
        with np.load(file, allow_pickle=False) as a:
            assert a["visual"].shape == (3, 512)
            assert np.isfinite(a["visual"]).all()
            assert np.allclose(np.linalg.norm(a["visual"], axis=1), 1, atol=1e-5)
            assert a["frame_time"].max() <= 1.85
            assert a["gaze_time"].max() <= 1.85
    report = {"status": "synthetic_frozen_clip_extraction_passed", "research_accuracy_available": False,
              "checkpoint": str(args.checkpoint), "feature_dimension": 512, "extracted_windows": 3,
              "elapsed_seconds": time.perf_counter() - started,
              "note": "Actual pretrained CLIP inference + ffmpeg on synthetic footage; not EGTEA validation or device benchmark."}
    (output / "execution_status.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "REPORT.md").write_text("# Synthetic implementation test, not research results\n\n"
                                      "Actual cached CLIP loading/inference and causal ffmpeg extraction passed.\n"
                                      "No EGTEA examples or behavioral accuracy were evaluated.\n")
    print(f"Frozen-CLIP synthetic extraction passed: {output}")


if __name__ == "__main__":
    main()
