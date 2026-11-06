#!/usr/bin/env python3
"""Download, safely import, and evaluate a small real original/overlay pilot.

Default: 3 EGTEA videos, up to 9 official QA samples, 6 observed frames, local
Qwen2.5-VL-3B, CPU, no paid APIs. A blocked download never yields an accuracy.
"""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gaze_memory.benchmark import (choose_frames, extract_frames, extract_selected_videos,
                                   file_hash, import_samples, match_overlay_frames, probe_timeline)
from gaze_memory.core import save_json


def write_report(output, manifest, summary, predictions):
    data_kind = manifest["data_kind"]
    lines = ["# Original versus publisher gaze-overlay pilot", "",
             f"Status: {summary['status']}. Data kind: `{data_kind}`.", "",
             f"Samples: {len(manifest['samples'])}; video groups: "
             f"{len({s['video'] for s in manifest['samples']})}.", "",
             "Identical questions, answer options, text instructions, frame counts, and "
             "448×448 image canvases are used in both conditions. Source presentation "
             "timestamps must match causally within 1 ms.", ""]
    if data_kind == "synthetic_test_fixture":
        lines += ["Both test archives contain an identical synthetic clip. No publisher "
                  "gaze overlay or measured eye-tracker data was used; this checks wiring only.", ""]
    else:
        lines += ["The publisher overlay is used as provided, not reconstructed from semantic "
                  "object labels. Its visual contents still require an audit.", ""]
    if summary["status"] == "model_evaluated":
        paired = {}
        for row in predictions:
            paired.setdefault(row["sample_id"], {})[row["mode"]] = row
        if any(set(pair) != {"full", "publisher_overlay"} for pair in paired.values()):
            raise ValueError("Report requires complete paired predictions")
        full = sum(pair["full"]["correct"] for pair in paired.values())
        overlay = sum(pair["publisher_overlay"]["correct"] for pair in paired.values())
        n = len(paired)
        gains = sum(not p["full"]["correct"] and p["publisher_overlay"]["correct"] for p in paired.values())
        losses = sum(p["full"]["correct"] and not p["publisher_overlay"]["correct"] for p in paired.values())
        lines += ["| Condition | Correct / total | Accuracy |", "|---|---:|---:|",
                  f"| Original | {full}/{n} | {100*full/n:.1f}% |",
                  f"| Publisher gaze overlay | {overlay}/{n} | {100*overlay/n:.1f}% |", "",
                  f"Paired accuracy change: {100*(overlay-full)/n:+.1f} percentage points. "
                  f"Overlay-only correct: {gains}; original-only correct: {losses}.", "",
                  "Per-example actual input/visual token counts, generated answers, scoring, "
                  "and latency are in `evaluation/predictions.json`.", ""]
    else:
        lines += ["No model was run. No predictions, accuracy, or improvement claim is reported.", ""]
    lines += ["This is a deterministic convenience subset, not a participant-disjoint test "
              "or a full StreamGaze reproduction. Earliest questions from alphabetically "
              "selected videos are sampled; selection bias and small sample size limit "
              "inference. Sparse uniformly sampled frames can miss required events. "
              "Publisher overlays require a visual audit for additional markings or label "
              "leakage before research conclusions. No gaze-crop or bounded-memory "
              "VideoLLM result follows from this baseline.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/external/streamgaze")
    parser.add_argument("--model", type=Path, default=ROOT / "data/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-videos", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=9)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--max-gb", type=float, default=32)
    parser.add_argument("--skip-download", action="store_true", help="Use already supplied local assets")
    parser.add_argument("--dry-run", action="store_true", help="Real-data preprocessing only, no model scoring")
    parser.add_argument("--synthetic-fixture", action="store_true", help="Explicitly label a test-only dry run")
    args = parser.parse_args()
    if min(args.max_videos, args.max_samples, args.frames) < 1:
        parser.error("Pilot limits must be positive")
    if args.synthetic_fixture and not (args.dry_run and args.skip_download):
        parser.error("Synthetic fixtures require --dry-run and --skip-download; they cannot be scored")
    if not args.skip_download and args.dataset_root.resolve() != (ROOT / "data/external/streamgaze").resolve():
        parser.error("Custom dataset roots require --skip-download")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or ROOT / "results/end_to_end" / timestamp).resolve()
    if output.exists():
        parser.error("Output exists; choose a fresh directory (download caches are reused)")
    output.mkdir(parents=True)
    state = {"status": "running", "stage": "preflight", "output": str(output),
             "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
             "accuracy_available": False}
    save_json(output / "execution_status.json", state)
    try:
        missing_tools = [name for name in ["ffprobe", "ffmpeg"] if shutil.which(name) is None]
        missing_modules = [name for name in ["numpy", "PIL"] if importlib.util.find_spec(name) is None]
        if not args.dry_run:
            missing_modules += [name for name in ["torch", "transformers"] if importlib.util.find_spec(name) is None]
        if missing_tools or missing_modules:
            raise RuntimeError(f"Missing local tools/modules: {missing_tools + missing_modules}")
        if not args.skip_download:
            state["stage"] = "download"
            save_json(output / "execution_status.json", state)
            selection = "videos" if args.dry_run else "pilot"
            print(f"Downloading/checking {selection} assets...", flush=True)
            subprocess.run([sys.executable, str(ROOT / "scripts/download_assets.py"),
                            "--assets", selection, "--max-gb", str(args.max_gb), "--execute"], check=True)
        if not args.dry_run and not (args.model / "model.safetensors.index.json").is_file():
            raise FileNotFoundError("No complete local Qwen2.5-VL-3B checkpoint is available")
        state["stage"] = "import_and_extract"
        save_json(output / "execution_status.json", state)
        samples, qa_hashes = import_samples(args.dataset_root / "qa", args.max_videos,
                                            args.max_samples, args.frames)
        names = {sample["video"] for sample in samples}
        print(f"Importing {len(samples)} questions from {len(names)} videos; safely extracting selected files...", flush=True)
        video_paths = {}
        provenance = {"expected_dataset_repository": "daeunni/StreamGaze", "qa_sha256": qa_hashes}
        for role, archive_name in [("original", "videos_egtea_original.tar"),
                                   ("overlay", "videos_egtea_viz.tar")]:
            archive = args.dataset_root / archive_name
            if not archive.is_file():
                raise FileNotFoundError(f"Missing archive: {archive_name}")
            video_paths[role] = extract_selected_videos(archive, names, output / "videos" / role)
            provenance[role + "_videos"] = video_paths[role]
        save_json(output / "asset_provenance.json", provenance)
        timelines = {}
        for role in ["original", "overlay"]:
            for name in sorted(names):
                print(f"Reading presentation timeline: {role}/{name}", flush=True)
                timelines[role, name] = probe_timeline(video_paths[role][name]["path"])
        state["stage"] = "align_frames"
        save_json(output / "execution_status.json", state)
        for sample in samples:
            name = sample["video"]
            chosen = choose_frames(timelines["original", name], sample["cutoff_s"], args.frames)
            overlay = match_overlay_frames(chosen, timelines["overlay", name])
            for role, selected, key in [("original", chosen, "frames_manifest"),
                                        ("overlay", overlay, "overlay_frames_manifest")]:
                path = extract_frames(video_paths[role][name]["path"], selected,
                                      output / "frames" / sample["id"] / role, sample["cutoff_s"], role)
                sample[key] = str(path.relative_to(output))
        manifest = {"data_kind": "synthetic_test_fixture" if args.synthetic_fixture else "expected_public_benchmark_assets",
                    "protocol": "causal_original_vs_publisher_overlay_convenience_pilot",
                    "samples": samples, "qa_source_hashes": qa_hashes,
                    "raw_gaze_coordinates": "not_available_not_inferred"}
        save_json(output / "qa_manifest.json", manifest)
        state["stage"] = "preprocessing" if args.dry_run else "model_inference"
        save_json(output / "execution_status.json", state)
        print("Running paired preprocessing" if args.dry_run else "Running local paired VideoLLM inference (CPU may be slow)...", flush=True)
        command = [sys.executable, str(ROOT / "scripts/run_videollm.py"),
                   "--manifest", str(output / "qa_manifest.json"), "--model", str(args.model),
                   "--modes", "full", "publisher_overlay", "--device", args.device,
                   "--output", str(output / "evaluation")]
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, check=True)
        summary = json.loads((output / "evaluation/summary.json").read_text())
        predictions = json.loads((output / "evaluation/predictions.json").read_text())
        write_report(output, manifest, summary, predictions)
        state.update(status="preprocessing_only" if args.dry_run else "model_evaluated",
                     stage="finished", accuracy_available=not args.dry_run)
        save_json(output / "execution_status.json", state)
        print(f"Finished: {output / 'REPORT.md'}")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        blocked = False
        if state["stage"] == "download":
            selection = "videos" if args.dry_run else "pilot"
            source_status = ROOT / "data/external/download_state" / (selection + "_status.json")
            if source_status.exists():
                download_status = json.loads(source_status.read_text())
                blocked = download_status["status"] == "network_blocked"
                state["download_status"] = str(source_status)
        state.update(status="blocked_network" if blocked else "incomplete",
                     error_type=type(exc).__name__, accuracy_available=False)
        save_json(output / "execution_status.json", state)
        message = ("Downloads blocked: this process cannot resolve huggingface.co; no videos or "
                   "VideoLLM weights were fetched, and no accuracy was computed.") if blocked else (
                   f"Pipeline stopped at {state['stage']} ({type(exc).__name__}). "
                   "Existing assets and partial outputs are retained; no completed result is claimed.")
        (output / "REPORT.md").write_text("# Execution status\n\n" + message + "\n")
        print(message, file=sys.stderr)
        print(f"Execution record: {output / 'execution_status.json'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
