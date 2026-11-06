#!/usr/bin/env python3
"""Export scene/gaze features from independently labeled, clock-audited windows.

Input JSON follows configs/behavior_windows.example.json. Gaze CSV is already
aligned to the VIDEO clock: timestamp_s,x,y,valid; x/y normalized, valid=0/1.
This is not an automatic adapter for uninspected original BeGaze files.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from gaze_behavior.clip import encode_images, load_vision
from gaze_behavior.data import load_manifest
from gaze_memory.benchmark import choose_frames, extract_frames, probe_timeline


def read_gaze(path):
    times, xy, valid = [], [], []
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            t, x, y = float(row["timestamp_s"]), float(row["x"]), float(row["y"])
            if row["valid"] not in {"0", "1"}:
                raise ValueError("Gaze valid field must explicitly use 0 or 1")
            tracked = row["valid"] == "1"
            if not np.isfinite(t) or t < 0 or (tracked and (not np.isfinite([x, y]).all() or not 0 <= x <= 1 or not 0 <= y <= 1)):
                raise ValueError("Invalid timestamp/tracked gaze coordinate")
            times.append(t)
            xy.append([x, y] if tracked else [0, 0])
            valid.append(tracked)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Gaze timestamps must increase; do not silently aggregate duplicates")
    return np.asarray(times), np.asarray(xy, np.float32).reshape(-1, 2), np.asarray(valid, bool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local OpenCLIP ViT-B/32 safetensors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--max-total-frames", type=int, default=200000)
    parser.add_argument("--synthetic-fixture", action="store_true")
    args = parser.parse_args()
    if args.frames < 2 or args.max_total_frames < args.frames:
        parser.error("Use at least two observed frames")
    manifest = json.loads(args.windows.read_text())
    if (manifest.get("synthetic") is True) != args.synthetic_fixture:
        parser.error("Synthetic flag must match input metadata")
    if not all(manifest.get("audit", {}).get(k) is True for k in
               ("alignment_verified", "subject_identity_verified", "session_order_verified", "labels_independent_of_gaze")):
        parser.error("Original gaze/identity/chronology must be audited before feature extraction")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    model = load_vision(args.checkpoint)
    timelines, gaze_cache, hashes, groups = {}, {}, {}, {}
    exported = [None] * len(manifest["samples"])
    # One decode and one embedding per selected source frame, even if windows
    # overlap. A frozen external encoder is not fitted on these pooled windows.
    for n, sample in enumerate(manifest["samples"]):
        video = (args.windows.parent / sample["video"]).resolve()
        gaze_path = (args.windows.parent / sample["gaze_csv"]).resolve()
        for path in (video, gaze_path):
            if str(path) not in hashes:
                with path.open("rb") as stream:
                    hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
        if video not in timelines:
            timelines[video] = probe_timeline(video)
        if gaze_path not in gaze_cache:
            gaze_cache[gaze_path] = read_gaze(gaze_path)
        start, cutoff = sample["start_s"], sample["cutoff_s"]
        shifted = [(i, t - start) for i, t in timelines[video] if t >= start]
        selected = choose_frames(shifted, cutoff - start, args.frames)
        selected = [(i, t + start) for i, t in selected]
        groups.setdefault(video, []).append((n, sample, gaze_path, selected))
    frame_count = sum(len({i for _, _, _, selected in windows for i, _ in selected}) for windows in groups.values())
    if frame_count > args.max_total_frames:
        raise ValueError("Selected source frames exceed --max-total-frames")
    if shutil.disk_usage(args.output).free < frame_count * 200000 + 1_000_000_000:
        raise ValueError("Insufficient disk headroom for bounded 224px frame intermediates")
    for v, (video, windows) in enumerate(groups.items()):
        all_selected = sorted({i: t for _, _, _, selected in windows for i, t in selected}.items())
        folder = args.output / f"video-{v:04d}-frames"
        frame_manifest = extract_frames(video, all_selected, folder,
                                        max(s["cutoff_s"] for _, s, _, _ in windows),
                                        "audited_original_video", max_side=224)
        frames = json.loads(frame_manifest.read_text())["frames"]
        vectors = []
        for b in range(0, len(frames), 8):
            images = []
            for frame in frames[b:b + 8]:
                with Image.open(folder / frame["path"]) as image:
                    images.append(image.convert("RGB"))
            vectors.append(encode_images(model, images))
            if b % 80 == 0:
                print(f"Scene features: recording {v + 1}/{len(groups)}, frame batch {b // 8 + 1}", flush=True)
        visual = np.concatenate(vectors)
        index = {i: j for j, (i, _) in enumerate(all_selected)}
        for n, sample, gaze_path, selected in windows:
            times, xy, valid = gaze_cache[gaze_path]
            mask = (times >= sample["start_s"]) & (times <= sample["cutoff_s"])
            name = f"sample-{n:05d}.npz"
            np.savez_compressed(args.output / name, visual=visual[[index[i] for i, _ in selected]],
                                frame_time=np.array([t for _, t in selected]), gaze_time=times[mask],
                                gaze_xy=xy[mask], gaze_valid=valid[mask])
            exported[n] = {k: val for k, val in sample.items() if k not in {"video", "gaze_csv"}} | {"features": name}
        print(f"Exported recording {v + 1}/{len(groups)}: {len(windows)} windows, {len(frames)} distinct frames", flush=True)
    with args.checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest["samples"] = exported
    manifest["provenance"] = {"window_provenance": manifest.get("provenance"), "source_sha256": hashes,
                              "checkpoint_path": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
                              "feature_encoder": "Frozen CLIP ViT-B/32, normalized 512-D image embedding",
                              "preprocessing": "ffmpeg bicubic max-side 224; full-scene 224x224 letterbox, mean-color padding; nonstandard CLIP crop",
                              "clock": "Original video PTS; gaze clock alignment independently verified upstream",
                              "gaze_features": "Raw observed coordinates/validity, no target text or gaze-derived labels",
                              "frames_per_window_max": args.frames, "distinct_source_frames": frame_count,
                              "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in (Path(__file__), ROOT / "gaze_behavior/clip.py", ROOT / "gaze_memory/benchmark.py")}}
    target = args.output / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    load_manifest(target, args.synthetic_fixture)
    print(f"Validated features: {target}")


if __name__ == "__main__":
    raise SystemExit(main())
