#!/usr/bin/env python3
"""Extract causal frames with original PTS and align normalized gaze CSV samples."""
import argparse
import bisect
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gaze_memory.core import save_json
from gaze_memory.frames import usable_gaze


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--gaze", type=Path, required=True,
                   help="CSV: timestamp_s,x,y,confidence; normalized coordinates; video clock")
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--cutoff", type=float, required=True)
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--max-gaze-age", type=float, default=.1)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.count < 1 or args.start < 0 or args.cutoff <= args.start or args.max_gaze_age < 0:
        p.error("Invalid time window/count/gaze age")
    if not args.video.is_file() or not args.gaze.is_file():
        p.error("Video and gaze files must exist")
    if args.output.exists():
        p.error("Output already exists; choose a new directory")
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(args.video),
    ], capture_output=True, text=True, check=True)
    entries = json.loads(probe.stdout)["frames"]
    indexed = [(i,float(f["best_effort_timestamp_time"])) for i,f in enumerate(entries)
               if "best_effort_timestamp_time" in f]
    if any(b[1] < a[1] for a,b in zip(indexed,indexed[1:])):
        raise ValueError("Non-monotonic presentation timestamps")
    eligible = [(i,t) for i,t in indexed if args.start <= t <= args.cutoff]
    times = [t for _,t in eligible]
    if not times:
        raise ValueError("No frames in the requested window")
    chosen = {}
    for j in range(args.count):
        target = args.cutoff if args.count == 1 else args.start+(args.cutoff-args.start)*j/(args.count-1)
        k = bisect.bisect_right(times,target)-1
        if k >= 0:
            i,t = eligible[k]
            chosen[i] = t
    if not chosen:
        raise ValueError("No causal frames selected")
    gaze_rows = []
    with args.gaze.open(newline="") as f:
        for r in csv.DictReader(f):
            item = {k:float(r[k]) for k in ["timestamp_s","x","y","confidence"]}
            if usable_gaze(item):
                gaze_rows.append(item)
    gaze_rows.sort(key=lambda r:r["timestamp_s"])
    gaze_times = [r["timestamp_s"] for r in gaze_rows]
    args.output.mkdir(parents=True)
    # select preserves the known presentation frame index; -ss would risk using
    # the next frame, whose timestamp could exceed a question's cutoff.
    selector = "+".join(f"eq(n\\,{i})" for i in sorted(chosen))
    subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(args.video), "-vf", f"select={selector}",
        "-vsync", "0", str(args.output/"frame-%04d.png"),
    ], check=True)
    frames = []
    for n,(_,timestamp) in enumerate(sorted(chosen.items()),1):
        k = bisect.bisect_right(gaze_times,timestamp)-1
        gaze = gaze_rows[k] if k >= 0 and timestamp-gaze_times[k] <= args.max_gaze_age else None
        frames.append({"timestamp_s":timestamp,"path":f"frame-{n:04d}.png","gaze":gaze})
    save_json(args.output/"frames.json",{
        "video":str(args.video.resolve()),"gaze_csv":str(args.gaze.resolve()),
        "cutoff_s":args.cutoff,"frames":frames,
        "alignment":"last valid gaze sample at or before video PTS, within max_gaze_age",
    })
    print(f"Prepared {len(frames)} frames; {sum(f['gaze'] is not None for f in frames)} have valid gaze")


if __name__ == "__main__":
    main()
