#!/usr/bin/env python3
"""Synthetic preprocessing check, explicitly not a research result."""
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image,ImageDraw
from gaze_memory.core import save_json


def main():
    folder=ROOT/"results/preprocessing_smoke"
    if folder.exists():
        raise SystemExit("Smoke-test output exists; use a fresh workspace or inspect existing outputs")
    sources=folder/"source"
    sources.mkdir(parents=True)
    for i in range(12):
        im=Image.new("RGB",(640,480),"#d7dadd")
        draw=ImageDraw.Draw(im)
        draw.rectangle((40,240,600,420),fill="#877c72")
        draw.rectangle((260+i*5,260,300+i*5,310),fill="#285d97")
        draw.ellipse((70,290,120,335),fill="#d97738")
        draw.text((20,20),f"SYNTHETIC CHECK - frame {i}",fill="black")
        im.save(sources/f"{i:03d}.png")
    subprocess.run(["ffmpeg","-v","error","-framerate","4","-i",str(sources/"%03d.png"),
                    "-c:v","libx264","-pix_fmt","yuv420p",str(sources/"test.mp4")],check=True)
    with (sources/"gaze.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["timestamp_s","x","y","confidence"])
        for i in range(12):
            writer.writerow([i/4,(280+i*5)/640,285/480,1])
    subprocess.run([sys.executable,str(ROOT/"scripts/prepare_frames.py"),"--video",str(sources/"test.mp4"),
                    "--gaze",str(sources/"gaze.csv"),"--cutoff","1.85","--count","6",
                    "--output",str(folder/"aligned")],check=True)
    save_json(folder/"manifest.json",{
        "status":"synthetic_preprocessing_only",
        "samples":[{"id":"synthetic-smoke","frames_manifest":"aligned/frames.json",
                    "cutoff_s":1.85,"frame_count":6,"question":"What color is the moving square?",
                    "options":["A. Blue","B. Orange","C. Green","D. Black"],"answer":"A"}]})
    subprocess.run([sys.executable,str(ROOT/"scripts/run_videollm.py"),"--manifest",str(folder/"manifest.json"),
                    "--dry-run","--output",str(folder/"dry_run")],check=True)
    rows=json.loads((folder/"dry_run/predictions.json").read_text())
    assert all(r["max_frame_time"]<=1.85 for r in rows)
    assert {r["mode"] for r in rows}=={"full","center_crop","gaze_overlay","gaze_crop","gaze_context","shuffled_gaze"}
    assert all("correct" not in r for r in rows)
    print("Passed real ffmpeg extraction, causal alignment, and all six frame treatments; no model inference.")


if __name__=="__main__":
    main()
