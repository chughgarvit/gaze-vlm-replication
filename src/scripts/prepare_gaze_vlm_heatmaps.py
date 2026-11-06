"""Offline bidirectional-flow gaze aggregation for the compact Gaze-VLM adaptation."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from gaze_transfer.data import Recording, frame_times, heatmaps, json_write

BASE = Path.home()/'Datasets/HoloAssist'
STUDY = BASE/'gaze_vlm_study'


def flow_maps(rec, row, grid=8):
    start, cutoff = row['start'], row['cutoff']
    gaze = rec.gaze(start, cutoff)
    absolute = start+gaze[:, 0]*(cutoff-start)
    times = frame_times(start, cutoff)
    maps, valid = [], []
    cache = {}
    def gray(t):
        index = int(t*rec.fps)
        if index not in cache:
            image = np.asarray(rec.frames([t])[0])
            cache[index] = cv2.cvtColor(cv2.resize(image, (256, 144)), cv2.COLOR_RGB2GRAY)
        return cache[index]
    for t in times:
        current = gray(t); warped = []
        selected = np.flatnonzero((gaze[:, 3] > 0) & (absolute <= t) & (absolute >= t-.2))
        pairs = {}
        for index in selected:
            source_t = float(absolute[index]); previous = gray(source_t)
            pair_key = (int(source_t*rec.fps), int(t*rec.fps))
            if pair_key not in pairs:
                forward = cv2.calcOpticalFlowFarneback(previous, current, None, .5, 3, 15, 3, 5, 1.2, 0)
                backward = cv2.calcOpticalFlowFarneback(current, previous, None, .5, 3, 15, 3, 5, 1.2, 0)
                pairs[pair_key] = forward, backward
            forward, backward = pairs[pair_key]
            x, y = gaze[index, 1:3]*[255, 143]
            dx, dy = forward[int(round(y)), int(round(x))]
            wx, wy = float(x+dx), float(y+dy)
            if not (0 <= wx < 255 and 0 <= wy < 143):
                continue
            reverse = backward[int(round(wy)), int(round(wx))]
            if np.linalg.norm(np.array([dx, dy])+reverse) > 1.5:
                continue
            warped.append([0., wx/255, wy/143, 1., 0.])
        if warped:
            # Reuse normalized Gaussian rasterization after motion transport.
            m, mask = heatmaps(np.asarray(warped), [0.], grid, 0., 1.)
            maps.append(m[0]); valid.append(True)
        else:
            maps.append(np.full(grid*grid, 1/(grid*grid))); valid.append(False)
    return np.asarray(maps, np.float32), np.asarray(valid)


def main():
    cv2.setNumThreads(2)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['train', 'dev'], default='train')
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    rows = [json.loads(x) for x in (STUDY/(args.split+'.jsonl')).read_text().splitlines()]
    if args.limit:
        rows = rows[:args.limit]
    rec = None; previous = None; started = time.time(); covered = 0
    try:
        for i, row in enumerate(rows):
            path = STUDY/'flow_heatmaps'/row['recording']/(str(int(row['cutoff']))+'.npz')
            if path.exists():
                continue
            if row['recording'] != previous:
                if rec:
                    rec.close()
                previous = row['recording']
                rec = Recording(BASE/'study_data/subset'/previous/'Export_py')
            maps, valid = flow_maps(rec, row)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, maps=maps, valid=valid, id=row['id'],
                                method='Farneback bidirectional, consistency <=1.5px at 256x144')
            covered += int(valid.sum())
            if (i+1) % 50 == 0 or i == len(rows)-1:
                json_write(STUDY/'flow_status.json', {'stage': 'running', 'done': i+1,
                           'total': len(rows), 'elapsed_s': time.time()-started})
                print('FLOW', i+1, '/', len(rows), flush=True)
    finally:
        if rec:
            rec.close()
    json_write(STUDY/'flow_status.json', {'stage': 'complete', 'split': args.split,
               'limited_run': bool(args.limit), 'rows': len(rows), 'elapsed_s': time.time()-started,
               'new_valid_frame_maps': covered})


if __name__ == '__main__':
    main()
