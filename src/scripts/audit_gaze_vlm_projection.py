"""Training-only projection spot checks and population gaze statistics."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from gaze_transfer.data import Recording, frame_times, json_write, stable_seed

BASE = Path.home()/'Datasets/HoloAssist'
STUDY = BASE/'gaze_vlm_study'


def main():
    rows = [json.loads(x) for x in (STUDY/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    by_record = {}
    for row in sorted(rows, key=lambda r: stable_seed(r['id'])):
        by_record.setdefault(row['recording'], row)
    selected = sorted(by_record.values(), key=lambda r: stable_seed(r['recording']))[:256]
    points = []; audits = []
    fig, axes = plt.subplots(4, 4, figsize=(16, 10))
    for i, row in enumerate(selected):
        rec = Recording(BASE/'study_data/subset'/row['recording']/'Export_py')
        try:
            gaze = rec.gaze(row['start'], row['cutoff'])
            valid = gaze[:, 3] > 0
            points.extend(gaze[valid, 1:3])
            sensitivities = {}
            for depth in [.3, .5, 1.]:
                sample = rec.gaze(row['start'], row['cutoff'], depth=depth)
                sensitivities[str(depth)] = float(sample[:, 3].mean())
            audits.append({'id': row['id'], 'valid_fraction_by_assumed_depth': sensitivities})
            if i < 4:
                times = frame_times(row['start'], row['cutoff'])
                images = rec.frames(times)
                absolute = row['start']+gaze[:, 0]*5
                for j, (image, time) in enumerate(zip(images, times)):
                    axis = axes[i, j]; axis.imshow(image)
                    mask = valid & (absolute <= time) & (absolute >= time-.2)
                    axis.scatter(gaze[mask, 1]*image.width, gaze[mask, 2]*image.height,
                                 c='red', marker='+', s=100, linewidths=1.5)
                    axis.set_title(f'{row["recording"]}\nt={time:.2f}, past samples={mask.sum()}', fontsize=8)
                    axis.axis('off')
        finally:
            rec.close()
    fig.suptitle('Training-only gaze projection: assumed 0.5 m ray depth; red = preceding 200 ms')
    fig.tight_layout()
    out = ROOT/'results/gaze_vlm_study'; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/'projection_spot_check.png', dpi=140); plt.close(fig)
    if not points:
        raise ValueError('No valid projected gaze')
    json_write(STUDY/'population_gaze.json', {'mean': np.mean(points, axis=0).tolist(),
               'std': np.std(points, axis=0).tolist(), 'samples': len(points),
               'training_recordings': len(selected), 'source_split': 'train'})
    json_write(out/'projection_audit.json', audits)
    print('PROJECTION AUDIT', len(selected), 'recordings', len(points), 'valid points', flush=True)


if __name__ == '__main__':
    main()
