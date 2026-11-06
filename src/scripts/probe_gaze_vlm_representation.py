"""Extract frozen actual VLM states and fit downstream heads on disjoint splits.

Use a representation checkpoint that omitted the probed target during training
before calling this held-out-task transfer. Otherwise this is a frozen readout.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from gaze_transfer.data import heatmaps, json_write
from gaze_transfer.model import CompactVLM, MODES
from scripts.run_gaze_vlm_study import Inputs, STUDY, manifest, target


def label(row, task, horizon):
    value = target(row, task, horizon)
    if value is None:
        return None
    if task in ['activity', 'anticipation']:
        return 'none' if value.get('event') == 'none' else value.get('verb', '')+'::'+value.get('object', '')
    return str(value[task])


def extract(args):
    rows = manifest(args.split)
    if args.limit:
        rows = rows[:args.limit]
    model = CompactVLM(STUDY/'models/student', args.device)
    model.load(args.checkpoint); model.eval(); model.requires_grad_(False)
    loader = Inputs(); features = []
    try:
        for i, row in enumerate(rows):
            images, gaze, times = loader.load(row, args.mode)
            maps, _ = heatmaps(gaze, times, 8, row['start'], row['cutoff'])
            model.set_context(torch.from_numpy(gaze), args.mode, maps)
            encoded = model.inputs(images, 'Summarize the visible objects and interactions in these past frames.')
            with torch.inference_mode():
                output = model.vlm(**encoded, use_cache=False, output_hidden_states=True)
            features.append(output.hidden_states[-1][0, -1].float().cpu().numpy())
            if (i+1) % 50 == 0:
                print('EMBED', i+1, '/', len(rows), flush=True)
    finally:
        loader.close()
    folder = STUDY/'representations'/args.run; folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/(args.split+'.npz'), features=np.asarray(features), ids=[r['id'] for r in rows])
    json_write(folder/(args.split+'_provenance.json'), {'checkpoint': args.checkpoint, 'mode': args.mode,
               'split': args.split, 'examples': len(rows), 'label_input': False,
               'prompt': 'Summarize the visible objects and interactions in these past frames.'})


def probe(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    folder = STUDY/'representations'/args.run
    tables = {}; features = {}
    for split in ['train', 'dev', 'test']:
        source = {r['id']: r for r in manifest(split)}
        with np.load(folder/(split+'.npz')) as saved:
            features[split] = saved['features']; tables[split] = [source[str(i)] for i in saved['ids']]
    run_config = Path(args.checkpoint).parent/'config.json'
    training_tasks = json.loads(run_config.read_text()).get('tasks', []) if run_config.exists() else None
    metrics = []; predictions = []
    for task in ['activity', 'anticipation', 'mistake', 'intervention']:
        labels = {s: np.asarray([label(r, task, args.horizon) for r in rows], dtype=object)
                  for s, rows in tables.items()}
        valid = {s: np.asarray([v is not None for v in y]) for s, y in labels.items()}
        classes = sorted(set(labels['train'][valid['train']]))
        if len(classes) < 2:
            continue
        encoder = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight='balanced', max_iter=2000))
        encoder.fit(features['train'][valid['train']], labels['train'][valid['train']].astype(str))
        for split in ['dev', 'test']:
            mask = valid[split]; y = labels[split][mask].astype(str)
            probabilities = encoder.predict_proba(features[split][mask]); predicted = encoder.classes_[probabilities.argmax(1)]
            covered = np.isin(y, classes)
            score = f1_score(y, predicted, average='macro', zero_division=0)
            item = {'task': task, 'split': split, 'macro_f1_including_unseen_labels': float(score),
                    'examples': len(y), 'train_class_coverage': float(covered.mean()),
                    'held_out_task_during_encoder_training': task not in training_tasks if training_tasks is not None else None}
            if task in ['mistake', 'intervention']:
                positive = 'incorrect' if task == 'mistake' else 'True'
                pos = list(encoder.classes_).index(positive)
                item['average_precision'] = float(average_precision_score(y == positive, probabilities[:, pos]))
            metrics.append(item)
            for row, truth, guess, probability in zip(np.asarray(tables[split], dtype=object)[mask], y, predicted, probabilities):
                predictions.append({'id': row['id'], 'group': row['group'], 'split': split, 'task': task,
                                    'target': truth, 'prediction': guess, 'probabilities': probability.tolist(),
                                    'classes': encoder.classes_.tolist()})
    json_write(folder/'probe_metrics.json', metrics)
    with (folder/'probe_predictions.jsonl').open('w', encoding='utf-8') as stream:
        for row in predictions:
            stream.write(json.dumps(row)+'\n')
    print(json.dumps(metrics, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['extract', 'probe'])
    p.add_argument('--run', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--mode', choices=MODES, default='structured')
    p.add_argument('--split', choices=['train', 'dev', 'test'], default='train')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--horizon', choices=[2, 5], type=int, default=2)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args(); torch.set_num_threads(8)
    {'extract': extract, 'probe': probe}[args.stage](args)


if __name__ == '__main__':
    main()
