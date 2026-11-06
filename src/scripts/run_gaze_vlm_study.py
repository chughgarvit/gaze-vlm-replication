"""Train/evaluate actual compact VLMs and extract offline teacher concepts.

Every stage writes a resumable status. Test evaluation is explicit, never automatic.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from gaze_transfer.data import Recording, frame_times, gaze_variant, heatmaps, json_write, select_frames, stable_seed
from gaze_transfer.knowledge import TEACHER_PROMPT, SymbolicState, parse_json, validate_concepts
from gaze_transfer.model import CompactVLM, MODES, TRANSFER_MODES
from gaze_transfer.personalization import GazeCalibration
from gaze_transfer.attention import attention_structure

BASE = Path.home()/'Datasets/HoloAssist'
STUDY = BASE/'gaze_vlm_study'
TASKS = ['activity', 'anticipation', 'mistake', 'intervention', 'task_state']


def calibration(args):
    if args.personalization == 'none':
        return None
    stats = json.loads((STUDY/'population_gaze.json').read_text())
    if stats['source_split'] != 'train':
        raise ValueError('Population statistics must use train only')
    return GazeCalibration(stats['mean'], stats['std'])
PROMPTS = {
    'activity': 'Describe the dominant observed activity. Return JSON {"verb":"...","object":"..."}.',
    'anticipation': 'Predict the first new action starting in the next {horizon} seconds. Return JSON {"verb":"...","object":"..."}, or {"event":"none"}.',
    'mistake': 'Does the observed window contain an incorrectly executed action? Return JSON {"mistake":"correct"}, {"mistake":"incorrect"}, or {"mistake":"unknown"}.',
    'intervention': 'Will corrective instructor assistance start in the next {horizon} seconds? Return JSON {"intervention":true} or {"intervention":false}.',
    'task_state': 'Describe the current procedural step visible at the end. Return JSON {"task_state":"short step description"}.',
}


def manifest(split):
    with (STUDY/(split+'.jsonl')).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def question(task, horizon, structured=False):
    text = PROMPTS[task].replace('{horizon}', str(horizon))
    if structured:
        text += (' Also include a "concepts" object with caption, entities (id,name,frame,box,confidence), '
                 'facts (subject,predicate,value,frame,confidence), and relations '
                 '(before,after,predicate,confidence), grounded only in observed frames.')
    return text


def target(row, task, horizon):
    value = row['targets'][str(horizon)][task]
    if value is None or (task == 'mistake' and value == 'unknown'):
        return None
    return value if isinstance(value, dict) else {task: value}


class Inputs:
    def __init__(self):
        self.recording = self.name = None

    def load(self, row, mode, budget=4):
        if row['recording'] != self.name:
            if self.recording:
                self.recording.close()
            self.name = row['recording']
            self.recording = Recording(BASE/'study_data/subset'/self.name/'Export_py')
        times = frame_times(row['start'], row['cutoff'])
        gaze = self.recording.gaze(row['start'], row['cutoff'])
        variant = 'shuffled' if mode == 'shuffled' else 'static' if mode == 'static' else 'temporal'
        gaze = gaze_variant(gaze, variant, stable_seed(row['id']))
        chosen = select_frames(gaze, 4, budget) if mode == 'allocated' else list(range(4))
        if mode != 'allocated' and budget < 4:
            chosen = np.linspace(0, 3, budget).round().astype(int).tolist()
        times = [times[i] for i in chosen]
        images = self.recording.frames(times)
        return images, gaze, times

    def close(self):
        if self.recording:
            self.recording.close()


def concept_path(row):
    return STUDY/'teacher'/row['recording']/(str(int(row['cutoff']))+'.json')


def teacher(args):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    if args.split != 'train':
        raise ValueError('Teacher supervision is restricted to train recordings')
    rows = manifest('train')
    if args.limit:
        rows = sorted(rows, key=lambda r: stable_seed(r['id']))[:args.limit]
    rows = [r for r in rows if not concept_path(r).exists()]
    if not rows:
        print('Teacher cache complete'); return
    processor = AutoProcessor.from_pretrained(STUDY/'models/teacher', local_files_only=True)
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {'longest_edge': 512}
    kwargs = {'local_files_only': True, 'torch_dtype': torch.bfloat16 if args.device == 'cuda' else torch.float32,
              'attn_implementation': 'sdpa'}
    if args.teacher_4bit:
        if args.device != 'cuda':
            raise ValueError('The 4-bit teacher configuration requires CUDA')
        kwargs.update({'quantization_config': BitsAndBytesConfig(load_in_4bit=True,
                       bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
                       bnb_4bit_use_double_quant=True), 'device_map': {'': 0}})
    model = AutoModelForImageTextToText.from_pretrained(STUDY/'models/teacher', **kwargs)
    if not args.teacher_4bit:
        model = model.to(args.device)
    model.eval()
    loader = Inputs(); started = time.time(); accepted = 0
    try:
        for i, row in enumerate(rows):
            images, gaze, times = loader.load(row, 'label_only')
            content = [{'type': 'image'} for _ in images]+[{'type': 'text', 'text': TEACHER_PROMPT}]
            prompt = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
            encoded = processor(text=prompt, images=images, return_tensors='pt').to(args.device)
            if args.device == 'cuda':
                encoded = {k: v.to(dtype=torch.bfloat16) if v.is_floating_point() else v for k,v in encoded.items()}
            with torch.inference_mode():
                output = model.generate(**encoded, max_new_tokens=384, do_sample=False)
            text = processor.decode(output[0, encoded['input_ids'].shape[1]:], skip_special_tokens=True)
            entry = {'id': row['id'], 'cutoff': row['cutoff'], 'observed_times': times, 'raw': text,
                     'annotation_input': False, 'teacher_4bit': args.teacher_4bit, 'seconds': time.time()-started}
            try:
                entry['concepts'] = validate_concepts(parse_json(text), len(times)); entry['accepted'] = True; accepted += 1
                entry['concepts']['attention'] = attention_structure(gaze, times, row['start'], row['cutoff'],
                                                                   entry['concepts']['entities'])
                entry['has_structured_facts'] = bool(entry['concepts']['entities'] and entry['concepts']['facts'])
            except (ValueError, KeyError, TypeError) as exc:
                entry['accepted'] = False; entry['validation_error'] = str(exc)
            json_write(concept_path(row), entry)
            json_write(STUDY/'teacher_status.json', {'stage': 'running', 'done': i+1, 'total_remaining': len(rows),
                       'accepted_this_run': accepted, 'elapsed_s': time.time()-started})
            print('TEACHER', i+1, '/', len(rows), entry['accepted'], flush=True)
    finally:
        loader.close()
    json_write(STUDY/'teacher_status.json', {'stage': 'complete', 'processed_this_run': len(rows),
               'accepted_this_run': accepted, 'elapsed_s': time.time()-started, 'limited_run': bool(args.limit)})


def teacher_answer(row, mode, answer):
    if mode not in TRANSFER_MODES:
        return answer, 1.
    path = concept_path(row)
    if not path.exists():
        return None, 0.
    entry = json.loads(path.read_text(encoding='utf-8'))
    if not entry['accepted']:
        return None, 0.
    concepts = entry['concepts']
    confidences = [f['confidence'] for f in concepts['entities']+concepts['facts']+concepts['relations']]
    weight = float(np.mean(confidences)) if confidences else .5
    if mode == 'caption_transfer':
        return {**answer, 'caption': concepts['caption']}, weight
    # Explicit bounded supervision, preserving referential integrity on small VRAM.
    entities = sorted(concepts['entities'], key=lambda e: -e['confidence'])[:4]
    known = {e['id'] for e in entities}
    concepts = {**concepts, 'entities': entities,
                'facts': [f for f in concepts['facts'] if f['subject'] in known][:4],
                'relations': concepts['relations'][:4]}
    if 'attention' in concepts:
        concepts['attention'] = {**concepts['attention'],
              'low_dispersion_episodes': concepts['attention']['low_dispersion_episodes'][:4],
              'entity_visits': [v for v in concepts['attention']['entity_visits'] if v['entity'] in known][:4]}
    return {**answer, 'concepts': concepts}, weight


def train(args):
    if args.mode == 'pretrained':
        raise ValueError('Use evaluate for the unadapted pretrained baseline')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rows = manifest('train')
    if args.limit:
        rows = sorted(rows, key=lambda r: stable_seed(r['id']))[:args.limit]
    # Same teacher-available examples for transfer comparisons; no pseudo facts.
    if args.mode in TRANSFER_MODES or args.teacher_matched:
        rows = [r for r in rows if concept_path(r).exists() and
                json.loads(concept_path(r).read_text(encoding='utf-8'))['accepted']]
        if not rows:
            raise ValueError('No validated teacher concepts: run teacher stage first')
    samples = [(r, task) for r in rows for task in args.tasks if target(r, task, args.horizon) is not None]
    if not samples:
        raise ValueError('No labeled train samples')
    model = CompactVLM(STUDY/'models/student', args.device)
    model.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.vlm.enable_input_require_grads()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
    output = STUDY/'runs'/args.run
    if (output/'config.json').exists() and not args.resume:
        raise ValueError('Existing run: use --resume or choose a new --run')
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({'train_windows': len(rows), 'train_examples': len(samples),
                   'parameters': sum(p.numel() for p in model.parameters()),
                   'trainable_parameters': sum(p.numel() for p in parameters),
                   'spec_sha256': hashlib.sha256((ROOT/'GAZE_VLM_STUDY.md').read_bytes()).hexdigest(),
                   'protocol_sha256': hashlib.sha256((STUDY/'protocol.json').read_bytes()).hexdigest()})
    if args.resume:
        old = json.loads((output/'config.json').read_text())
        for key in ['mode', 'seed', 'horizon', 'frames', 'keep_tokens', 'lr', 'accumulation',
                    'tasks', 'personalization', 'heatmap_mode', 'teacher_matched', 'protocol_sha256']:
            if old.get(key) != config.get(key):
                raise ValueError('Resume configuration changed: '+key)
    else:
        config['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in [*sorted((ROOT/'gaze_transfer').glob('*.py')), Path(__file__).resolve()]}
        json_write(output/'config.json', config)
    step = 0
    if args.resume:
        model.load(output/'checkpoint')
        saved = torch.load(output/'optimizer.pt', map_location=args.device, weights_only=False)
        optimizer.load_state_dict(saved['optimizer']); step = saved['step']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    loader = Inputs(); population = calibration(args)
    started = time.time(); model.train(); optimizer.zero_grad(set_to_none=True)
    losses = []; trained = 0
    try:
        for epoch in range(args.epochs):
            order = np.random.default_rng(args.seed+epoch).permutation(len(samples))
            for ordinal, index in enumerate(order):
                global_step = epoch*len(samples)+ordinal
                if global_step < step:
                    continue
                row, task = samples[index]
                images, gaze, times = loader.load(row, args.mode, args.frames)
                answer, weight = teacher_answer(row, args.mode, target(row, task, args.horizon))
                prompt = question(task, args.horizon, 'structured' in args.mode or args.mode == 'allocated')
                if args.mode == 'caption_transfer':
                    prompt += ' Also include a caption of the observed actions.'
                maps, valid = heatmaps(gaze, times, 8, row['start'], row['cutoff'])
                model_gaze = population.transform(gaze) if population else gaze
                model.set_context(torch.from_numpy(model_gaze), args.mode, maps, args.keep_tokens)
                encoded = model.inputs(images, prompt, json.dumps(answer, ensure_ascii=False))
                result = model(encoded)
                kl = result.loss*0
                if args.mode == 'attention_kl':
                    grid = int(round(model.adapter.attention.shape[-1]**.5))
                    if args.heatmap_mode == 'flow':
                        path = STUDY/'flow_heatmaps'/row['recording']/(str(int(row['cutoff']))+'.npz')
                        with np.load(path) as cache:
                            if str(cache['id']) != row['id']:
                                raise ValueError('Flow cache ID mismatch')
                            maps, valid = cache['maps'], cache['valid']
                        if args.frames != 4:
                            raise ValueError('Cached flow supervision currently requires four frames')
                    else:
                        maps, valid = heatmaps(gaze, times, grid, row['start'], row['cutoff'])
                    kl = model.adapter.kl(torch.from_numpy(maps), torch.from_numpy(valid))
                loss = weight*result.loss + args.kl_weight*kl
                (loss/args.accumulation).backward()
                trained += 1; step = global_step+1
                stopping = bool(args.max_steps and trained >= args.max_steps)
                if trained % args.accumulation == 0 or ordinal == len(order)-1 or stopping:
                    remainder = trained % args.accumulation
                    if remainder:
                        for parameter in parameters:
                            if parameter.grad is not None:
                                parameter.grad.mul_(args.accumulation/remainder)
                    torch.nn.utils.clip_grad_norm_(parameters, 1.)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                losses.append({'step': step, 'id': row['id'], 'task': task, 'loss': float(loss.detach()), 'kl': float(kl.detach())})
                if step % 10 == 0 or trained == 1:
                    status = {'stage': 'training', 'mode': args.mode, 'step': step,
                              'total_steps': len(samples)*args.epochs, 'loss': losses[-1]['loss'],
                              'elapsed_s': time.time()-started, 'peak_vram_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None}
                    json_write(output/'status.json', status)
                    print('TRAIN', args.run, step, round(status['loss'], 4), flush=True)
                if len(losses) >= 10:
                    with (output/'losses.jsonl').open('a', encoding='utf-8') as stream:
                        for entry in losses:
                            stream.write(json.dumps(entry)+'\n')
                    losses.clear()
                if step % args.checkpoint_every == 0 or (args.max_steps and trained >= args.max_steps):
                    model.save(output/'checkpoint')
                    torch.save({'optimizer': optimizer.state_dict(), 'step': step,
                                'torch_rng': torch.get_rng_state(),
                                'cuda_rng': torch.cuda.get_rng_state_all() if args.device == 'cuda' else []}, output/'optimizer.pt')
                if args.max_steps and trained >= args.max_steps:
                    break
            if args.max_steps and trained >= args.max_steps:
                break
    finally:
        loader.close()
    model.save(output/'checkpoint')
    torch.save({'optimizer': optimizer.state_dict(), 'step': step,
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if args.device == 'cuda' else []}, output/'optimizer.pt')
    with (output/'losses.jsonl').open('a', encoding='utf-8') as stream:
        for loss in losses:
            stream.write(json.dumps(loss)+'\n')
    json_write(output/'status.json', {'stage': 'engineering_run_complete' if args.max_steps or args.limit else 'training_complete',
               'step': step, 'trained_this_run': trained, 'elapsed_s': time.time()-started,
               'scientific_evaluation_complete': False})


def evaluate(args):
    rows = manifest(args.split)
    if args.limit:
        rows = sorted(rows, key=lambda r: stable_seed(r['id']))[:args.limit]
    rows.sort(key=lambda r: (r['recording'], r['cutoff']))
    model = CompactVLM(STUDY/'models/student', args.device)
    if args.checkpoint:
        model.load(Path(args.checkpoint))
    model.eval()
    output = STUDY/'runs'/args.run
    output.mkdir(parents=True, exist_ok=True)
    file = output/(args.split+'_predictions.jsonl')
    done = set()
    if file.exists():
        for line in file.read_text(encoding='utf-8').splitlines():
            saved = json.loads(line); done.add((saved['id'], saved['task']))
    loader = Inputs(); ledger = SymbolicState(); previous = None; started = time.time()
    try:
        with file.open('a', encoding='utf-8') as stream:
            for row in rows:
                if row['recording'] != previous:
                    ledger = SymbolicState(); previous = row['recording']
                    population = calibration(args)
                state_before = dict(ledger.predicates)
                for task in TASKS:
                    expected = target(row, task, args.horizon)
                    if expected is None:
                        continue
                    if (row['id'], task) in done:
                        if args.mode in ['structured_state', 'allocated']:
                            raise ValueError('Stateful evaluation resume requires replay; choose a new run')
                        continue
                    begin = time.perf_counter()
                    images, gaze, times = loader.load(row, args.mode, args.frames)
                    prompt = question(task, args.horizon, 'structured' in args.mode or args.mode == 'allocated')
                    if args.mode in ['structured_state', 'allocated']:
                        prompt += '\nPrevious uncertain category-level state: '+json.dumps(state_before)
                    maps, _ = heatmaps(gaze, times, 8, row['start'], row['cutoff'])
                    model_gaze = population.transform(gaze) if population else gaze
                    model.set_context(torch.from_numpy(model_gaze), args.mode, maps, args.keep_tokens)
                    encoded = model.inputs(images, prompt)
                    if args.device == 'cuda':
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    forward_start = time.perf_counter()
                    text = model.generate(encoded, args.max_new_tokens)
                    candidate_probability = None
                    if task in ['mistake', 'intervention']:
                        answers = ([{'mistake': 'correct'}, {'mistake': 'incorrect'}] if task == 'mistake'
                                   else [{'intervention': False}, {'intervention': True}])
                        probabilities = model.score_answers(images, prompt, [json.dumps(a) for a in answers])
                        candidate_probability = probabilities[1]
                    if args.device == 'cuda':
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter()-forward_start
                    predicted = None; state = None
                    try:
                        predicted = parse_json(text)
                        if task == 'activity' and args.mode in ['structured_state', 'allocated']:
                            concepts = validate_concepts(predicted['concepts'], len(times))
                            state = ledger.update(concepts, row['cutoff'])
                    except (ValueError, TypeError, KeyError):
                        pass
                    result = {'id': row['id'], 'group': row['group'], 'recording': row['recording'], 'task': task,
                              'horizon': args.horizon, 'mode': args.mode, 'raw': text, 'prediction': predicted,
                              'target': expected, 'frames': len(images), 'times': times, 'generation_s': elapsed,
                              'total_s': time.perf_counter()-begin, 'input_tokens': encoded['input_ids'].numel(),
                              'image_tokens': int((encoded['input_ids']==model.raw.config.image_token_id).sum()),
                              'output_tokens': len(model.processor.tokenizer.encode(text, add_special_tokens=False)),
                              'peak_vram_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                              'energy_joules': None, 'state': state,
                              'positive_probability': candidate_probability,
                              'probability_definition': 'normalized complete-answer likelihood over two fixed candidates',
                              'corrective_speech_in_observation': row['targets'][str(args.horizon)]['corrective_speech_in_observation']}
                    stream.write(json.dumps(result, ensure_ascii=False)+'\n'); stream.flush()
                    done.add((row['id'], task))
                    json_write(output/'evaluation_status.json', {'stage': 'running', 'split': args.split,
                               'predictions': len(done), 'elapsed_s': time.time()-started})
                    print('EVAL', row['id'], task, round(elapsed, 2), flush=True)
                if population and args.personalization == 'online':
                    population.observe(gaze)
    finally:
        loader.close()
    json_write(output/'evaluation_status.json', {'stage': 'complete', 'split': args.split,
               'predictions': len(done), 'limited_run': bool(args.limit), 'elapsed_s': time.time()-started})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['teacher', 'train', 'evaluate'])
    p.add_argument('--mode', choices=MODES, default='label_only')
    p.add_argument('--split', choices=['train', 'dev', 'test'], default='train')
    p.add_argument('--run', default='label_only_seed0')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--horizon', type=int, choices=[2, 5], default=2)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--kl-weight', type=float, default=1.)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--frames', type=int, choices=[1, 2, 4], default=4)
    p.add_argument('--keep-tokens', type=int, choices=[0, 16, 32, 64], default=0,
                   help='Post-vision language-input tokens per frame; vision cost is retained')
    p.add_argument('--checkpoint-every', type=int, default=1000)
    p.add_argument('--checkpoint')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--teacher-matched', action='store_true', help='Restrict controls to identical validated-teacher windows')
    p.add_argument('--teacher-4bit', action='store_true', help='Quantized offline teacher; recorded in provenance')
    p.add_argument('--personalization', choices=['none', 'population', 'online'], default='none')
    p.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    p.add_argument('--heatmap-mode', choices=['unwarped', 'flow'], default='unwarped')
    args = p.parse_args()
    torch.set_num_threads(8)
    try:
        {'teacher': teacher, 'train': train, 'evaluate': evaluate}[args.stage](args)
    except Exception as exc:
        json_write(STUDY/'runs'/args.run/'failure.json', {'stage': args.stage, 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
