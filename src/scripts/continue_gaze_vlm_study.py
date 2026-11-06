"""Sequential full-data experiment queue; resumes completed stages, no test peeking.

Runs after the already-started B1 fit. It never runs two GPU model jobs together.
All scientific results remain pending until their actual evaluations finish.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = Path.home()/'Datasets/HoloAssist/gaze_vlm_study'
PYTHON = sys.executable
sys.path.insert(0, str(ROOT))
from gaze_transfer.data import json_write


def write_status(value):
    path = BASE/'queue_status.json'
    json_write(path, value)
    subprocess.run([PYTHON, str(ROOT/'scripts/build_gaze_vlm_review.py')], cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_job(name, command):
    folder = BASE/'queue_logs'; folder.mkdir(parents=True, exist_ok=True)
    write_status({'stage': 'running', 'job': name, 'command': command, 'pid': os.getpid()})
    with (folder/(name+'.log')).open('a', encoding='utf-8') as stream:
        process = subprocess.Popen([PYTHON, '-u', *command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            write_status({'stage': 'running', 'job': name, 'child_pid': process.pid, 'pid': os.getpid()})
            time.sleep(30)
        if process.returncode:
            write_status({'stage': 'failed', 'job': name, 'exit_code': process.returncode,
                          'log': str(folder/(name+'.log'))})
            raise RuntimeError('Queue stopped at '+name)


def main():
    import psutil
    first = BASE/'runs/B1_seed0'
    while True:
        status = json.loads((first/'status.json').read_text()) if (first/'status.json').exists() else {}
        if status.get('stage') == 'training_complete':
            break
        process_info = json.loads((first/'process.json').read_text())
        if not psutil.pid_exists(process_info['pid']):
            raise RuntimeError('Initial B1 fit stopped before completion; inspect its logs and checkpoint')
        write_status({'stage': 'waiting_for_active_full_fit', 'job': 'B1_seed0', 'pid': os.getpid(),
                      'train_step': status.get('step'), 'total_steps': status.get('total_steps'),
                      'remaining': 'core gaze comparisons, teacher extraction, structured transfer, development evaluation'})
        time.sleep(30)
    # Flow computation is CPU-only, started independently; wait before KL fit.
    while True:
        flow = json.loads((BASE/'flow_status.json').read_text())
        if flow.get('stage') == 'complete' and not flow.get('limited_run'):
            break
        write_status({'stage': 'waiting_for_flow_maps', 'pid': os.getpid()})
        time.sleep(30)
    commands = []
    modes = [('B1', 'label_only'), ('B2', 'attention_kl'), ('B3', 'static'),
             ('B4', 'temporal'), ('B5', 'shuffled')]
    for seed in [0, 1, 2]:
        for code, mode in modes:
            name = f'{code}_seed{seed}'; path = BASE/'runs'/name
            status = json.loads((path/'status.json').read_text()) if (path/'status.json').exists() else {}
            if status.get('stage') != 'training_complete':
                command = ['scripts/run_gaze_vlm_study.py', 'train', '--mode', mode, '--run', name, '--seed', str(seed)]
                if mode == 'attention_kl':
                    command += ['--heatmap-mode', 'flow']
                if (path/'optimizer.pt').exists():
                    command += ['--resume']
                run_job(name, command)
            commands.append((name, mode))
    # Explicit train-only teacher extraction. This queue does not fabricate missing facts.
    provenance = json.loads((BASE/'model_provenance.json').read_text())
    if not any(r['role'] == 'teacher' for r in provenance):
        raise RuntimeError('Teacher download has not completed')
    run_job('teacher_extraction', ['scripts/run_gaze_vlm_study.py', 'teacher', '--teacher-4bit', '--run', 'teacher_extraction'])
    for seed in [0, 1, 2]:
        for code, mode in [('T0', 'caption_transfer'), ('T1', 'structured_no_gaze'), ('T2', 'structured')]:
            name = f'{code}_seed{seed}'; path = BASE/'runs'/name
            status = json.loads((path/'status.json').read_text()) if (path/'status.json').exists() else {}
            if status.get('stage') != 'training_complete':
                command = ['scripts/run_gaze_vlm_study.py', 'train', '--mode', mode, '--run', name,
                           '--seed', str(seed), '--teacher-matched']
                if (path/'optimizer.pt').exists():
                    command += ['--resume']
                run_job(name, command)
            commands.append((name, mode))
    for name, mode in commands:
        run_job(name+'_dev', ['scripts/run_gaze_vlm_study.py', 'evaluate', '--mode', mode,
                 '--run', name, '--split', 'dev', '--checkpoint', str(BASE/'runs'/name/'checkpoint')])
    # Runtime state and allocation reuse T2; they are distinct inference ablations.
    for mode, frames, tokens in [('structured_state', 4, 0), ('allocated', 2, 32)]:
        run_job(mode+'_dev', ['scripts/run_gaze_vlm_study.py', 'evaluate', '--mode', mode,
                 '--run', mode+'_seed0', '--split', 'dev', '--frames', str(frames), '--keep-tokens', str(tokens),
                 '--checkpoint', str(BASE/'runs/T2_seed0/checkpoint')])
    write_status({'stage': 'implemented_development_matrix_complete',
        'locked_test_evaluated': False, 'scientific_study_complete': False,
        'pending': ['teacher factual audit', 'development configuration selection', 'locked-test evaluation',
                    'matched conventional distillation', 'verified wearer personalization',
                    'frozen task transfer', 'cross-dataset and mobile energy experiments']})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write_status({'stage': 'failed', 'error': repr(exc), 'scientific_study_complete': False})
        raise
