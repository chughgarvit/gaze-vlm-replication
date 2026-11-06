"""Resumable verified transfer; detached server training starts only after completion."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import tarfile
import threading
import time

import paramiko

ROOT = Path(__file__).resolve().parents[1]
BASE = Path.home() / 'Datasets/HoloAssist'
STATE = BASE / 'server_transfer'
REMOTE = '/mnt/hdd/Garvit'
LOCK = threading.Lock()


def connect():
    client = paramiko.SSHClient()
    client.load_host_keys(str(STATE / 'known_hosts'))
    client.connect('10.4.0.73', username='garvit1',
                   key_filename=str(STATE / 'server_key'), look_for_keys=False,
                   allow_agent=False, timeout=30)
    client.get_transport().set_keepalive(30)
    return client


def command(client, text):
    _, out, err = client.exec_command(text)
    result = out.read().decode()
    error = err.read().decode()
    if out.channel.recv_exit_status():
        raise RuntimeError(error + result)
    return result


def upload(item):
    local, remote = item
    digest = hashlib.file_digest(local.open('rb'), 'sha256').hexdigest()
    for attempt in range(12):
        client = None
        try:
            client = connect()
            sftp = client.open_sftp()
            command(client, 'mkdir -p ' + shlex.quote(str(Path(remote).parent).replace('\\', '/')))
            receipt = remote + '.sha256'
            try:
                with sftp.open(receipt) as stream:
                    done = stream.read().decode().strip() == digest
                if done and sftp.stat(remote).st_size == local.stat().st_size:
                    return local.stat().st_size
            except IOError:
                pass
            part = remote + '.part'
            try:
                offset = sftp.stat(part).st_size
            except IOError:
                offset = 0
            if offset > local.stat().st_size:
                sftp.remove(part)
                offset = 0
            with local.open('rb') as source, sftp.open(part, 'ab' if offset else 'wb') as dest:
                dest.set_pipelined(True)
                source.seek(offset)
                while chunk := source.read(1024 * 1024):
                    dest.write(chunk)
            actual = command(client, 'sha256sum ' + shlex.quote(part)).split()[0]
            if actual != digest:
                sftp.remove(part)
                raise RuntimeError('SHA256 mismatch: ' + remote)
            sftp.posix_rename(part, remote)
            with sftp.open(receipt, 'w') as stream:
                stream.write(digest + '\n')
            return local.stat().st_size
        except Exception:
            if attempt == 11:
                raise
            time.sleep(min(60, 5 * (attempt + 1)))
        finally:
            if client:
                client.close()


def pack(path, entries):
    with tarfile.open(path, 'w') as archive:
        for local, name in entries:
            archive.add(local, arcname=name, recursive=False)


def main():
    STATE.mkdir(exist_ok=True)
    code = []
    for folder in ['gaze_transfer', 'scripts', 'tests']:
        code.extend((p, p.relative_to(ROOT).as_posix()) for p in (ROOT / folder).glob('*.py'))
    code += [(ROOT / p, p) for p in ['GAZE_VLM_STUDY.md', 'requirements-gaze-vlm-study.lock.txt']]
    pack(STATE / 'code.tar', code)
    upload((STATE / 'code.tar', REMOTE + '/code.tar'))
    client = connect()
    command(client, 'mkdir -p ' + REMOTE + '/gaze-study; tar -xf ' + REMOTE + '/code.tar -C ' + REMOTE + '/gaze-study')
    setup = '''#!/bin/bash
set -euo pipefail
export UV_CACHE_DIR=/mnt/hdd/Garvit/cache/uv
export UV_PYTHON_INSTALL_DIR=/mnt/hdd/Garvit/python
export TMPDIR=/mnt/hdd/Garvit/tmp
export HF_HOME=/mnt/hdd/Garvit/cache/huggingface
export UV_INSTALL_DIR=/mnt/hdd/Garvit/bin
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$UV_INSTALL_DIR"
cd /mnt/hdd/Garvit/gaze-study
exec >> /mnt/hdd/Garvit/server_pipeline.log 2>&1
trap 'echo "FAILED at line $LINENO"; touch /mnt/hdd/Garvit/server_failed' ERR
date
if [ ! -x "$UV_INSTALL_DIR/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh -o /mnt/hdd/Garvit/install_uv.sh
  sh /mnt/hdd/Garvit/install_uv.sh
fi
UV="$UV_INSTALL_DIR/uv"
"$UV" venv --python 3.12 /mnt/hdd/Garvit/venv
PY=/mnt/hdd/Garvit/venv/bin/python
"$UV" pip install --python "$PY" torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python "$PY" -r requirements-gaze-vlm-study.lock.txt
"$UV" pip check --python "$PY"
"$PY" -c 'import torch; print(torch.__version__, torch.cuda.get_device_name()); assert torch.cuda.is_available()'
touch /mnt/hdd/Garvit/environment_ready
echo 'Environment ready; waiting for verified data and checkpoint transfer'
while [ ! -f /mnt/hdd/Garvit/transfer_complete ]; do sleep 30; done
"$PY" -m unittest discover -s tests -p test_gaze_transfer.py
echo 'Resuming B1 checkpoint on server'
"$PY" -u scripts/run_gaze_vlm_study.py train --mode label_only --run B1_seed0 --resume
echo 'Starting the remaining experiment queue'
"$PY" -u scripts/continue_gaze_vlm_study.py
touch /mnt/hdd/Garvit/server_pipeline_complete
'''
    sftp = client.open_sftp()
    with sftp.open(REMOTE + '/server_pipeline.sh', 'w') as stream:
        stream.write(setup)
    existing = command(client, "tmux has-session -t gaze-study 2>/dev/null && echo running || true")
    if not existing.strip():
        command(client, 'tmux new-session -d -s gaze-study "bash /mnt/hdd/Garvit/server_pipeline.sh"')
    client.close()
    study = BASE / 'gaze_vlm_study'
    metadata = [(p, p.relative_to(BASE).as_posix()) for p in study.glob('*.json*')
                if 'queue' not in p.name and 'download' not in p.name]
    metadata.extend((p, p.relative_to(BASE).as_posix()) for p in (study / 'flow_heatmaps').rglob('*.npz'))
    run = study / 'runs/B1_seed0'
    metadata.extend((p, p.relative_to(BASE).as_posix()) for p in run.rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.log', '.tmp'])
    pack(STATE / 'metadata_checkpoint.tar', metadata)
    upload((STATE / 'metadata_checkpoint.tar', REMOTE + '/metadata_checkpoint.tar'))
    client = connect()
    command(client, 'tar -xf ' + REMOTE + '/metadata_checkpoint.tar -C ' + REMOTE + '/HoloAssist')
    client.close()
    recordings = set()
    for split in ['train', 'dev', 'test']:
        recordings.update(json.loads(line)['recording'] for line in (study / (split + '.jsonl')).read_text().splitlines())
    files = []
    for name in sorted(recordings):
        directory = BASE / 'study_data/subset' / name
        files.extend(p for p in directory.rglob('*') if p.is_file())
    files.extend(p for p in (study / 'models').rglob('*') if p.is_file() and '.cache' not in p.parts)
    total = sum(p.stat().st_size for p in files)
    if total > 400_000_000_000:
        raise RuntimeError('Transfer exceeds user limit')
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(upload, (p, REMOTE + '/HoloAssist/' + p.relative_to(BASE).as_posix())) for p in files]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            done += future.result()
            status = {'stage': 'transferring', 'files_done': i, 'files_total': len(files),
                      'bytes_verified': done, 'bytes_total': total, 'updated': time.time()}
            (STATE / 'status.json').write_text(json.dumps(status, indent=2))
            if i % 20 == 0:
                print(json.dumps(status), flush=True)
    client = connect()
    command(client, 'touch ' + REMOTE + '/transfer_complete')
    client.close()
    (STATE / 'status.json').write_text(json.dumps({'stage': 'transfer_complete', 'bytes_verified': done,
        'server_session': 'gaze-study', 'pipeline': REMOTE + '/server_pipeline.log'}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        (STATE / 'failure.json').write_text(json.dumps({'error': repr(error), 'time': time.time()}))
        raise
