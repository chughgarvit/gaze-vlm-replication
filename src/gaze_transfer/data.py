"""Causal windows, explicit label provenance, and calibrated gaze projection."""
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

AXIS = np.linalg.inv(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float))


def stable_seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    # Windows readers can briefly deny replacement of an open status file.
    for attempt in range(40):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.25)


def overlap(event, start, end):
    return max(0., min(float(event['end']), end) - max(float(event['start']), start))


def corrective(event):
    purpose = str(event.get('attributes', {}).get('Conversation Purpose', '')).lower()
    return event['label'] == 'Conversation' and 'correct' in purpose and 'wrong' in purpose


def window_targets(record, start, cutoff, horizon=2.):
    """Labels only; this output must never enter teacher or student input prompts."""
    fine = [e for e in record['events'] if e['label'] == 'Fine grained action']
    observed = [e for e in fine if overlap(e, start, cutoff) > 0]
    dominant = max(observed, key=lambda e: overlap(e, start, cutoff), default=None)
    future = sorted([e for e in fine if cutoff < e['start'] <= cutoff + horizon],
                    key=lambda e: (e['start'], e['end'], str(e['id'])))
    coarse = [e for e in record['events'] if e['label'] == 'Coarse grained action'
              and e['start'] <= cutoff < e['end']]
    state = max(coarse, key=lambda e: e['start'], default=None)
    wrong = [e for e in observed if str(e['attributes'].get('Action Correctness', '')).startswith('Wrong Action')]
    eligible = [e for e in observed if e['attributes'].get('Action Correctness', '') == 'Correct Action'
                or str(e['attributes'].get('Action Correctness', '')).startswith('Wrong Action')]
    def action(e):
        if e is None:
            return None
        a = e['attributes']
        return {'verb': a.get('Verb', 'unknown'), 'object': a.get('Noun', 'unknown')}
    return {'activity': action(dominant), 'anticipation': action(future[0]) if future else {'event': 'none'},
            'mistake': 'incorrect' if wrong else 'correct' if eligible else 'unknown',
            'intervention': any(corrective(e) and cutoff < e['start'] <= cutoff + horizon
                                for e in record['events']),
            'task_state': state['attributes'].get('Action sentence') if state else None,
            'corrective_speech_in_observation': any(corrective(e) and overlap(e, start, cutoff) > 0
                                                   for e in record['events']),
            'label_event_ids': {'observed': [str(e['id']) for e in observed],
                                'future': [str(e['id']) for e in future]}}


class Recording:
    def __init__(self, base):
        self.base = Path(base)
        g = np.loadtxt(self.base / 'Eyes/Eyes_sync.txt', delimiter='\t')
        p = np.loadtxt(self.base / 'Video/Pose_sync.txt', delimiter='\t')
        start, end = map(int, (self.base / 'Video/VideoMp4Timing.txt').read_text().split())
        if g.ndim != 2 or g.shape[1] != 9 or p.ndim != 2 or p.shape[1] != 18:
            raise ValueError('Unsupported source schema')
        self.duration = (end - start) / 1e7
        self.gt = (g[:, 1] - start) / 1e7
        _, indices = np.unique(self.gt, return_index=True)
        self.g, self.gt = g[indices], self.gt[indices]
        self.pt = (p[:, 1] - start) / 1e7
        self.poses = p[:, 2:].reshape(-1, 4, 4)
        if np.any(np.diff(self.pt) < 0):
            raise ValueError('Nonmonotonic camera clock')
        intr = np.loadtxt(self.base / 'Video/Intrinsics.txt').ravel()
        self.K, self.width, self.height = intr[:9].reshape(3, 3), intr[-2], intr[-1]
        self.cap = cv2.VideoCapture(str(self.base / 'Video_compress.mp4'))
        self.fps, self.count = self.cap.get(cv2.CAP_PROP_FPS), int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.fps <= 0 or abs(self.count / self.fps - self.duration) > max(.25, 2 / self.fps):
            self.cap.release()
            raise ValueError('Video/UTC duration mismatch')

    def close(self):
        self.cap.release()

    def gaze(self, start, cutoff, max_samples=64, depth=.5):
        """Publisher ray-at-fixed-depth projection, NOT measured 3D fixation depth."""
        selected = np.flatnonzero((self.gt >= start) & (self.gt <= cutoff))
        if len(selected) > max_samples:
            selected = selected[np.linspace(0, len(selected)-1, max_samples).astype(int)]
        times, g = self.gt[selected], self.g[selected]
        output = np.zeros((max_samples, 5), np.float32)  # relative time, u, v, valid, source age
        if not len(times):
            return output
        pi = np.searchsorted(self.pt, times, side='right') - 1
        poses = self.poses[np.maximum(pi, 0)]
        direction = g[:, 5:8]
        norm = np.linalg.norm(direction, axis=1)
        point = g[:, 2:5] + depth * direction / np.maximum(norm[:, None], 1e-8)
        camera = np.einsum('nij,nj->ni', np.linalg.inv(poses), np.c_[point, np.ones(len(point))])[:, :3] @ AXIS.T
        projected = camera @ self.K.T
        uv = projected[:, :2] / np.where(abs(projected[:, 2:3]) > 1e-8, projected[:, 2:3], np.nan)
        uv /= [self.width, self.height]
        age = times - self.pt[np.maximum(pi, 0)]
        valid = (g[:, -1] > 0) & (norm > .5) & (pi >= 0) & (age <= .1) & (camera[:, 2] > 0)
        valid &= np.isfinite(uv).all(1) & (uv >= 0).all(1) & (uv <= 1).all(1)
        output[:len(times), 0] = (times-start) / max(cutoff-start, 1e-6)
        output[:len(times), 1:3] = np.where(valid[:, None], uv, 0)
        output[:len(times), 3] = valid
        output[:len(times), 4] = np.clip(age, 0, 1)
        return output

    def frames(self, times):
        images = []
        for time in times:
            index = min(self.count-1, max(0, int(time*self.fps)))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self.cap.read()
            if not ok:
                raise ValueError(f'Decode failed at {time}')
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return images


def frame_times(start, cutoff, count=4):
    # Strictly past cutoff, without action-end knowledge.
    return np.linspace(start, cutoff, count, endpoint=False).tolist()


def heatmaps(gaze, times, grid, start, cutoff, sigma=.10):
    """Per-frame preceding-200ms maps; no unimplemented flow is implied."""
    xy = (np.arange(grid)+.5)/grid
    xx, yy = np.meshgrid(xy, xy)
    maps, valid = [], []
    absolute = start+gaze[:, 0]*(cutoff-start)
    for time in times:
        mask = (gaze[:, 3] > 0) & (absolute <= time) & (absolute >= time-.2)
        pts = gaze[mask, 1:3]
        if len(pts):
            m = np.exp(-((xx[None]-pts[:, 0, None, None])**2 +
                         (yy[None]-pts[:, 1, None, None])**2)/(2*sigma**2)).sum(0)
            maps.append((m/m.sum()).ravel()); valid.append(True)
        else:
            maps.append(np.full(grid*grid, 1/(grid*grid))); valid.append(False)
    return np.asarray(maps, np.float32), np.asarray(valid)


def gaze_variant(gaze, mode, seed=0):
    result = gaze.copy()
    valid = result[:, 3] > 0
    if mode == 'shuffled':
        result[valid, 1:3] = result[valid, 1:3][np.random.default_rng(seed).permutation(valid.sum())]
    elif mode == 'static' and valid.any():
        result[valid, 1:3] = result[valid, 1:3].mean(0)
    elif mode == 'missing':
        result[:] = 0
    return result


def select_frames(gaze, count=4, budget=2):
    """Causal fixed-window selector using only gaze, with first-frame context."""
    if budget >= count:
        return list(range(count))
    scores = []
    for i in range(1, count):
        part = gaze[(gaze[:, 0] >= (i-1)/count) & (gaze[:, 0] < i/count) & (gaze[:, 3] > 0)]
        scores.append((float(np.linalg.norm(np.diff(part[:, 1:3], axis=0), axis=1).sum())
                       if len(part) > 1 else 0., i))
    return sorted([0] + [i for _, i in sorted(scores, reverse=True)[:max(0, budget-1)]])
