"""Measured attention summaries; no physiological fixation/physical-ID claim."""
import numpy as np


def attention_structure(gaze, frame_times, start, cutoff, entities):
    times = start+gaze[:, 0]*(cutoff-start)
    valid = gaze[:, 3] > 0
    episodes, current = [], []
    def flush():
        if len(current) > 1 and times[current[-1]]-times[current[0]] >= .1:
            episodes.append({'start': round(float(times[current[0]]-start), 3),
                'end': round(float(times[current[-1]]-start), 3),
                'duration_s': round(float(times[current[-1]]-times[current[0]]), 3),
                'position': gaze[current, 1:3].mean(0).round(4).tolist()})
    for i in range(len(gaze)):
        if not valid[i]:
            flush(); current = []; continue
        candidate = current+[i]
        spread = np.ptp(gaze[candidate, 1:3], axis=0).sum()
        if current and (times[i]-times[current[-1]] > .12 or spread > .07):
            flush(); current = [i]
        else:
            current = candidate
    flush()
    visits = []; previous = None; visited = set()
    for i in np.flatnonzero(valid):
        frame = np.searchsorted(frame_times, times[i], side='right')-1
        if frame < 0 or times[i]-frame_times[frame] > .2:
            previous = None; continue
        x, y = gaze[i, 1:3]
        candidates = [e for e in entities if e['frame'] == frame and e['confidence'] >= .5
                      and e['box'][0] <= x <= e['box'][2] and e['box'][1] <= y <= e['box'][3]]
        if not candidates:
            previous = None; continue
        entity = min(candidates, key=lambda e: (e['box'][2]-e['box'][0])*(e['box'][3]-e['box'][1]))
        identifier = entity['id']
        if previous is not None and previous['entity'] == identifier and times[i]-previous['absolute_end'] <= .12:
            previous['end'] = round(float(times[i]-start), 3)
            previous['absolute_end'] = float(times[i])
            previous['duration_s'] = round(previous['end']-previous['start'], 3)
        else:
            previous = {'entity': identifier, 'frame': int(frame), 'start': round(float(times[i]-start), 3),
                        'end': round(float(times[i]-start), 3), 'duration_s': 0.,
                        'revisited_teacher_id': identifier in visited, 'absolute_end': float(times[i])}
            visits.append(previous); visited.add(identifier)
    for visit in visits:
        visit.pop('absolute_end', None)
    return {'source': 'measured_projected_gaze', 'low_dispersion_episodes': episodes,
            'entity_visits': visits, 'physical_instance_identity': 'unverified',
            'association_max_age_s': .2, 'fixation_validation': 'not_physiologically_validated'}
