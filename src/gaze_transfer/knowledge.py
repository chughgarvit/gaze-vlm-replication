"""Evidence-checked weak teacher supervision and uncertain symbolic state."""
import json
import math


TEACHER_PROMPT = '''Describe only what is visible in these chronological video frames.
Return one JSON object, no markdown, using this schema:
{"caption":"short visible action description","entities":[{"id":"e1","name":"object name","frame":0,"box":[0.1,0.2,0.4,0.6],"confidence":0.8}],"facts":[{"subject":"e1","predicate":"held","value":"true","frame":0,"confidence":0.8}],"relations":[{"before":0,"after":1,"predicate":"visible_change","confidence":0.7}]}
Frame indices start at 0. Boxes are normalized xyxy and must locate visible entities.
Use value "unknown" when a state cannot be seen; omit unsupported facts.
Never infer correctness, hidden intention, unseen objects, or future events.'''


def parse_json(text):
    begin, end = text.find('{'), text.rfind('}')
    if begin < 0 or end < begin:
        raise ValueError('No JSON object')
    return json.loads(text[begin:end+1])


def validate_concepts(value, frames):
    if not isinstance(value, dict) or not isinstance(value.get('caption'), str):
        raise ValueError('Missing caption')
    entities, facts, relations = [], [], []
    known = set()
    def confidence(item):
        c = float(item['confidence'])
        if not math.isfinite(c) or not 0 <= c <= 1:
            raise ValueError('Invalid confidence')
        return c
    def frame(index):
        if type(index) is not int or not 0 <= index < frames:
            raise ValueError('Evidence frame outside observed window')
    for item in value.get('entities', []):
        frame(item['frame']); confidence(item)
        box = item['box']
        if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in box):
            raise ValueError('Invalid bounding box')
        if box[0] >= box[2] or box[1] >= box[3] or item['id'] in known:
            raise ValueError('Invalid entity identity or area')
        if not isinstance(item['name'], str) or not item['name'].strip():
            raise ValueError('Missing entity name')
        known.add(item['id']); entities.append(item)
    for item in value.get('facts', []):
        frame(item['frame']); confidence(item)
        if item['subject'] not in known or item['value'] not in ['true', 'false', 'unknown']:
            raise ValueError('Ungrounded fact')
        if not isinstance(item['predicate'], str):
            raise ValueError('Invalid predicate')
        facts.append(item)
    for item in value.get('relations', []):
        frame(item['before']); frame(item['after']); confidence(item)
        if item['before'] >= item['after']:
            raise ValueError('Noncausal temporal relation')
        relations.append(item)
    return {'caption': value['caption'], 'entities': entities, 'facts': facts, 'relations': relations}


class SymbolicState:
    """Category-level uncertain ledger; it does not assert tracked object identity."""
    def __init__(self, threshold=.7):
        self.threshold, self.predicates, self.last_cutoff = threshold, {}, -1.

    def update(self, concepts, cutoff):
        if cutoff <= self.last_cutoff:
            raise ValueError('State update must be chronological')
        self.last_cutoff = cutoff
        names = {e['id']: e['name'] for e in concepts['entities']}
        conflicts = []
        transitions = [r for r in concepts['relations'] if r['confidence'] >= self.threshold]
        for fact in sorted(concepts['facts'], key=lambda x: x['frame']):
            if fact['confidence'] < self.threshold:
                continue
            key = names[fact['subject']]+'::'+fact['predicate']
            previous = self.predicates.get(key)
            value = fact['value']
            # Same-window opposing claims without a temporal change become unknown.
            if previous and previous['cutoff'] == cutoff and previous['value'] != value:
                explained = any(r['before'] <= previous['frame'] < fact['frame'] <= r['after']
                                for r in transitions)
                if not explained:
                    conflicts.append(key); value = 'unknown'
            self.predicates[key] = {'value': value, 'cutoff': cutoff, 'frame': fact['frame'],
                                    'confidence': fact['confidence']}
        return {'predicates': dict(self.predicates), 'conflicts': conflicts,
                'physical_identity': 'unverified', 'last_cutoff': cutoff}
