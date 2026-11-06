"""Generate the browser review from the actual VLM study specification/status."""
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gaze_transfer.data import json_write

BASE = Path.home()/'Datasets/HoloAssist/gaze_vlm_study'


def main():
    import markdown
    status = {'prepared': False, 'runs': {}, 'teacher': None}
    if (BASE/'prepare_status.json').exists():
        status['data'] = json.loads((BASE/'prepare_status.json').read_text())
        status['prepared'] = status['data']['stage'] == 'complete'
    for path in (BASE/'runs').glob('*/status.json'):
        status['runs'][path.parent.name] = json.loads(path.read_text())
    for name in ['teacher_status', 'model_download_status', 'flow_status', 'queue_status']:
        path = BASE/(name+'.json')
        if path.exists():
            status[name] = json.loads(path.read_text())
    json_write(ROOT/'results/gaze_vlm_study/status.json', status)
    body = markdown.markdown((ROOT/'GAZE_VLM_STUDY.md').read_text(encoding='utf-8'),
                             extensions=['tables', 'fenced_code', 'toc'])
    cards = '<h2>Execution status</h2><p>These are new VLM runs. Engineering checks are not accuracy results.</p>'
    cards += '<pre>'+html.escape(json.dumps(status, indent=2))+'</pre>'
    cards += '<h3>Model components</h3><table><tr><th>Component</th><th>Status</th></tr>'
    for name, state in [
        ('500M student + language LoRA + gaze adapter', 'Implemented; real GPU training and generation checked'),
        ('Training-only attention KL', 'Implemented; finite nonzero KL/backward checked'),
        ('Temporal / static / shuffled gaze', 'Implemented; controlled capacity and invalid-gaze checks'),
        ('Visual frame / post-vision token budgets', 'Implemented; token pruning requires separate measured comparison'),
        ('2.2B offline structured teacher', 'Download/extraction stage; quality and training not established'),
        ('Structured concept supervision and runtime state', 'Implemented; end-to-end teacher/student validation pending'),
        ('Personalization statistics', 'Module exists; wearer identity audit and evaluation pending'),
        ('Conventional distillation comparator', 'Pending'),
        ('Exact strongest OpenFlamingo reproduction', 'Pending original setup/checkpoints and suitable hardware'),
        ('Frozen VLM readouts / held-out-task protocol', 'Extraction/probe script implemented; scientific evaluation pending'),
        ('Cross-dataset transfer', 'Pending dataset access/alignment and evaluation'),
        ('Mobile-device energy', 'Pending device instrumentation; laptop timing is not mobile energy')]:
        cards += '<tr><td>'+html.escape(name)+'</td><td>'+html.escape(state)+'</td></tr>'
    cards += '</table>'
    cards += '<h3>Projection review</h3><img src="results/gaze_vlm_study/projection_spot_check.png" alt="Training gaze projection spot check">'
    css = '''body{font:16px/1.65 system-ui,sans-serif;color:#233047;background:#eef2f8;margin:0}
    main{max-width:1120px;margin:30px auto;background:#fff;padding:36px 48px;border-radius:12px}
    h1,h2,h3{line-height:1.25;color:#102a47}h2{margin-top:2em}a{color:#165cb8}
    table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow:auto}
    th,td{border:1px solid #dae2ed;padding:10px;vertical-align:top;text-align:left}th{background:#edf3fa}
    pre{background:#eff3f8;padding:16px;overflow:auto;max-height:450px}nav{border-bottom:1px solid #ddd;padding-bottom:14px}
    img{max-width:100%}@media(max-width:700px){main{margin:0;padding:18px}}'''
    page = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gaze-VLM full study review</title><style>'+css+'</style></head><body><main><nav>Gaze-VLM + Indrajeet’s extensions · Full study specification and execution status</nav>'+cards+body+'</main></body></html>'
    path = ROOT/'GAZE_VLM_STUDY.html'; path.write_text(page, encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
