import sys, json
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from foresight.reason_loop import summarize, print_summary, CASE_LOG, case_root, _safe_model
import foresight.config as cfg

def _arg(flag):
    a = sys.argv
    return a[a.index(flag) + 1] if flag in a and a.index(flag) + 1 < len(a) else None

def main():
    model = _arg('--model') or cfg.LLM_MODEL
    dataset = next((a for a in sys.argv[1:] if not a.startswith('-')), None)
    if not dataset:
        mdir = CASE_LOG / _safe_model(model)
        subs = sorted((d.name for d in mdir.iterdir() if d.is_dir())) if mdir.exists() else []
        print(f'please data (case_log/{_safe_model(model)} : {subs})')
        return
    ddir = case_root(dataset, model)
    cases = sorted(ddir.glob('*/case.json'))
    if not cases:
        print(f'no case log: {ddir}\nplease  scripts/eval.sh (predict case_log)')
        return
    rows = []
    for cf in cases:
        c = json.loads(cf.read_text(encoding='utf-8'))
        gt = c.get('GROUND_TRUTH', {})
        u = c.get('OUTPUT', {}).get('usage', {})
        rows.append({'correct': bool(c['EVAL']['correct']), 'conf': float(c['OUTPUT'].get('pred_conf_verbal', 0.5)), 'true_dir': 1 if gt.get('true_direction') == 'positive' else -1, 'pt': int(u.get('prompt_tokens', 0)), 'ct': int(u.get('completion_tokens', 0))})
    s = summarize(rows)
    print(f'data: {dataset}  |   {len(rows)}  case')
    print_summary(s)
    out = ddir / '_eval.json'
    out.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f': {out.relative_to(ROOT)}')
if __name__ == '__main__':
    main()