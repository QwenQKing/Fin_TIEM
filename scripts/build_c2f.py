from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from foresight.textkg.episode_builder import EpisodeBuilder
from foresight.textkg.theme_builder import ThemeBuilder
import foresight.config as cfg
DS_LIST = ['Astock', 'FinPURE', 'CMIN-US_ood', 'EDT_ood', 'CSMD_ood']
WINDOW_OVERRIDE = {'EDT_ood': 30}

def build_one_ds(ds: str, force: bool, l2_only: bool, l3_only: bool, window_days: int):
    store_dir = ROOT / f'databases/{ds}/02_event'
    ds_data_path = ROOT / f'datasets/data/{ds}.json'
    done_flag = store_dir / '.done_c2f'
    if done_flag.exists() and (not force):
        info = json.loads(done_flag.read_text(encoding='utf-8'))
        print(f"[{ds}] ✅  (eps={info.get('n_eps', '?')}, themes={info.get('n_themes', '?')}), skip")
        return True
    if force:
        for fname in ('episode_vdb.json', 'episode_graph.graphml', 'theme_vdb.json', 'theme_graph.graphml', '.done_episodes', '.done_themes', '.skipped_l2_reason', '.skipped_l3_reason', '.done_c2f'):
            p = store_dir / fname
            if p.exists():
                p.unlink()
                print(f'  [force] removed {p.name}')
    win = WINDOW_OVERRIDE.get(ds, window_days)
    print(f'\n[{ds}] 02_event L2/L3 build (window={win}d, l2_only={l2_only}, l3_only={l3_only})')
    stock_names = {}
    if ds_data_path.exists():
        j = json.loads(ds_data_path.read_text(encoding='utf-8'))
        stocks = j.get('stocks', {}) or {}
        for (sid, info) in stocks.items():
            if isinstance(info, dict):
                stock_names[sid] = info.get('name', sid)
    n_eps = 0
    n_themes = 0
    if not l3_only:
        try:
            eb = EpisodeBuilder(store_dir)
            eps = eb.build(max_window_days=win, stock_names=stock_names)
            n_eps = len(eps)
            print(f'[{ds}] L2 built: {n_eps} episodes')
        except Exception as e:
            print(f'[{ds}] ❌ L2 fail: {type(e).__name__}: {e}', file=sys.stderr)
            return False
    if not l2_only:
        ep_path = store_dir / 'episode_vdb.json'
        n_existing = 0
        if ep_path.exists():
            n_existing = len(json.loads(ep_path.read_text(encoding='utf-8')))
        if n_existing < 50:
            print(f'[{ds}] ⚠ L2 episode  {n_existing} < 50,  L3 (sparse data)')
            (store_dir / '.skipped_l3_reason').write_text(f'sparse_l2_{n_existing}')
        else:
            try:
                tb = ThemeBuilder(store_dir, ds_data_path)
                themes = tb.build()
                n_themes = len(themes)
                print(f'[{ds}] L3 built: {n_themes} themes')
            except Exception as e:
                print(f'[{ds}] ❌ L3 fail: {type(e).__name__}: {e}', file=sys.stderr)
                return False
    done_flag.write_text(json.dumps({'n_eps': n_eps, 'n_themes': n_themes, 'window_days': win, 'built_at': time.strftime('%Y-%m-%d %H:%M:%S')}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'[{ds}] ✅ done. eps={n_eps}, themes={n_themes}')
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default=None)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--l2-only', action='store_true')
    ap.add_argument('--l3-only', action='store_true')
    ap.add_argument('--window', type=int, default=cfg.C2F_L2_WINDOW_DAYS)
    args = ap.parse_args()
    ds_to_run = [args.ds] if args.ds else DS_LIST
    summary = {}
    for ds in ds_to_run:
        ok = build_one_ds(ds, args.force, args.l2_only, args.l3_only, args.window)
        summary[ds] = 'ok' if ok else 'fail'
    print('\n' + '=' * 50)
    for (ds, st) in summary.items():
        mark = '✅' if st == 'ok' else '❌'
        print(f'  {mark} {ds}: {st}')
    if any((s == 'fail' for s in summary.values())):
        sys.exit(1)
if __name__ == '__main__':
    main()