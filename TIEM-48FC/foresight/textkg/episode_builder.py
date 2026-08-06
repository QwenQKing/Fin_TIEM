from __future__ import annotations
import hashlib
import json
import os
import random
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import networkx as nx
from tqdm import tqdm
import foresight.config as cfg
from foresight.retry import with_retry
from foresight.textkg.extractor import _clean_json, _repair_json
from foresight.textkg.episode_prompts import EPISODE_TYPES_8, EPISODE_DETECT_PROMPT, EPISODE_CLASSIFY_PROMPT, EPISODE_SUMMARIZE_PROMPT

def _md5(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def _atomic_write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (fd, tmp) = tempfile.mkstemp(dir=str(path.parent), prefix='.tmp_', suffix=path.suffix)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            if isinstance(data, (dict, list)):
                json.dump(data, f, ensure_ascii=False, indent=1)
            else:
                f.write(data)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        if len(ts) >= 16:
            return datetime.strptime(ts[:16], '%Y-%m-%d %H:%M')
        return datetime.strptime(ts[:10], '%Y-%m-%d')
    except Exception:
        return None

class EpisodeBuilder:

    def __init__(self, store_dir: Path, llm_caller=None, chunks: dict=None):
        self.store_dir = Path(store_dir)
        self._llm = llm_caller
        self._chunks = chunks
        self._cache_path = self.store_dir / '.episode_llm_cache.json'
        self._cache = self._load_cache()
        self._cache_lock = threading.Lock()
        random.seed(cfg.GLOBAL_SEED)

    def _load_cache(self) -> dict:
        if not self._cache_path.exists():
            return {}
        try:
            return json.loads(self._cache_path.read_text(encoding='utf-8'))
        except Exception:
            print(f'⚠ episode LLM cache , ', file=sys.stderr)
            return {}

    def _save_cache(self) -> None:
        with self._cache_lock:
            cache_snapshot = dict(self._cache)
        _atomic_write(self._cache_path, cache_snapshot)

    def _load_l1_edges(self) -> list[dict]:
        g_path = self.store_dir / 'knowledge_graph.graphml'
        if not g_path.exists():
            print(f'⚠ {g_path} ,  L1', file=sys.stderr)
            return []
        g = nx.read_graphml(str(g_path))
        if self._chunks is None:
            cp = self.store_dir / 'chunks.json'
            self._chunks = json.loads(cp.read_text(encoding='utf-8')) if cp.exists() else {}
        edges = []
        entity_count = {}
        for (u, v, d) in g.edges(data=True):
            if d.get('role') == 'entity_to_hyperedge':
                entity_count[v] = entity_count.get(v, 0) + 1
                entity_count[u] = entity_count.get(u, 0) + 1
        for (nid, attr) in g.nodes(data=True):
            if attr.get('role') != 'hyperedge':
                continue
            stock = attr.get('stock', '')
            ts = attr.get('ts', '')
            if not stock or not ts:
                continue
            chunk_id = attr.get('source_id', '')
            chunk = self._chunks.get(chunk_id, {}) if chunk_id else {}
            title = chunk.get('content', '')[:80] if chunk else ''
            edges.append({'id': nid, 'ts': ts, 'stock': stock, 'source': attr.get('source', ''), 'title': title, 'n_entities': entity_count.get(nid, 0)})
        return edges

    def _slide_windows(self, edges: list[dict], window_days: int) -> list[dict]:
        from collections import defaultdict
        by_stock = defaultdict(list)
        for e in edges:
            by_stock[e['stock']].append(e)
        windows = []
        for (stock, es) in by_stock.items():
            es.sort(key=lambda e: e['ts'])
            i = 0
            while i < len(es):
                start_ts = _parse_ts(es[i]['ts'])
                if start_ts is None:
                    i += 1
                    continue
                end_ts = start_ts + timedelta(days=window_days)
                j = i + 1
                while j < len(es):
                    cur = _parse_ts(es[j]['ts'])
                    if cur is None or cur > end_ts:
                        break
                    j += 1
                if j - i >= 2:
                    win_edges = es[i:j]
                    if len(win_edges) > cfg.C2F_L2_MAX_PER_WINDOW:
                        win_edges = sorted(win_edges, key=lambda e: -e.get('n_entities', 0))[:cfg.C2F_L2_MAX_PER_WINDOW]
                        win_edges.sort(key=lambda e: e['ts'])
                    windows.append({'stock': stock, 'edges': win_edges, 'start': es[i]['ts'], 'end': es[j - 1]['ts']})
                    i = j
                else:
                    i += 1
        return windows

    def _call_llm(self, prompt: str, scope: str, max_tokens: int=500) -> Optional[dict]:
        ckey = _md5(prompt + '\x00' + cfg.PROMPT_VERSION + '\x00' + scope)
        with self._cache_lock:
            if ckey in self._cache:
                return self._cache[ckey]
        from foresight.reason_loop import _chat
        try:
            r = with_retry(lambda : _chat([{'role': 'user', 'content': prompt}], max_tokens), max_retry=cfg.LLM_MAX_RETRY, backoff=cfg.LLM_RETRY_BACKOFF, what=scope)
            raw = r.choices[0].message.content or ''
            cleaned = _repair_json(_clean_json(raw))
            data = json.loads(cleaned)
            with self._cache_lock:
                self._cache[ckey] = data
            return data
        except Exception as e:
            print(f'  ⚠ {scope} LLM fail: {type(e).__name__}: {str(e)[:60]}', file=sys.stderr)
            return None

    def _format_events_block(self, window: dict) -> str:
        lines = []
        for e in window['edges']:
            title = e.get('title', '').strip().replace('\n', ' ')[:200]
            lines.append(f"  [{e['ts']}] ({e['stock']}) — {title}")
        return '\n'.join(lines)

    def _detect_episode(self, window: dict, stock_name: str) -> dict:
        events_block = self._format_events_block(window)
        prompt = EPISODE_DETECT_PROMPT.format(stock_name=stock_name or window['stock'], stock_id=window['stock'], start=window['start'], end=window['end'], events_block=events_block)
        votes_yes = 0
        confs = []
        for vote_i in range(cfg.C2F_DETECT_VOTES):
            result = self._call_llm(prompt + f'\n# vote_{vote_i}', 'episode_detect', max_tokens=200)
            if not isinstance(result, dict):
                continue
            is_ep = bool(result.get('is_episode', False))
            conf = float(result.get('confidence', 0) or 0)
            if is_ep and conf >= cfg.C2F_DETECT_CONF_THRESHOLD:
                votes_yes += 1
                confs.append(conf)
        passed = votes_yes >= cfg.C2F_DETECT_VOTES_PASS
        return {'is_episode': passed, 'confidence': sum(confs) / len(confs) if confs else 0.0, 'votes_yes': votes_yes, 'votes_total': cfg.C2F_DETECT_VOTES}

    def _classify_episode(self, window: dict) -> str:
        events_block = self._format_events_block(window)
        prompt = EPISODE_CLASSIFY_PROMPT.format(events_block=events_block)
        result = self._call_llm(prompt, 'episode_classify', max_tokens=100)
        if not isinstance(result, dict):
            return 'other'
        t = result.get('type', 'other')
        return t if t in EPISODE_TYPES_8 else 'other'

    def _summarize_episode(self, window: dict) -> str:
        events_block = self._format_events_block(window)
        prompt = EPISODE_SUMMARIZE_PROMPT.format(events_block=events_block)
        result = self._call_llm(prompt, 'episode_summarize', max_tokens=300)
        if isinstance(result, dict) and result.get('summary'):
            return str(result['summary'])[:200]
        titles = ' | '.join((e.get('title', '')[:30] for e in window['edges']))
        return titles[:200]

    def build(self, max_window_days: int=None, stock_names: dict=None) -> list[dict]:
        window_days = max_window_days or cfg.C2F_L2_WINDOW_DAYS
        stock_names = stock_names or {}
        done_flag = self.store_dir / '.done_episodes'
        if done_flag.exists():
            info = json.loads(done_flag.read_text(encoding='utf-8'))
            print(f"  ✅ episodes  ({info.get('n', 0)} ), skip")
            ep_path = self.store_dir / 'episode_vdb.json'
            return json.loads(ep_path.read_text(encoding='utf-8')) if ep_path.exists() else []
        print(f'  📚 load L1 edges ...')
        edges = self._load_l1_edges()
        print(f'  {len(edges)} L1 edges loaded')
        if not edges:
            _atomic_write(self.store_dir / '.skipped_l2_reason', 'no_l1_edges')
            return []
        print(f'  🔍  {window_days}d ...')
        windows = self._slide_windows(edges, window_days)
        print(f'  {len(windows)}  windows')
        if not windows:
            _atomic_write(self.store_dir / '.skipped_l2_reason', 'no_windows')
            return []
        n_workers = min(cfg.LLM_WORKERS, max(1, len(windows)))
        print(f'  ⚡ concurrent: {n_workers} workers')
        episodes = []
        episodes_lock = threading.Lock()

        def _process_one_window(window) -> Optional[dict]:
            assert len(window['edges']) >= 2
            stock_name = stock_names.get(window['stock'], window['stock'])
            detect = self._detect_episode(window, stock_name)
            if not detect['is_episode']:
                return None
            ep_type = self._classify_episode(window)
            summary = self._summarize_episode(window)
            subedge_ids = [e['id'] for e in window['edges']]
            ep_id = 'ep_' + _md5(f"{window['stock']}|{window['start']}|{window['end']}|{ep_type}")
            return {'episode_id': ep_id, 'type': ep_type, 'summary': summary, 'stock': window['stock'], 'subedge_ids': subedge_ids, 'tau_start': window['start'], 'tau_end': window['end'], 'confidence': detect['confidence'], 'n_subedges': len(subedge_ids), '_built_at': time.strftime('%Y-%m-%d %H:%M:%S'), '_prompt_version': cfg.PROMPT_VERSION}
        with ThreadPoolExecutor(max_workers=n_workers) as exe:
            futs = {exe.submit(_process_one_window, w): w for w in windows}
            done_count = 0
            for fut in tqdm(as_completed(futs), total=len(futs), desc='  episode-build'):
                try:
                    ep = fut.result()
                except Exception as e:
                    print(f'  ⚠ window fail: {type(e).__name__}: {str(e)[:60]}', file=sys.stderr)
                    ep = None
                if ep is not None:
                    with episodes_lock:
                        episodes.append(ep)
                done_count += 1
                if done_count % 50 == 0:
                    self._save_cache()
        _atomic_write(self.store_dir / 'episode_vdb.json', episodes)
        self._save_cache()
        from collections import Counter
        type_dist = Counter((e['type'] for e in episodes))
        _atomic_write(done_flag, {'n': len(episodes), 'type_distribution': dict(type_dist), 'window_days': window_days, 'built_at': time.strftime('%Y-%m-%d %H:%M:%S')})
        if episodes and type_dist.get('other', 0) / len(episodes) > 0.5:
            print(f"  ⚠ 'other'  > 50% ({type_dist.get('other')}/{len(episodes)}), prompt need", file=sys.stderr)
        print(f'  ✅ built {len(episodes)} episodes, types={dict(type_dist.most_common())}')
        return episodes