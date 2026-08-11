from __future__ import annotations
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm
import foresight.config as cfg
from foresight.retry import with_retry
from foresight.textkg.extractor import _clean_json, _repair_json
from foresight.textkg.theme_prompts import THEME_TYPES_6, THEME_NAME_PROMPT, THEME_CLASSIFY_PROMPT

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
                f.write(str(data))
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
try:
    import hdbscan
    _HAS_HDBSCAN = True
except ImportError:
    _HAS_HDBSCAN = False

def _cluster_episodes(embeddings: np.ndarray) -> np.ndarray:
    n = embeddings.shape[0]
    if n < cfg.C2F_L3_MIN_CLUSTER_SIZE:
        return np.array([-1] * n)
    if _HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=cfg.C2F_L3_MIN_CLUSTER_SIZE, cluster_selection_method='eom', metric='euclidean')
        return clusterer.fit_predict(embeddings.astype(np.float64))
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=0.85, min_samples=cfg.C2F_L3_MIN_CLUSTER_SIZE).fit_predict(embeddings)

class ThemeBuilder:

    def __init__(self, store_dir: Path, ds_data_path: Path, embedder=None, llm_caller=None):
        self.store_dir = Path(store_dir)
        self.ds_data_path = Path(ds_data_path)
        self._embedder = embedder
        self._llm = llm_caller
        self._cache_path = self.store_dir / '.theme_llm_cache.json'
        self._cache = self._load_cache()
        random.seed(cfg.GLOBAL_SEED)
        np.random.seed(cfg.GLOBAL_SEED)

    def _load_cache(self) -> dict:
        if not self._cache_path.exists():
            return {}
        try:
            return json.loads(self._cache_path.read_text(encoding='utf-8'))
        except Exception:
            print(f'⚠ theme LLM cache , ', file=sys.stderr)
            return {}

    def _save_cache(self) -> None:
        _atomic_write(self._cache_path, self._cache)

    def _get_embedder(self):
        if self._embedder is None:
            from foresight.textkg.embedder import Embedder
            self._embedder = Embedder(cfg.OPENAI_API_KEY, cfg.OPENAI_BASE_URL, cfg.EMBED_MODEL, cfg.EMBED_DIM)
        return self._embedder

    def _load_episodes(self) -> list[dict]:
        ep_path = self.store_dir / 'episode_vdb.json'
        if not ep_path.exists():
            return []
        return json.loads(ep_path.read_text(encoding='utf-8'))

    def _load_stocks_industry(self) -> dict[str, str]:
        if not self.ds_data_path.exists():
            return {}
        j = json.loads(self.ds_data_path.read_text(encoding='utf-8'))
        return {sid: info.get('industry_l1') if isinstance(info, dict) else None for (sid, info) in (j.get('stocks') or {}).items()}

    def _embed_episodes(self, eps: list[dict]) -> np.ndarray:
        cache_p = self.store_dir / '.theme_embed_cache.npz'
        try:
            if cache_p.exists():
                z = np.load(str(cache_p), allow_pickle=False)
                if z['n'].item() == len(eps):
                    return z['embeddings']
        except Exception:
            pass
        emb = self._get_embedder()
        texts = [ep.get('summary', '') or '' for ep in eps]
        vecs = emb.embed(texts) if hasattr(emb, 'embed') else np.array([emb.embed_one(t) for t in tqdm(texts, desc='  embed-theme')])
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-08
        arr = arr / norms
        np.savez_compressed(str(cache_p), embeddings=arr, n=np.array(len(eps)))
        return arr

    def _cluster_in_window(self, eps_subset: list[dict], embs_subset: np.ndarray, industries_map: dict[str, str]) -> list[list[int]]:
        labels = _cluster_episodes(embs_subset)
        from collections import defaultdict
        groups = defaultdict(list)
        for (idx, lab) in enumerate(labels):
            if lab == -1:
                continue
            groups[lab].append(idx)
        valid_clusters = []
        for (lab, idxs) in groups.items():
            if len(idxs) > cfg.C2F_L3_MAX_CLUSTER_SIZE:

                def score(i):
                    ep = eps_subset[i]
                    return ep.get('n_subedges', 0) + (len(ep.get('tau_end', '')) > 0)
                idxs = sorted(idxs, key=lambda i: -score(i))[:cfg.C2F_L3_MAX_CLUSTER_SIZE]
            if len(idxs) < cfg.C2F_L3_MIN_CLUSTER_SIZE:
                continue
            stocks_in_c = set((eps_subset[i]['stock'] for i in idxs))
            if len(stocks_in_c) < cfg.C2F_L3_MIN_STOCKS:
                continue
            industries_in_c = set()
            for s in stocks_in_c:
                ind = industries_map.get(s)
                if ind and ind not in ('Unknown', ''):
                    industries_in_c.add(ind)
            if len(industries_in_c) < cfg.C2F_L3_MIN_INDUSTRIES:
                continue
            valid_clusters.append(idxs)
        return valid_clusters

    def _call_llm(self, prompt: str, scope: str, max_tokens: int=400) -> Optional[dict]:
        ckey = _md5(prompt + '\x00' + cfg.PROMPT_VERSION + '\x00' + scope)
        if ckey in self._cache:
            return self._cache[ckey]
        from foresight.reason_loop import _chat
        try:
            r = with_retry(lambda : _chat([{'role': 'user', 'content': prompt}], max_tokens), max_retry=cfg.LLM_MAX_RETRY, backoff=cfg.LLM_RETRY_BACKOFF, what=scope)
            raw = r.choices[0].message.content or ''
            data = json.loads(_repair_json(_clean_json(raw)))
            self._cache[ckey] = data
            return data
        except Exception as e:
            print(f'  ⚠ {scope} LLM fail: {type(e).__name__}: {str(e)[:60]}', file=sys.stderr)
            return None

    def _name_and_classify(self, cluster_eps: list[dict], industries: set[str]) -> Optional[dict]:
        episodes_block = '\n'.join((f"  [{ep['tau_start'][:10]} | {ep['stock']} | {ep['type']}] {ep['summary'][:120]}" for ep in cluster_eps[:10]))
        stocks_list = sorted(set((ep['stock'] for ep in cluster_eps)))
        ind_list = sorted(industries)
        name_result = self._call_llm(THEME_NAME_PROMPT.format(episodes_block=episodes_block, stocks=stocks_list, industries=ind_list), 'theme_name', max_tokens=400)
        if not isinstance(name_result, dict):
            return None
        if not name_result.get('is_theme'):
            return None
        conf = float(name_result.get('confidence', 0) or 0)
        if conf < cfg.C2F_L3_NAME_CONF_THRESHOLD:
            return None
        name = (name_result.get('name', '') or '')[:100]
        desc = (name_result.get('description', '') or '')[:500]
        classify_result = self._call_llm(THEME_CLASSIFY_PROMPT.format(theme_name=name, theme_desc=desc, stocks=stocks_list, sample_episodes=episodes_block), 'theme_classify', max_tokens=100)
        theme_type = 'other'
        if isinstance(classify_result, dict):
            t = classify_result.get('type', 'other')
            theme_type = t if t in THEME_TYPES_6 else 'other'
        return {'name': name, 'description': desc, 'type': theme_type, 'confidence': conf}

    def _dedup_themes(self, themes: list[dict]) -> list[dict]:
        if not themes:
            return themes
        emb = self._get_embedder()
        names = [t['name'] for t in themes]
        vecs = np.asarray(emb.embed(names) if hasattr(emb, 'embed') else [emb.embed_one(n) for n in names], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-08
        vecs = vecs / norms
        order = sorted(range(len(themes)), key=lambda i: -themes[i]['confidence'])
        kept_indices = []
        merged_into = {}
        for i in order:
            merged = False
            for j in kept_indices:
                cos = float(np.dot(vecs[i], vecs[j]))
                if cos > cfg.C2F_L3_DEDUP_COS_THRESHOLD:
                    tj = themes[j]
                    ti = themes[i]
                    tj['stocks'] = sorted(set(tj['stocks']) | set(ti['stocks']))
                    tj['subepisode_ids'] = list(set(tj['subepisode_ids']) | set(ti['subepisode_ids']))
                    tj['industries'] = sorted(set(tj['industries']) | set(ti['industries']))
                    if ti['tau_start'] < tj['tau_start']:
                        tj['tau_start'] = ti['tau_start']
                    if ti['tau_end'] > tj['tau_end']:
                        tj['tau_end'] = ti['tau_end']
                    tj['n_subepisodes'] = len(tj['subepisode_ids'])
                    merged_into[i] = j
                    merged = True
                    break
            if not merged:
                kept_indices.append(i)
        return [themes[i] for i in kept_indices]

    def build(self) -> list[dict]:
        done_flag = self.store_dir / '.done_themes'
        if done_flag.exists():
            info = json.loads(done_flag.read_text(encoding='utf-8'))
            print(f"  ✅ themes  ({info.get('n', 0)} ), skip")
            th_path = self.store_dir / 'theme_vdb.json'
            return json.loads(th_path.read_text(encoding='utf-8')) if th_path.exists() else []
        eps = self._load_episodes()
        print(f'  📚 {len(eps)} L2 episodes loaded')
        if not eps:
            _atomic_write(self.store_dir / '.skipped_l3_reason', 'no_l2_episodes')
            return []
        industries_map = self._load_stocks_industry()
        print(f"  📚 {sum((1 for v in industries_map.values() if v and v != 'Unknown'))}/{len(industries_map)} stocks have industry tag")
        print(f'  🔢 embed episodes ...')
        embs = self._embed_episodes(eps)
        ts_keys = [_parse_ts(ep['tau_end']) for ep in eps]
        valid_eps_idx = [i for (i, t) in enumerate(ts_keys) if t is not None]
        if not valid_eps_idx:
            _atomic_write(self.store_dir / '.skipped_l3_reason', 'no_valid_ts')
            return []
        all_ts_sorted = sorted((ts_keys[i] for i in valid_eps_idx))
        t_start = all_ts_sorted[0]
        t_end = all_ts_sorted[-1]
        themes = []
        cursor = t_start
        n_windows = 0
        while cursor <= t_end:
            win_end = cursor + timedelta(days=cfg.C2F_L3_WINDOW_DAYS)
            win_idxs = [i for i in valid_eps_idx if cursor <= ts_keys[i] <= win_end]
            if len(win_idxs) >= cfg.C2F_L3_MIN_CLUSTER_SIZE:
                eps_subset = [eps[i] for i in win_idxs]
                embs_subset = embs[win_idxs]
                clusters = self._cluster_in_window(eps_subset, embs_subset, industries_map)
                for cluster_idxs in clusters:
                    cluster_eps = [eps_subset[i] for i in cluster_idxs]
                    stocks_in_c = set((ep['stock'] for ep in cluster_eps))
                    industries_in_c = set()
                    for s in stocks_in_c:
                        ind = industries_map.get(s)
                        if ind and ind not in ('Unknown', ''):
                            industries_in_c.add(ind)
                    info = self._name_and_classify(cluster_eps, industries_in_c)
                    if info is None:
                        continue
                    theme_id = 'th_' + _md5(','.join(sorted(stocks_in_c)) + '|' + info['name'])
                    themes.append({'theme_id': theme_id, 'type': info['type'], 'name': info['name'], 'description': info['description'], 'stocks': sorted(stocks_in_c), 'industries': sorted(industries_in_c), 'subepisode_ids': [ep['episode_id'] for ep in cluster_eps], 'tau_start': min((ep['tau_start'] for ep in cluster_eps)), 'tau_end': max((ep['tau_end'] for ep in cluster_eps)), 'confidence': info['confidence'], 'n_subepisodes': len(cluster_eps), '_prompt_version': cfg.PROMPT_VERSION})
            cursor += timedelta(days=cfg.C2F_L3_STEP_DAYS)
            n_windows += 1
        print(f'   {n_windows} ,  {len(themes)} ')
        themes = self._dedup_themes(themes)
        if len(themes) > cfg.C2F_L3_MAX_THEMES_PER_DS:
            themes = sorted(themes, key=lambda t: -(len(t['stocks']) * t.get('n_subepisodes', 1)))[:cfg.C2F_L3_MAX_THEMES_PER_DS]
        print(f'   + cap : {len(themes)} themes')
        _atomic_write(self.store_dir / 'theme_vdb.json', themes)
        self._save_cache()
        from collections import Counter
        type_dist = Counter((t['type'] for t in themes))
        _atomic_write(done_flag, {'n': len(themes), 'type_distribution': dict(type_dist), 'built_at': time.strftime('%Y-%m-%d %H:%M:%S')})
        print(f'  ✅ built {len(themes)} themes, types={dict(type_dist.most_common())}')
        return themes