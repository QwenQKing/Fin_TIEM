from __future__ import annotations
import warnings
from pathlib import Path

def _pit_strict_lt(ts: str, T: str) -> bool:
    if not ts or not T:
        return False
    if len(T) <= 10:
        return ts[:10] < T[:10]
    return ts < T

class TextKG:

    def __init__(self, store_dir: str):
        import foresight.config as cfg
        from foresight.textkg.storage import JsonKVStorage, VectorStorage, GraphStorage
        from foresight.textkg.embedder import Embedder
        from foresight.textkg.extractor import LLMExtractor
        self.cfg = cfg
        d = Path(store_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.kv = JsonKVStorage(str(d / 'chunks.json'))
        self.vdb_entity = VectorStorage(str(d / 'entity_vdb.json'), cfg.EMBED_DIM)
        self.vdb_hyperedge = VectorStorage(str(d / 'hyperedge_vdb.json'), cfg.EMBED_DIM)
        self.graph = GraphStorage(str(d / 'knowledge_graph.graphml'))
        self.embedder = Embedder(cfg.OPENAI_API_KEY, cfg.OPENAI_BASE_URL, cfg.EMBED_MODEL, cfg.EMBED_DIM)
        self.extractor = LLMExtractor(cfg.OPENAI_API_KEY, cfg.OPENAI_BASE_URL, cfg.LLM_MODEL, cache_path=str(cfg.BASE_DIR / '.extract_cache.json'))

    def build(self, text_events: list) -> dict:
        from foresight.textkg.text_builder import build_from_text_events
        return build_from_text_events(text_events, self.kv, self.vdb_entity, self.vdb_hyperedge, self.graph, self.extractor, self.embedder, max_tokens=self.cfg.KB_CHUNK_SIZE, overlap=self.cfg.KB_CHUNK_OVERLAP, extract_batch_size=self.cfg.KB_EXTRACT_BATCH, model=self.cfg.LLM_MODEL)

    def _ensure_index(self):
        if getattr(self, '_idx_built', False):
            return
        cbs: dict = {}
        for (_cid, rec) in self.kv.all().items():
            cbs.setdefault(rec.get('stock', ''), []).append(rec)
        self._chunks_by_stock = cbs
        hbs: dict = {}
        sbs: dict = {}
        for (nid, attr) in self.graph.all_nodes():
            if attr.get('role') == 'hyperedge':
                sent = nid.split(' | Source:')[0].replace('<hyperedge>', '')
                ts = attr.get('ts', '')
                stock = attr.get('stock')
                src = attr.get('source')
                hbs.setdefault(src, []).append({'ts': ts, 'fact': sent, 'stock': stock})
                sbs.setdefault(stock, []).append({'ts': ts, 'fact': sent, 'source': src})
        self._he_by_source = hbs
        self._he_by_stock = sbs
        self._idx_built = True

    def concurrent_events(self, stock: str, T: str, window_days: int=1, max_facts: int=12) -> list:
        self._ensure_index()
        if not T:
            return []
        from datetime import datetime, timedelta
        try:
            Td = datetime.strptime(T[:10], '%Y-%m-%d')
            lo = (Td - timedelta(days=window_days)).strftime('%Y-%m-%d')
        except Exception:
            return []
        hit = [h for h in self._he_by_stock.get(stock, []) if h.get('ts') and lo < h['ts'][:10] <= T[:10]]
        hit.sort(key=lambda h: h.get('ts', ''))
        return hit[:max_facts]

    def retrieve_focal(self, stock: str, T: str, k: int=1, pit_strict: bool=True) -> dict:
        if not pit_strict:
            warnings.warn('PIT-strict disabled in retrieve_focal — leaks future. AUDIT ONLY.', stacklevel=2)
        self._ensure_index()
        by_doc: dict = {}
        for rec in self._chunks_by_stock.get(stock, []):
            ts = rec.get('ts', '')
            if pit_strict and T and ts and (ts[:10] > T[:10] if len(T) <= 10 else ts > T):
                continue
            d = by_doc.setdefault(rec['source'], {'ts': ts, 'chunks': []})
            d['chunks'].append(rec['content'])
            if ts > d['ts']:
                d['ts'] = ts
        if not by_doc:
            return {'text': '', 'facts': [], 'ts': None, 'doc_ids': []}
        focal = sorted(by_doc.items(), key=lambda kv: kv[1]['ts'], reverse=True)[:max(1, k)]
        text = '\n'.join((' '.join(d['chunks']) for (_src, d) in focal))
        (facts, doc_ids) = ([], [])
        for (src, _d) in focal:
            doc_ids.append(src)
            facts.extend(self._he_by_source.get(src, []))
        return {'text': text, 'facts': facts, 'ts': focal[0][1]['ts'], 'doc_ids': doc_ids}

    def relations_of(self, name: str, before_ts: str | None=None, top_k: int=8, max_hops: int=2, decay: float=0.6) -> list:
        if not name:
            return []
        adj: dict = {}
        for (u, v, d) in self.graph.all_edges():
            if d.get('role') != 'relation':
                continue
            ts = d.get('ts', '')
            if before_ts and ts and (ts >= before_ts):
                continue
            su = u.split(' | Source:')[0].strip()
            dv = v.split(' | Source:')[0].strip()
            rel = d.get('rel_type', 'related')
            adj.setdefault(su, []).append((dv, rel, ts, 'out'))
            adj.setdefault(dv, []).append((su, rel, ts, 'in'))
        starts = [e for e in adj if name in e or e in name]
        if not starts:
            return []
        (out, seen_tri, visited) = ([], set(), set(starts))
        frontier = [(s, 1) for s in starts]
        while frontier and len(out) < top_k:
            nxt = []
            for (ent, hop) in frontier:
                for (dst, rel, ts, direction) in adj.get(ent, []):
                    tri = (ent, rel, dst) if direction == 'out' else (dst, rel, ent)
                    if tri in seen_tri:
                        continue
                    seen_tri.add(tri)
                    out.append({'src': tri[0], 'rel': tri[1], 'dst': tri[2], 'ts': ts, 'hop': hop, 'conf': round(decay ** (hop - 1), 3)})
                    if len(out) >= top_k:
                        break
                    if dst not in visited and hop < max_hops:
                        visited.add(dst)
                        nxt.append((dst, hop + 1))
                if len(out) >= top_k:
                    break
            frontier = nxt
        return out

    def query(self, q: str, top_k: int=10, before_ts: str | None=None, stock: str | None=None, min_sim: float | None=None, pit_strict: bool=True) -> list:
        if not pit_strict:
            warnings.warn('PIT-strict disabled in query — leaks future. AUDIT ONLY.', stacklevel=2)
        qv = self.embedder.embed_one(q)
        pool = top_k * (8 if before_ts or stock else 1)
        raw = self.vdb_hyperedge.search(qv, top_k=pool)
        out = []
        for r in raw:
            ts = r.get('ts', '')
            if pit_strict and before_ts and ts and (ts >= before_ts):
                continue
            if stock and r.get('stock') != stock:
                continue
            sc = r.get('__metrics__', r.get('__score__'))
            if min_sim is not None and sc is not None and (float(sc) < min_sim):
                continue
            out.append({'fact': r.get('content'), 'stock': r.get('stock'), 'ts': ts, 'score': float(sc) if sc is not None else None})
            if len(out) >= top_k:
                break
        return out

    def _load_l2_index(self):
        if getattr(self, '_l2_loaded', False):
            return
        import json
        p = Path(self.kv._path).parent / 'episode_vdb.json'
        if p.exists():
            try:
                self._l2_eps = json.loads(p.read_text(encoding='utf-8'))
            except Exception as e:
                warnings.warn(f'episode_vdb.json : {e}', stacklevel=2)
                self._l2_eps = []
        else:
            self._l2_eps = []
        self._l2_loaded = True

    def _load_l3_index(self):
        if getattr(self, '_l3_loaded', False):
            return
        import json
        p = Path(self.kv._path).parent / 'theme_vdb.json'
        if p.exists():
            try:
                self._l3_themes = json.loads(p.read_text(encoding='utf-8'))
            except Exception as e:
                warnings.warn(f'theme_vdb.json : {e}', stacklevel=2)
                self._l3_themes = []
        else:
            self._l3_themes = []
        self._l3_loaded = True

    def retrieve_c2f(self, stock: str, T: str, query: str, k_l1: int | None=None, k_l2: int | None=None, k_l3: int | None=None, pit_strict: bool=True) -> dict:
        import foresight.config as cfg
        if not pit_strict:
            warnings.warn('PIT-strict disabled in retrieve_c2f — leaks future. AUDIT ONLY.', stacklevel=2)
        kk_l1 = cfg.C2F_RETRIEVE_K_L1 if k_l1 is None else k_l1
        kk_l2 = cfg.C2F_RETRIEVE_K_L2 if k_l2 is None else k_l2
        kk_l3 = cfg.C2F_RETRIEVE_K_L3 if k_l3 is None else k_l3
        self._ensure_index()
        self._load_l2_index()
        self._load_l3_index()
        focal = self.retrieve_focal(stock, T, k=1, pit_strict=pit_strict)
        l2_filtered = []
        for ep in self._l2_eps:
            if ep.get('stock') != stock:
                continue
            if pit_strict and (not _pit_strict_lt(ep.get('tau_end', ''), T)):
                continue
            l2_filtered.append(ep)
        if l2_filtered and query:
            try:
                qv = self.embedder.embed_one(query)
                import numpy as np
                qv = np.asarray(qv, dtype=np.float32)
                qv = qv / (float(np.linalg.norm(qv)) + 1e-08)
                summaries = [ep.get('summary', '') for ep in l2_filtered]
                evs = np.asarray(self.embedder.embed(summaries) if hasattr(self.embedder, 'embed') else [self.embedder.embed_one(s) for s in summaries], dtype=np.float32)
                norms = np.linalg.norm(evs, axis=1, keepdims=True) + 1e-08
                evs = evs / norms
                sims = evs @ qv
                order = np.argsort(-sims)
                l2_hits = [{**l2_filtered[int(i)], 'score': float(sims[int(i)])} for i in order[:kk_l2]]
            except Exception:
                l2_filtered.sort(key=lambda e: -float(e.get('confidence', 0)))
                l2_hits = l2_filtered[:kk_l2]
        else:
            l2_filtered.sort(key=lambda e: -float(e.get('confidence', 0)))
            l2_hits = l2_filtered[:kk_l2]
        l3_filtered = []
        for th in self._l3_themes:
            if stock not in (th.get('stocks') or []):
                continue
            if pit_strict and (not _pit_strict_lt(th.get('tau_end', ''), T)):
                continue
            l3_filtered.append(th)
        if l3_filtered and query:
            try:
                qv = self.embedder.embed_one(query)
                import numpy as np
                qv = np.asarray(qv, dtype=np.float32)
                qv = qv / (float(np.linalg.norm(qv)) + 1e-08)
                names = [f"{th.get('name', '')}. {th.get('description', '')}" for th in l3_filtered]
                tvs = np.asarray(self.embedder.embed(names) if hasattr(self.embedder, 'embed') else [self.embedder.embed_one(n) for n in names], dtype=np.float32)
                norms = np.linalg.norm(tvs, axis=1, keepdims=True) + 1e-08
                tvs = tvs / norms
                sims = tvs @ qv
                order = np.argsort(-sims)
                l3_hits = [{**l3_filtered[int(i)], 'score': float(sims[int(i)])} for i in order[:kk_l3]]
            except Exception:
                l3_filtered.sort(key=lambda t: -float(t.get('confidence', 0)))
                l3_hits = l3_filtered[:kk_l3]
        else:
            l3_filtered.sort(key=lambda t: -float(t.get('confidence', 0)))
            l3_hits = l3_filtered[:kk_l3]
        l1_hits = self.query(query, top_k=kk_l1, before_ts=T, stock=stock, pit_strict=pit_strict)
        return {'focal': focal, 'l2': l2_hits, 'l3': l3_hits, 'l1': l1_hits}