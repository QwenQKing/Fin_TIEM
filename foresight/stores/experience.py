from __future__ import annotations
import json, hashlib, os, sys, tempfile
from datetime import datetime, time
from pathlib import Path
import numpy as np

def _exp_id(name: str, source: str) -> str:
    return hashlib.md5(f'{name}|{source}'.encode('utf-8')).hexdigest()

def _ancestor_key(ancestor: dict) -> tuple:
    fields = ('source_catalyst_id', 'decision_time', 'resolve_time',
              'outcome_source_hash', 'distillation_prompt_hash')
    return tuple(str(ancestor.get(field, '')) for field in fields)

def _union_ancestors(*groups) -> list:
    merged = {}
    for group in groups:
        if isinstance(group, list):
            for ancestor in group:
                if isinstance(ancestor, dict):
                    merged[_ancestor_key(ancestor)] = dict(ancestor)
    return [merged[key] for key in sorted(merged)]

def _parse_pit_time(value: str, date_only_eod: bool=True):
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().replace('Z', '+00:00')
    try:
        if len(raw) == 10:
            day = datetime.fromisoformat(raw).date()
            return datetime.combine(day, time.max) if date_only_eod else None
        return datetime.fromisoformat(raw)
    except ValueError:
        return None

def _strict_ancestor_pit(skill: dict, before_date: str, date_only_eod: bool=True) -> bool:
    cutoff = _parse_pit_time(before_date, date_only_eod=date_only_eod)
    ancestors = skill.get('provenance_ancestors')
    required = ('source_catalyst_id', 'decision_time', 'resolve_time',
                'outcome_source_hash', 'distillation_prompt_hash')
    if cutoff is None or not isinstance(ancestors, list) or not ancestors:
        return False
    for ancestor in ancestors:
        if not isinstance(ancestor, dict) or any(not ancestor.get(k) for k in required):
            return False
        resolved = _parse_pit_time(ancestor['resolve_time'], date_only_eod=date_only_eod)
        if resolved is None:
            return False
        try:
            if resolved >= cutoff:
                return False
        except TypeError:
            return False
    return True

class ExperienceLibrary:

    def __init__(self, store_dir: str, embedder=None):
        self.dir = Path(store_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / 'kv_experiences.json'
        self.vpath = self.dir / 'vecs.json'
        self.exp: dict = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
        if not isinstance(self.exp, dict):
            self.exp = {}
        self.vecs: dict = json.loads(self.vpath.read_text(encoding='utf-8')) if self.vpath.exists() else {}
        self._embedder = embedder

    def _get_embedder(self):
        if self._embedder is None:
            import foresight.config as cfg
            from foresight.textkg.embedder import Embedder
            self._embedder = Embedder(cfg.OPENAI_API_KEY, cfg.OPENAI_BASE_URL, cfg.EMBED_MODEL, cfg.EMBED_DIM)
        return self._embedder

    @staticmethod
    def _norm(v) -> list:
        a = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(a))
        return (a / (n + 1e-08)).tolist()

    def insert_lesson(self, name: str, description: str, domain: str, advantage: float, resolve_date: str, tags: list=None, source: str='', _save_after: bool=True):
        eid = _exp_id(name, source)
        if eid in self.exp:
            advantage = 0.9 * self.exp[eid].get('advantage', 0.0) + 0.1 * advantage
        self.exp[eid] = {'experience_id': eid, 'name': name, 'description': description, 'domain': domain, 'advantage': round(float(advantage), 4), 'tags': tags or [], 'resolve_date': resolve_date, 'source': source}
        if eid not in self.vecs:
            self.vecs[eid] = self._norm(self._get_embedder().embed_one(f'{name}. {description}'))
        if _save_after:
            self.save()

    def evoke_skill(self, name: str, description: str, domain: str, advantage: float,
                    resolve_date: str, tags: list=None, source: str='',
                    stability_hint: str='regime_dependent', merge_threshold: float=0.85,
                    provenance_ancestor: dict=None):
        new_vec = self._norm(self._get_embedder().embed_one(f'{name}. {description}'))
        qv = np.asarray(new_vec, dtype=np.float32)
        (best_sim, best_eid) = (-1.0, None)
        for (eid_existing, sk) in self.exp.items():
            if sk.get('domain') != domain:
                continue
            v = self.vecs.get(eid_existing)
            if v is None:
                continue
            sim = float(np.dot(qv, np.asarray(v, dtype=np.float32)))
            if sim > best_sim:
                (best_sim, best_eid) = (sim, eid_existing)
        if best_eid is not None and best_sim >= merge_threshold:
            si = self.exp[best_eid]
            si['advantage'] = round(0.9 * si.get('advantage', 0.0) + 0.1 * float(advantage), 4)
            si['tags'] = list(set(si.get('tags', []) + (tags or [])))
            srcs = si.get('sources')
            if not isinstance(srcs, list):
                srcs = [si.get('source', '')] if si.get('source') else []
            if source and source not in srcs:
                srcs.append(source)
            si['sources'] = srcs[-20:]
            si['provenance_ancestors'] = _union_ancestors(
                si.get('provenance_ancestors'),
                [provenance_ancestor] if provenance_ancestor else [])
            si['merge_count'] = int(si.get('merge_count', 0)) + 1
            si['stability'] = 'unevaluable'
            self.save()
            return best_eid
        eid = _exp_id(name, source)
        if eid in self.exp:
            advantage = 0.9 * self.exp[eid].get('advantage', 0.0) + 0.1 * float(advantage)
        self.exp[eid] = {'experience_id': eid, 'name': name, 'description': description,
                         'domain': domain, 'advantage': round(float(advantage), 4),
                         'tags': tags or [], 'resolve_date': resolve_date, 'source': source,
                         'sources': [source] if source else [], 'merge_count': 0,
                         'stability_hint': stability_hint, 'stability': 'unevaluable',
                         'provenance_ancestors': [dict(provenance_ancestor)] if provenance_ancestor else []}
        self.vecs[eid] = new_vec
        self.save()
        return eid

    def revise_skill(self, eid: str, advantage: float=None, description: str=None) -> bool:
        if eid not in self.exp:
            return False
        s = self.exp[eid]
        if advantage is not None:
            s['advantage'] = round(max(-1.0, min(1.0, float(advantage))), 4)
        if description is not None and description != s.get('description'):
            s['description'] = description[:500]
            try:
                self.vecs[eid] = self._norm(self._get_embedder().embed_one(f"{s['name']}. {s['description']}"))
            except Exception:
                pass
        self.save()
        return True

    def merge_skill(self, eid_i: str, eid_j: str) -> bool:
        if eid_i not in self.exp or eid_j not in self.exp or eid_i == eid_j:
            return False
        si = self.exp[eid_i]
        sj = self.exp[eid_j]
        si['description'] = (si['description'] + ' || ' + sj['description'])[:500]
        si['advantage'] = round((si['advantage'] + sj['advantage']) / 2, 4)
        si['tags'] = list(set(si.get('tags', []) + sj.get('tags', [])))
        si['provenance_ancestors'] = _union_ancestors(
            si.get('provenance_ancestors'), sj.get('provenance_ancestors'))
        si['stability'] = 'unevaluable'
        try:
            self.vecs[eid_i] = self._norm(self._get_embedder().embed_one(f"{si['name']}. {si['description']}"))
        except Exception:
            pass
        self.delete(eid_j)
        return True

    def retire_skill(self, eid: str) -> bool:
        if eid not in self.exp:
            return False
        self.delete(eid)
        return True

    def set_stability(self, eid: str, label: str) -> bool:
        assert label in ('stable', 'unstable', 'miss', 'unevaluable')
        if eid not in self.exp:
            return False
        if label == 'miss':
            self.retire_skill(eid)
            return True
        self.exp[eid]['stability'] = label
        self.save()
        return True

    def delete(self, eid: str):
        removed = self.exp.pop(eid, None) is not None
        self.vecs.pop(eid, None)
        if removed:
            self.save()

    def retrieve(self, catalyst_type: str, query: str, k: int=5,
                 before_date: str=None, min_sim: float=0.0,
                 domain_boost: float=0.15, adv_weight: bool=True,
                 query_vec=None, use_stability_weight: bool=False,
                 strict_provenance: bool=False, date_only_eod: bool=True) -> list:
        if k <= 0 or not self.exp:
            return []
        qv = np.asarray(query_vec if query_vec is not None else self._get_embedder().embed_one(query), dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-08)
        stability_map = {}
        if use_stability_weight:
            import foresight.config as cfg
            stability_map = cfg.CSM_STABILITY_WEIGHT_MAP
        scored = []
        for e in self.exp.values():
            if before_date:
                if strict_provenance:
                    if not _strict_ancestor_pit(e, before_date, date_only_eod=date_only_eod):
                        continue
                elif e.get('resolve_date', '') >= before_date:
                    continue
            v = self.vecs.get(e['experience_id'])
            sim = float(np.dot(qv, np.asarray(v, dtype=np.float32))) if v else 0.0
            base = sim + (domain_boost if e.get('domain') == catalyst_type else 0.0)
            adv = e.get('advantage', 0.0)
            score = base * (1 + (adv + 1) / 2) if adv_weight else base
            if use_stability_weight:
                stability = e.get('stability', 'unevaluable')
                score *= stability_map.get(stability, 0.3)
            scored.append((sim, score, e))
        scored.sort(key=lambda t: -t[1])
        return [e for (sim, _sc, e) in scored[:k] if sim >= min_sim]

    def update_advantage(self, eid: str, signal: float, eta: float=0.1):
        if eid in self.exp:
            old = self.exp[eid].get('advantage', 0.0)
            self.exp[eid]['advantage'] = round(max(-1.0, min(1.0, (1 - eta) * old + eta * signal)), 4)
            self.save()

    def prune(self, min_advantage: float=-0.5):
        rm = [eid for (eid, e) in self.exp.items() if e.get('advantage', 0.0) < min_advantage]
        for eid in rm:
            del self.exp[eid]
            self.vecs.pop(eid, None)
        if rm:
            self.save()
        return rm

    def save(self):
        self.vecs = {k: v for (k, v) in self.vecs.items() if k in self.exp}
        self._atomic_write(self.path, json.dumps(self.exp, ensure_ascii=False, indent=1))
        self._atomic_write(self.vpath, json.dumps(self.vecs, ensure_ascii=False))

    @staticmethod
    def _atomic_write(path: Path, content: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        (fd, tmp) = tempfile.mkstemp(dir=str(path.parent), prefix='.tmp_', suffix=path.suffix)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(content)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def __len__(self):
        return len(self.exp)
