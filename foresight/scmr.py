from __future__ import annotations
import hashlib
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import foresight.config as cfg
from foresight.retry import with_retry
from foresight.textkg.extractor import _clean_json, _repair_json
from foresight.scmr_prompts import EVIDENCE_TYPES_4, HYPOTHESIS_EXTRACT_PROMPT, PATH_REASONING_PROMPT

def _md5(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()

class SCMRunner:

    def __init__(self, text_kg, exp_lib, cache, k: int=None, fallback_to_single: bool=None, vote_uniform: bool=False):
        self.tk = text_kg
        self.exp_lib = exp_lib
        self.cache = cache
        self.k = k or cfg.SCMR_K_PATHS
        self.fallback_to_single = fallback_to_single if fallback_to_single is not None else cfg.SCMR_FALLBACK_TO_SINGLE
        self.vote_uniform = vote_uniform

    def _call_llm(self, prompt: str, scope: str, max_tokens: int=600) -> Optional[dict]:
        ckey = self.cache.key(prompt + '\x00' + cfg.PROMPT_VERSION, '', [scope])
        cached = self.cache.get(ckey)
        if cached is not None:
            try:
                return json.loads(cached) if isinstance(cached, str) else cached
            except Exception:
                pass
        from foresight.reason_loop import _chat
        try:
            r = with_retry(lambda : _chat([{'role': 'user', 'content': prompt}], max_tokens), max_retry=cfg.LLM_MAX_RETRY, backoff=cfg.LLM_RETRY_BACKOFF, what=scope)
            raw = r.choices[0].message.content or ''
            data = json.loads(_repair_json(_clean_json(raw)))
            self.cache.set(ckey, data)
            return data
        except Exception as e:
            print(f'  ⚠ scmr {scope} fail: {type(e).__name__}: {str(e)[:60]}', file=sys.stderr)
            return None

    def _extract_hypothesis(self, skill: dict, horizon: int) -> Optional[dict]:
        prompt = HYPOTHESIS_EXTRACT_PROMPT.format(skill_name=skill.get('name', ''), skill_description=skill.get('description', ''), skill_domain=skill.get('domain', ''), horizon=horizon)
        result = self._call_llm(prompt, 'scmr_hypothesis', max_tokens=400)
        if not isinstance(result, dict):
            return None
        predicate = (result.get('predicate') or '')[:200]
        target = result.get('target_evidence_types') or []
        if not isinstance(target, list):
            target = ['focal', 'prior_event']
        target = [t for t in target if t in EVIDENCE_TYPES_4]
        if not target:
            target = ['focal', 'prior_event']
        expected = result.get('expected_direction', 'positive')
        if expected not in ('positive', 'negative'):
            expected = 'positive'
        return {'predicate': predicate, 'target_evidence_types': target, 'expected_direction': expected}

    def _retrieve_for_path(self, stock: str, T: str, query: str, target_types: list) -> dict:
        full = self.tk.retrieve_c2f(stock, T, query)
        kept = {'focal': full.get('focal', {}) if 'focal' in target_types else None, 'l2': full.get('l2', []) if 'episode' in target_types else [], 'l3': full.get('l3', []) if 'theme' in target_types else [], 'l1': full.get('l1', []) if 'prior_event' in target_types else []}
        return kept

    def _evidence_block(self, ev: dict) -> str:
        lines = []
        focal = ev.get('focal') or {}
        if focal and focal.get('text'):
            lines.append(f"  [focal] {focal.get('text', '')[:200]}")
        for ep in (ev.get('l2') or [])[:3]:
            lines.append(f"  [episode {ep.get('type')}] {ep.get('summary', '')[:120]}")
        for th in (ev.get('l3') or [])[:2]:
            lines.append(f"  [theme {th.get('type')}] {th.get('name', '')}: {th.get('description', '')[:120]}")
        for f in (ev.get('l1') or [])[:3]:
            lines.append(f"  [prior {f.get('ts', '')}] {f.get('fact', '')[:120]}")
        return '\n'.join(lines) if lines else '  (no evidence retrieved)'

    def _reason_one_path(self, skill: dict, hypothesis: dict, evidence: dict, event_text: str, horizon: int) -> Optional[dict]:
        prompt = PATH_REASONING_PROMPT.format(skill_name=skill.get('name', ''), skill_description=skill.get('description', '')[:300], hypothesis_predicate=hypothesis['predicate'], expected_direction=hypothesis['expected_direction'], event_text=event_text[:600] or '(no event content)', evidence_block=self._evidence_block(evidence), horizon=horizon)
        result = self._call_llm(prompt, 'scmr_reason', max_tokens=600)
        if not isinstance(result, dict):
            return None
        y_i = result.get('y_i', '')
        if y_i == 'abstain':
            return None
        if y_i not in ('positive', 'negative'):
            return None
        support_i = float(result.get('support_i', 0.5) or 0.5)
        support_i = max(0.0, min(1.0, support_i))
        return {'y_i': y_i, 'support_i': support_i, 'rationale_i': (result.get('rationale_i') or '')[:300]}

    def predict(self, ev: dict, x_event: str, exps: list[dict], focal_evidence: dict) -> Optional[dict]:
        if not exps:
            return None
        k_eff = min(self.k, len(exps))
        skills_to_use = exps[:k_eff]
        stock = ev.get('stock_id', '')
        T = ev.get('decision_time', '')
        horizon = int(ev.get('horizon_days', ev.get('horizon', 5)) or 5)
        query = (x_event or ev.get('catalyst_type', ''))[:500]

        def _run_one_path(skill):
            hypothesis = self._extract_hypothesis(skill, horizon)
            if not hypothesis:
                return None
            evidence = self._retrieve_for_path(stock, T, query, hypothesis['target_evidence_types'])
            r = self._reason_one_path(skill, hypothesis, evidence, x_event, horizon)
            if r is None:
                return None
            stability = skill.get('stability', 'unevaluable')
            stability_w = cfg.CSM_STABILITY_WEIGHT_MAP.get(stability, 0.3)
            w_i = 1.0 if self.vote_uniform else stability_w * r['support_i']
            return {'skill_id': skill.get('experience_id', ''), 'skill_name': skill.get('name', ''), 'hypothesis': hypothesis, 'evidence_ids': {'focal_doc_ids': (evidence.get('focal') or {}).get('doc_ids', []) if evidence.get('focal') else [], 'l2': [e.get('episode_id') for e in evidence.get('l2') or []], 'l3': [t.get('theme_id') for t in evidence.get('l3') or []]}, 'y_i': r['y_i'], 'support_i': r['support_i'], 'stability': stability, 'w_i': w_i, 'rationale_i': r['rationale_i']}
        paths = []
        with ThreadPoolExecutor(max_workers=min(k_eff, 8)) as exe:
            futs = [exe.submit(_run_one_path, skill) for skill in skills_to_use]
            for fut in as_completed(futs):
                try:
                    p = fut.result()
                    if p is not None:
                        paths.append(p)
                except Exception as e:
                    print(f'  ⚠ scmr path fail: {type(e).__name__}: {str(e)[:60]}', file=sys.stderr)
        if not paths:
            return None
        votes = {'positive': 0.0, 'negative': 0.0}
        for p in paths:
            votes[p['y_i']] += p['w_i']
        if votes['positive'] > votes['negative']:
            y_final = 'positive'
        elif votes['negative'] > votes['positive']:
            y_final = 'negative'
        else:
            seed_int = int(hashlib.md5(ev.get('catalyst_id', '').encode()).hexdigest()[:8], 16)
            y_final = random.Random(cfg.GLOBAL_SEED ^ seed_int).choice(['positive', 'negative'])
        total_w = votes['positive'] + votes['negative'] + 1e-08
        conf_final = votes[y_final] / total_w
        return {'pred_direction': y_final, 'conf': float(conf_final), 'trail': paths[:3], 'n_paths': len(paths), 'n_skills_used': k_eff}
