from __future__ import annotations

import hashlib
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import foresight.config as cfg
from foresight.retry import with_retry
from foresight.textkg.extractor import _clean_json, _repair_json


STABILITY_PROMPT = """Evaluate whether a case-derived forecasting skill generalizes across events.

Skill:
  name: {skill_name}
  description: {skill_description}
  domain: {skill_domain}

Candidate events (each \"[id] (stock) catalyst_type -> realized outcome\"):
{test_block}

For each event, return exactly one verdict:
  APPLY: the skill is relevant and its implied direction matches the outcome
  MISAPPLY: the skill is relevant but its implied direction conflicts with the outcome
  SKIP: the skill is not relevant to this event

Return JSON only:
{{"results": [{{"catalyst_id": "<id>", "verdict": "APPLY|MISAPPLY|SKIP"}}, ...]}}"""


class StabilityEvaluator:
    def __init__(self, exp_lib, catalysts: list[dict], K: int=None, seed: int=None):
        self.exp = exp_lib
        self.catalysts = catalysts
        self.K = cfg.CSM_STABILITY_K if K is None else K
        self.seed = cfg.CSM_STABILITY_SEED if seed is None else seed

    @staticmethod
    def _test_block(sampled: list[dict]) -> str:
        lines = []
        for catalyst in sampled:
            lines.append(
                f"  [{catalyst.get('catalyst_id', '?')}] "
                f"({catalyst.get('stock_id', '?')}) "
                f"{catalyst.get('catalyst_type', '?')} -> "
                f"{catalyst.get('outcome', '?')}")
        return '\n'.join(lines)

    def _call_llm(self, prompt: str, scope: str,
                  max_tokens: int=1500) -> Optional[dict]:
        from foresight.reason_loop import _chat
        try:
            response = with_retry(
                lambda: _chat([{'role': 'user', 'content': prompt}], max_tokens),
                max_retry=cfg.LLM_MAX_RETRY,
                backoff=cfg.LLM_RETRY_BACKOFF,
                what=scope)
            raw = response.choices[0].message.content or ''
            return json.loads(_repair_json(_clean_json(raw)))
        except Exception as exc:
            print(f'stability evaluation failed: {type(exc).__name__}: {str(exc)[:80]}',
                  file=sys.stderr)
            return None

    def evaluate_skill(self, skill: dict) -> dict:
        skill_seed = int(hashlib.md5(
            skill.get('experience_id', '').encode('utf-8')).hexdigest()[:8], 16)
        rng = random.Random(self.seed ^ skill_seed)
        k_eff = min(self.K, len(self.catalysts))
        if k_eff < 5:
            return {'apply_rate': 0.0, 'label': 'unevaluable',
                    'reason': f'K_eff={k_eff}<5'}
        sampled = rng.sample(self.catalysts, k_eff)
        prompt = STABILITY_PROMPT.format(
            skill_name=skill.get('name', ''),
            skill_description=skill.get('description', ''),
            skill_domain=skill.get('domain', ''),
            test_block=self._test_block(sampled))
        result = self._call_llm(prompt, 'csm_stability')
        if not isinstance(result, dict) or 'results' not in result:
            return {'apply_rate': 0.0, 'label': 'unevaluable',
                    'reason': 'llm_fail'}

        verdicts = {}
        for item in result.get('results', []):
            verdict = item.get('verdict')
            if verdict in ('APPLY', 'MISAPPLY', 'SKIP'):
                verdicts[item.get('catalyst_id')] = verdict

        n_apply = n_misapply = n_skip = 0
        for catalyst in sampled:
            verdict = verdicts.get(catalyst.get('catalyst_id'), 'SKIP')
            if verdict == 'APPLY':
                n_apply += 1
            elif verdict == 'MISAPPLY':
                n_misapply += 1
            else:
                n_skip += 1

        counts = {'n_apply': n_apply, 'n_misapply': n_misapply,
                  'n_skip': n_skip}
        if n_skip >= k_eff * cfg.CSM_STABILITY_SKIP_RATIO_LIMIT:
            return {'apply_rate': 0.0, 'label': 'unevaluable', **counts,
                    'reason': f'skip_too_high({n_skip}/{k_eff})'}
        relevant = n_apply + n_misapply
        if relevant == 0:
            return {'apply_rate': 0.0, 'label': 'unevaluable', **counts,
                    'reason': 'no_apply_misapply'}

        apply_rate = n_apply / relevant
        if apply_rate >= cfg.CSM_STABILITY_STABLE_THRESHOLD:
            label = 'stable'
        elif apply_rate <= cfg.CSM_STABILITY_MISS_THRESHOLD:
            label = 'miss'
        else:
            label = 'unstable'
        return {'apply_rate': apply_rate, 'label': label, **counts}

    def evaluate_all(self, force: bool=False, max_skills: int=None) -> dict:
        skill_ids = list(self.exp.exp)
        if max_skills is not None:
            skill_ids = skill_ids[:max_skills]
        summary = {'stable': 0, 'unstable': 0, 'miss': 0, 'unevaluable': 0}
        to_evaluate = []
        for skill_id in skill_ids:
            label = self.exp.exp[skill_id].get('stability', 'unevaluable')
            if not force and label != 'unevaluable':
                summary[label] = summary.get(label, 0) + 1
            else:
                to_evaluate.append(skill_id)

        details = []
        lock = threading.Lock()

        def evaluate_one(skill_id):
            return skill_id, self.evaluate_skill(self.exp.exp[skill_id])

        workers = min(cfg.LLM_WORKERS, max(1, len(to_evaluate)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(evaluate_one, sid) for sid in to_evaluate]
            for future in as_completed(futures):
                skill_id, result = future.result()
                label = result['label']
                with lock:
                    summary[label] = summary.get(label, 0) + 1
                    details.append({'experience_id': skill_id, **result})
                self.exp.set_stability(skill_id, label)
        return {'summary': summary, 'details': details}
