from __future__ import annotations
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List
from openai import OpenAI
import foresight.config as cfg
from foresight.retry import with_retry
logger = logging.getLogger(__name__)
_MAX_WORKERS = cfg.LLM_WORKERS
REL_TYPES = {'supplier_of', 'customer_of', 'subsidiary_of', 'invests_in', 'competes_with', 'regulates', 'rates', 'partners_with', 'affects', 'belongs_to_sector', 'related'}
EXTRACT_PROMPT = 'Extract financial facts (propositions) and their entities from the financial text (news / announcements) below, to build an event knowledge hypergraph.\nFor each atomic fact, list the entities involved and, if stated, the relation between them.\nEntity types: company (listed firm; keep ticker), subsidiary, person (executive/analyst), regulator (securities regulator / exchange, e.g. SEC/CSRC), product (product/business line), sector, financial_metric (revenue/profit/margin/etc), event (earnings/M&A/regulatory/rating/index/contract), institution (brokerage/fund), other.\nPick the MOST SPECIFIC entity type — use "other" ONLY when none of the above fits.\nFor relations, pick the MOST SPECIFIC type — use "affects" only for a general causal influence, and "related" only as a last resort.\n\nReturn JSON ONLY (no markdown, no explanation):\n{{\n  "propositions": [\n    {{\n      "sentence": "atomic fact or claim",\n      "entities": [\n        {{"name": "<entity name; keep stock ticker if present>", "type": "<one of the types above>"}}\n      ],\n      "relations": [\n        {{"src": "<entity name>", "dst": "<entity name>", "type": "<MUST be exactly one of: supplier_of, customer_of, subsidiary_of, invests_in, competes_with, regulates, rates, partners_with, affects, belongs_to_sector, related>"}}\n      ]\n    }}\n  ]\n}}\nText: {text}'

def _clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub('^```(?:json)?\\s*', '', raw)
    raw = re.sub('\\s*```$', '', raw)
    return raw.strip()

def _repair_json(s: str) -> str:
    s = re.sub('//[^\\n]*', '', s)
    s = re.sub('/\\*.*?\\*/', '', s, flags=re.DOTALL)
    s = s.replace('True', 'true').replace('False', 'false').replace('None', 'null')
    s = re.sub(',(\\s*[}\\]])', '\\1', s)
    return s

class LLMExtractor:

    def __init__(self, api_key: str, base_url: str, model: str, cache_path: str=None):
        from foresight.llm import chat_client
        self._client = chat_client(model)
        self._model = model
        self._cache_path = cache_path
        self._cache = {}
        if cache_path:
            from pathlib import Path as _P
            p = _P(cache_path)
            if p.exists():
                try:
                    self._cache = json.loads(p.read_text(encoding='utf-8'))
                except Exception:
                    self._cache = {}

    def _save_cache(self):
        if self._cache_path:
            from pathlib import Path as _P
            _P(self._cache_path).write_text(json.dumps(self._cache, ensure_ascii=False), encoding='utf-8')

    def _extract_one(self, text: str) -> List[Dict[str, Any]]:
        prompt = EXTRACT_PROMPT.format(text=text)
        ckey = __import__('hashlib').md5((self._model + '\x00' + prompt).encode('utf-8')).hexdigest()
        if ckey in self._cache:
            return self._cache[ckey]
        try:
            resp = with_retry(lambda : self._client.chat.completions.create(model=self._model, messages=[{'role': 'user', 'content': prompt}], temperature=0, max_tokens=cfg.KB_EXTRACT_MAX_TOKENS), max_retry=cfg.LLM_MAX_RETRY, backoff=cfg.LLM_RETRY_BACKOFF, what='extract')
            raw = resp.choices[0].message.content or ''
            raw = _clean_json(raw)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = json.loads(_repair_json(raw))
            propositions = data.get('propositions', [])
            validated = []
            for p in propositions:
                if not isinstance(p, dict) or 'sentence' not in p:
                    continue
                raw_entities = p.get('entities', [])
                entities: List[Dict[str, str]] = []
                for e in raw_entities:
                    if isinstance(e, dict):
                        name = str(e.get('name', '')).strip()
                        etype = str(e.get('type', '')).strip() or 'other'
                    else:
                        name = str(e).strip()
                        etype = 'other'
                    if name:
                        entities.append({'name': name, 'type': etype})
                rels: List[Dict[str, str]] = []
                for r in p.get('relations', []):
                    if isinstance(r, dict) and str(r.get('src', '')).strip() and str(r.get('dst', '')).strip():
                        rt = str(r.get('type', '')).strip().lower()
                        if rt not in REL_TYPES:
                            rt = 'related'
                        rels.append({'src': str(r['src']).strip(), 'dst': str(r['dst']).strip(), 'type': rt})
                validated.append({'sentence': str(p['sentence']).strip(), 'entities': entities, 'relations': rels})
            self._cache[ckey] = validated
            return validated
        except json.JSONDecodeError as exc:
            logger.warning('JSON parse error in extractor: %s', exc)
            return []
        except Exception as exc:
            logger.error('LLMExtractor error: %s', exc)
            return []

    def extract_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        if not texts:
            return []
        results: List[List[Dict[str, Any]]] = [[] for _ in texts]
        workers = min(len(texts), _MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=workers) as exe:
            future_to_idx = {exe.submit(self._extract_one, t): i for (i, t) in enumerate(texts)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error('extract_batch worker %d failed: %s', idx, exc)
        self._save_cache()
        return results