from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from openai import OpenAI
from tqdm import tqdm
import foresight.config as cfg
from foresight.retry import with_retry
logger = logging.getLogger(__name__)
_BATCH_SIZE = 100
_MAX_WORKERS = cfg.EMBED_WORKERS

class Embedder:

    def __init__(self, api_key: str, base_url: str, model: str, dim: int):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dim = dim

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        batch = [t if t.strip() else ' ' for t in batch]
        response = with_retry(lambda : self._client.embeddings.create(model=self._model, input=batch), max_retry=cfg.LLM_MAX_RETRY, backoff=cfg.LLM_RETRY_BACKOFF, what='embedding')
        return [d.embedding for d in sorted(response.data, key=lambda d: d.index)]

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        batches = [texts[i:i + _BATCH_SIZE] for i in range(0, len(texts), _BATCH_SIZE)]
        batch_results: List[List[List[float]]] = [None] * len(batches)
        workers = min(len(batches), _MAX_WORKERS)
        show_pbar = len(batches) > 1 and len(texts) > 200
        with ThreadPoolExecutor(max_workers=workers) as exe:
            future_to_idx = {exe.submit(self._embed_batch, b): i for (i, b) in enumerate(batches)}
            it = as_completed(future_to_idx)
            if show_pbar:
                it = tqdm(it, total=len(batches), desc=f'🔢 embed {len(texts)} texts', unit='batch')
            for future in it:
                idx = future_to_idx[future]
                batch_results[idx] = future.result()
        return [vec for batch in batch_results for vec in batch]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]