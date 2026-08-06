from __future__ import annotations
import time
import logging
logger = logging.getLogger(__name__)

def with_retry(fn, max_retry: int=5, backoff: float=1.5, what: str='call'):
    last = None
    for attempt in range(max_retry + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt < max_retry:
                wait = backoff ** attempt
                logger.warning('%s failed (attempt %d/%d/%.1fsretry, : %s', what, attempt + 1, max_retry, wait, type(e).__name__)
                time.sleep(wait)
            else:
                logger.error('%s retry %d  exhausted: %s', what, max_retry, e)
                raise
    raise last