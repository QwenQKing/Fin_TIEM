from __future__ import annotations
import os
from openai import OpenAI
import foresight.config as cfg


def _no_proxy_http():
    import httpx
    for k in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY'):
        os.environ.pop(k, None)
    return httpx.Client(trust_env=False, verify=True)


def chat_client(model: str | None = None):
    model = model or cfg.LLM_MODEL
    prov = cfg.LLM_PROVIDER
    if prov == 'litellm':
        key = os.getenv('LITELLM_API_KEY', '')
        if not key:
            raise SystemExit('LLM_PROVIDER=litellm requires LITELLM_API_KEY')
        return OpenAI(api_key=key, base_url=cfg.LITELLM_BASE, timeout=60.0, http_client=_no_proxy_http())
    return OpenAI(api_key=cfg.OPENAI_API_KEY, base_url=cfg.OPENAI_BASE_URL)


def embed_client():
    return OpenAI(api_key=cfg.OPENAI_API_KEY, base_url=cfg.OPENAI_BASE_URL)
