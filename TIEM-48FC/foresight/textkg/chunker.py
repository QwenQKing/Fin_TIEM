from __future__ import annotations
import logging
from typing import Dict, List
import tiktoken
logger = logging.getLogger(__name__)
_ENCODERS: Dict[str, tiktoken.Encoding] = {}

def _get_encoder(model: str) -> tiktoken.Encoding:
    if model not in _ENCODERS:
        try:
            _ENCODERS[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _ENCODERS[model] = tiktoken.get_encoding('cl100k_base')
    return _ENCODERS[model]

def chunk_text(text: str, max_tokens: int=512, overlap: int=64, model: str='gpt-4o') -> List[Dict]:
    enc = _get_encoder(model)
    tokens = enc.encode(text)
    if not tokens:
        return []
    chunks = []
    start = 0
    chunk_index = 0
    step = max(1, max_tokens - overlap)
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens)
        chunks.append({'content': chunk_text_str, 'tokens': len(chunk_tokens), 'chunk_index': chunk_index})
        if end == len(tokens):
            break
        start += step
        chunk_index += 1
    return chunks