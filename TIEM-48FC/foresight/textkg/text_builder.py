from __future__ import annotations
import hashlib
import logging
from typing import Any, Dict, List, Set
from foresight.textkg.chunker import chunk_text
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it=None, **k):
        return it if it is not None else []
logger = logging.getLogger(__name__)
GRAPH_FIELD_SEP = '<SEP>'

def _md5(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def _chunk_id(source: str, chunk_index: int) -> str:
    return f'{_md5(source)}_{chunk_index}'

def _entity_node_id(entity_name: str, source: str) -> str:
    return f'{entity_name.upper()} | Source: {source}'

def _hyperedge_node_id(sentence: str, source: str) -> str:
    return f'<hyperedge>{sentence} | Source: {source}'

def _upsert_text_vdb(vdb, texts: List[str], metas: List[Dict[str, Any]], embedder, key: str) -> int:
    if not texts:
        return 0
    vectors = embedder.embed(texts)
    (entries, seen) = ([], set())
    for (vec, meta, txt) in zip(vectors, metas, texts):
        vid = meta['id']
        if vid in seen:
            continue
        seen.add(vid)
        m = {'content': txt, key: meta['node_id'], 'source': meta['source']}
        if meta.get('etype'):
            m['etype'] = meta['etype']
        if meta.get('ts'):
            m['ts'] = meta['ts']
        if meta.get('stock'):
            m['stock'] = meta['stock']
        entries.append({'id': vid, 'vector': vec, 'meta': m})
    vdb.upsert_batch(entries)
    return len(entries)

def build_from_text_events(events: List[Dict[str, Any]], kv, vdb_entity, vdb_hyperedge, graph, extractor, embedder, max_tokens: int=512, overlap: int=64, extract_batch_size: int=5, model: str='gpt-4o') -> Dict[str, int]:
    chunk_recs: List[Dict[str, Any]] = []
    for ev in events:
        src = ev['doc_id']
        ts = ev.get('ts', '')
        stock = (ev.get('mentioned_stocks') or [''])[0]
        title = (ev.get('title') or '').strip()
        body = (ev.get('body') or '').strip()
        text = (title if title == body or not body else f'{title}. {body}').strip('. ')
        for ch in chunk_text(text, max_tokens=max_tokens, overlap=overlap, model=model):
            cid = _chunk_id(src, ch['chunk_index'])
            rec = {'content': ch['content'], 'chunk_id': cid, 'source': src, 'ts': ts, 'stock': stock}
            kv.set(cid, rec)
            chunk_recs.append(rec)
    all_props: List[Dict[str, Any]] = []
    texts = [c['content'] for c in chunk_recs]
    n_batches = (len(texts) + extract_batch_size - 1) // max(1, extract_batch_size)
    for i in tqdm(range(0, len(texts), extract_batch_size), total=n_batches, desc='🕸️ LLM edge', unit='batch'):
        batch = texts[i:i + extract_batch_size]
        recs = chunk_recs[i:i + extract_batch_size]
        for (rec, props) in zip(recs, extractor.extract_batch(batch)):
            for p in props:
                all_props.append({'sentence': p['sentence'], 'entities': p['entities'], 'relations': p.get('relations', []), 'chunk_id': rec['chunk_id'], 'source': rec['source'], 'ts': rec['ts'], 'stock': rec['stock']})
    (ent_texts, ent_metas, he_texts, he_metas) = ([], [], [], [])
    for prop in all_props:
        src_doc = prop['source']
        he_id = _hyperedge_node_id(prop['sentence'], src_doc)
        graph.upsert_node(he_id, {'role': 'hyperedge', 'source_id': prop['chunk_id'], 'source': src_doc, 'ts': prop['ts'], 'stock': prop['stock']})
        he_vid = _md5(he_id)
        he_texts.append(prop['sentence'])
        he_metas.append({'id': he_vid, 'node_id': he_id, 'source': src_doc, 'ts': prop['ts'], 'stock': prop['stock']})
        for e in prop['entities']:
            if isinstance(e, dict):
                name = str(e.get('name', '')).strip()
                etype = str(e.get('type', '')).strip() or 'other'
            else:
                name = str(e).strip()
                etype = 'other'
            if not name:
                continue
            ent_id = _entity_node_id(name, src_doc)
            graph.upsert_node(ent_id, {'role': 'entity', 'source_id': prop['chunk_id'], 'source': src_doc, 'etype': etype, 'stock': prop['stock']})
            graph.upsert_edge(he_id, ent_id)
            ent_texts.append(name)
            ent_metas.append({'id': _md5(ent_id), 'node_id': ent_id, 'source': src_doc, 'etype': etype})
        for r in prop.get('relations', []):
            s_id = _entity_node_id(r['src'], src_doc)
            d_id = _entity_node_id(r['dst'], src_doc)
            if graph.has_node(s_id) and graph.has_node(d_id):
                graph.upsert_edge(s_id, d_id, {'role': 'relation', 'rel_type': r.get('type', 'related'), 'ts': prop['ts']})
    ne = _upsert_text_vdb(vdb_entity, ent_texts, ent_metas, embedder, key='entity_name')
    nh = _upsert_text_vdb(vdb_hyperedge, he_texts, he_metas, embedder, key='hyperedge_name')
    kv.save()
    vdb_entity.save()
    vdb_hyperedge.save()
    graph.save()
    return {'chunks': len(chunk_recs), 'props': len(all_props), 'entities': ne, 'hyperedges': nh}