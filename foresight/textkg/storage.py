from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import networkx as nx
import numpy as np
logger = logging.getLogger(__name__)

class JsonKVStorage:

    def __init__(self, file_path: str):
        self._path = Path(file_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            try:
                self._data: Dict[str, Any] = json.loads(self._path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                logger.warning('Corrupt KV file %s – starting fresh.', self._path)
                self._data = {}
        else:
            self._data = {}

    def get(self, key: str, default: Any=None) -> Any:
        return self._data.get(key, default)

    def all(self) -> Dict[str, Any]:
        return dict(self._data)

    def filter_new(self, keys: List[str]) -> List[str]:
        return [k for k in keys if k not in self._data]

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def upsert(self, items: Dict[str, Any]) -> None:
        self._data.update(items)
        self.save()

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def save(self) -> None:
        self._path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding='utf-8')

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

class VectorStorage:

    def __init__(self, storage_file: str, embedding_dim: int):
        from nano_vectordb import NanoVectorDB
        self._path = Path(storage_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._dim = embedding_dim
        self._db = NanoVectorDB(embedding_dim, storage_file=str(self._path))
        _s = self._db._NanoVectorDB__storage
        if _s['matrix'].ndim < 2:
            _s['matrix'] = np.zeros((len(_s.get('data', [])), embedding_dim), dtype=np.float32)

    def upsert(self, id: str, vector: List[float], meta: Dict[str, Any]) -> None:
        data = {'__id__': id, **meta}
        self._db.upsert([{'__id__': id, '__vector__': np.array(vector, dtype=np.float32), **meta}])

    def upsert_batch(self, entries: List[Dict[str, Any]]) -> None:
        records = []
        for e in entries:
            rec = {'__id__': e['id'], '__vector__': np.array(e['vector'], dtype=np.float32)}
            rec.update(e.get('meta', {}))
            records.append(rec)
        if records:
            self._db.upsert(records)

    def search(self, query_vec: List[float], top_k: int=10) -> List[Dict[str, Any]]:
        q = np.array(query_vec, dtype=np.float32)
        results = self._db.query(q, top_k=top_k)
        return results

    def save(self) -> None:
        self._db.save()

    def __len__(self) -> int:
        try:
            return len(self._db._NanoVectorDB__storage['data'])
        except Exception:
            return 0

class GraphStorage:

    def __init__(self, storage_file: str):
        self._path = Path(storage_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            try:
                self._graph: nx.DiGraph = nx.read_graphml(str(self._path))
                logger.info('Loaded graph from %s (%d nodes, %d edges)', self._path, self._graph.number_of_nodes(), self._graph.number_of_edges())
            except Exception as exc:
                logger.warning('Could not load graph %s: %s – starting fresh.', self._path, exc)
                self._graph = nx.DiGraph()
        else:
            self._graph = nx.DiGraph()

    def upsert_node(self, node_id: str, attrs: Dict[str, Any]) -> None:
        from foresight.textkg.text_builder import GRAPH_FIELD_SEP
        if self._graph.has_node(node_id):
            existing = dict(self._graph.nodes[node_id])
            if 'source_id' in attrs and 'source_id' in existing:
                existing_ids = set(existing['source_id'].split(GRAPH_FIELD_SEP))
                new_ids = set(attrs['source_id'].split(GRAPH_FIELD_SEP))
                merged = existing_ids | new_ids
                attrs = {**attrs, 'source_id': GRAPH_FIELD_SEP.join(sorted(merged))}
            self._graph.nodes[node_id].update(attrs)
        else:
            self._graph.add_node(node_id, **attrs)

    def upsert_edge(self, src: str, dst: str, attrs: Dict[str, Any] | None=None) -> None:
        self._graph.add_edge(src, dst, **attrs or {})

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if self._graph.has_node(node_id):
            return dict(self._graph.nodes[node_id])
        return None

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    def get_edges(self, node_id: str) -> List[tuple]:
        edges = []
        for (src, dst, data) in self._graph.out_edges(node_id, data=True):
            edges.append((src, dst, data))
        for (src, dst, data) in self._graph.in_edges(node_id, data=True):
            edges.append((src, dst, data))
        return edges

    def neighbors(self, node_id: str) -> List[str]:
        return list(self._graph.successors(node_id))

    def predecessors(self, node_id: str) -> List[str]:
        return list(self._graph.predecessors(node_id))

    def nodes_by_role(self, role: str) -> List[str]:
        return [n for (n, d) in self._graph.nodes(data=True) if d.get('role') == role]

    def all_nodes(self) -> List[tuple]:
        return [(n, dict(d)) for (n, d) in self._graph.nodes(data=True)]

    def all_edges(self) -> List[tuple]:
        return [(s, t, dict(d)) for (s, t, d) in self._graph.edges(data=True)]

    def save(self) -> None:
        nx.write_graphml(self._graph, str(self._path))
        logger.debug('Graph saved to %s', self._path)

    def number_of_nodes(self) -> int:
        return self._graph.number_of_nodes()

    def number_of_edges(self) -> int:
        return self._graph.number_of_edges()

    def degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)