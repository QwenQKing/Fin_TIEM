from __future__ import annotations
import json, hashlib, threading
from pathlib import Path

class Cache:

    def __init__(self, store_dir: str):
        self.dir = Path(store_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / 'cache.json'
        self.d = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
        self._lock = threading.Lock()

    @staticmethod
    def key(query: str, cutoff: str='', channels=None) -> str:
        raw = f"{query}|{cutoff}|{','.join(channels or [])}"
        return hashlib.md5(raw.encode('utf-8')).hexdigest()

    def get(self, key: str, default=None):
        return self.d.get(key, default)

    def set(self, key: str, value):
        with self._lock:
            self.d[key] = value
            self.path.write_text(json.dumps(self.d, ensure_ascii=False), encoding='utf-8')

    def save(self):
        with self._lock:
            self.path.write_text(json.dumps(self.d, ensure_ascii=False), encoding='utf-8')

    def __len__(self):
        return len(self.d)