from __future__ import annotations
import json
from pathlib import Path

class SampleStore:

    def __init__(self, store_dir: str):
        self.dir = Path(store_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        if (self.dir / 'stocks.json').exists():
            self._load()

    def build(self, data: dict):
        for name in ('meta', 'stocks'):
            (self.dir / f'{name}.json').write_text(json.dumps(data.get(name, {}), ensure_ascii=False), encoding='utf-8')
        self._load()
        return {'stocks': len(self.stocks), 'catalysts': len(self.catalysts)}

    def _load(self):
        self.stocks = json.loads((self.dir / 'stocks.json').read_text(encoding='utf-8'))
        cp = self.dir / 'catalysts.json'
        self.catalysts = json.loads(cp.read_text(encoding='utf-8')) if cp.exists() else []

    def name(self, stock: str) -> str:
        return self.stocks.get(stock, {}).get('name', stock)