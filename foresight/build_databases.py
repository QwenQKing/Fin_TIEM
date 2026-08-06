import sys, json
from pathlib import Path
BASE = Path(__file__).parent.parent

def _arg_val(flag: str, default: str) -> str:
    a = sys.argv
    if flag in a and a.index(flag) + 1 < len(a):
        return a[a.index(flag) + 1]
    return default
_data = Path(_arg_val('--data', str(BASE / 'data-all/Astock/Astock.json')))
DATA = _data if _data.is_absolute() else BASE / _data
WITH_TEXTKG = '--no-textkg' not in sys.argv

def main():
    data = json.loads(DATA.read_text(encoding='utf-8'))
    dataset = data.get('meta', {}).get('dataset', 'dataset')
    DB = BASE / 'databases' / dataset
    DB.mkdir(parents=True, exist_ok=True)
    print(f"load {DATA.name}: {len(data.get('catalysts', []))} events / {len(data.get('text', []))} news  → libdir databases/{dataset}/")
    from foresight.stores.sample_store import SampleStore
    from foresight.stores.experience import ExperienceLibrary
    from foresight.stores.cache import Cache
    print('\n🗂️ base info store (stocks+catalysts) …')
    print('  ', SampleStore(str(DB / '01_info')).build(data))
    print('🧠 experience store (empty shell) …')
    exp = ExperienceLibrary(str(DB / '03_experience'))
    print('   experience count:', len(exp))
    print('💾 cache (empty) …')
    print('   cache:', len(Cache(str(DB / '04_cache'))))
    text_events = data.get('text') or []
    if WITH_TEXTKG and text_events:
        print(f'\nevents hypergraph (LLM): {len(text_events)} ...')
        from foresight.stores.text_kg import TextKG
        print('  ', TextKG(str(DB / '02_event')).build(text_events))
    elif WITH_TEXTKG:
        print('\nevents hypergraph: skip (no text)')
    else:
        print('\nevents hypergraph: skip (--no-textkg)')
    print(f'\ndone. databases/{dataset}/ :')
    for p in sorted(DB.glob('*')):
        print('  ', p.name, '->', [f.name for f in p.glob('*')])
if __name__ == '__main__':
    main()