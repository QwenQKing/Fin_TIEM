import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

def main():
    print('build …')
    from foresight.build_databases import main as build_main
    build_main()
if __name__ == '__main__':
    main()