import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from foresight.reason_loop import main
if __name__ == '__main__':
    main()