import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from foresight.stores.csl_stability import StabilityEvaluator
from foresight.stores.experience import ExperienceLibrary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-kb', required=True)
    parser.add_argument('--catalysts', required=True)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--max-skills', type=int)
    parser.add_argument('--output')
    args = parser.parse_args()

    exp_dir = ROOT / 'databases' / args.exp_kb / '03_experience'
    payload = json.loads(Path(args.catalysts).read_text(encoding='utf-8'))
    catalysts = payload.get('catalysts', payload) if isinstance(payload, dict) else payload
    if not isinstance(catalysts, list):
        raise SystemExit('catalyst file must contain a list or a {"catalysts": [...]} object')

    library = ExperienceLibrary(str(exp_dir))
    result = StabilityEvaluator(library, catalysts).evaluate_all(
        force=args.force, max_skills=args.max_skills)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()
