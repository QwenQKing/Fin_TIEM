#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
DATASET="${1:-Astock-eval}"
python scripts/get_score.py "$DATASET"
