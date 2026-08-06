#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
DATASET="Astock-test"
python scripts/get_score.py "$DATASET"
