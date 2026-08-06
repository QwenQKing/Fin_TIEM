#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python scripts/run_inference.py --kb Astock \
  --catalysts datasets/data-db/expr/Astock.json --build-exp --reset-exp
