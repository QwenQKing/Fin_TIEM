#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export TIEM_USE_C2F=1
export TIEM_USE_CSM=0
export TIEM_USE_SCMR=0
export TIEM_PIT_OFF=0
python scripts/run_inference.py --kb Astock \
  --catalysts datasets/data-db/expr/Astock.json --build-exp --reset-exp
