#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
SPLIT="${1:-eval}"
for ds in Astock FinPURE CMIN-US_ood EDT_ood CSMD_ood; do
  echo "==== eval [$SPLIT] $ds (events= EEH, =Astock) ===="
  python scripts/run_inference.py --kb "$ds" --catalysts "datasets/$SPLIT/$ds.json" --exp-kb Astock
done
