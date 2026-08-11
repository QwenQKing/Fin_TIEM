#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
for ds in Astock FinPURE CMIN-US_ood EDT_ood CSMD_ood; do
  echo "====  EEH: $ds ===="
  python scripts/build_databases.py --data "datasets/data-db/event/$ds.json"
done
