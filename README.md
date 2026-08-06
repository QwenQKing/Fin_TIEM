# TIEM

TIEM is a point-in-time-audited framework that integrates timestamp-filtered event-evidence hypergraph retrieval, provenance-tagged causal skill memory, and heterogeneous evidence-experience fusion reasoning for event-driven catalyst-outcome forecasting.

## Install

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export LLM_MODEL=gpt-4o-mini
```

## Build

```bash
bash scripts/build_databases.sh
bash scripts/build_expr.sh
```

## Evaluate

```bash
bash scripts/eval.sh
bash scripts/get_score.sh
```
