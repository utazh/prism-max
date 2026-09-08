#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" - <<'PY'
import importlib

modules = [
    "torch",
    "transformers",
    "numpy",
    "tqdm",
    "psutil",
    "lm_eval",
    "datasets",
    "sqlitedict",
]

for module_name in modules:
    importlib.import_module(module_name)
print("Python dependencies: OK")
PY

for path in \
  fewshots_datasets/input/openbookqa.jsonl \
  fewshots_datasets/input/copa.jsonl \
  fewshots_datasets/input/rte.jsonl \
  fewshots_datasets/input/copa-expand.jsonl \
  fewshots_datasets/input/rte-expand.jsonl \
  fewshots_datasets/input/openbookqa-small.jsonl \
  fewshots_datasets/input/piqa-small.jsonl
do
  test -f "$path"
done

bash -n scripts/figure_11.sh
bash -n scripts/figure_15.sh
bash -n scripts/figure22_acc.sh
bash -n scripts/lm_performance_fast_pcache.sh

echo "Script and input checks: OK"
