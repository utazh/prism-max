#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  cache/kvs \
  cache/prefix/opt-6.7b \
  cache/prefix/opt-13b \
  cache/prefix/opt-30b \
  fewshots_datasets/output \
  logs \
  time_logs

echo "Runtime directories are ready under $(pwd)."
