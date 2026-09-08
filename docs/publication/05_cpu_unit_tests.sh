set -euo pipefail
cd repo
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" /home/panzihang/venvs/vllm-stable/bin/python -m pytest -p no:cacheprovider -q tests/test_flexgen_pcache.py tests/test_flexgen_qwen_reprefill.py tests/test_promixed.py tests/test_quantized_key_index.py tests/test_prism_max_runner.py