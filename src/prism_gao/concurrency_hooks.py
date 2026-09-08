"""Opt-in diagnostic hooks; defaults leave the V4 experiment unchanged."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    trace_file = os.environ.get("PRISM_GAO_CAPTURE_TRACE")
    if not trace_file:
        return
    from contiguous_fuxian.flexgen_pcache import FlexGenLayerLoader
    from contiguous_fuxian import flexgen_qwen_reprefill as runner
    from .precision_integration import _unique_blocks

    root = Path(__file__).resolve().parent
    target = Path(trace_file).resolve()
    if root not in target.parents:
        raise ValueError("trace output must stay inside prism_gao")
    target.parent.mkdir(parents=True, exist_ok=True)
    original_configure = FlexGenLayerLoader.configure_impress_layer
    original_completion = runner.greedy_flexgen_completion

    def configure(self, *args, **kwargs):
        result = original_configure(self, *args, **kwargs)
        b = int(self._config.chunk_size)
        selected = kwargs["selected_tokens"]
        priority = kwargs.get("prefetch_priority_tokens")
        if priority is None:
            priority = selected
        layers = getattr(self, "_prism_gao_capture_layers", {})
        layers.setdefault(int(kwargs["layer"]), {
            "layer": int(kwargs["layer"]),
            "selected_blocks": list(_unique_blocks(selected, b)),
            "priority_blocks": list(_unique_blocks(priority, b)),
        })
        self._prism_gao_capture_layers = layers
        return result

    def completion(**kwargs):
        result = original_completion(**kwargs)
        loader = kwargs["loader"]
        layers = getattr(loader, "_prism_gao_capture_layers", {})
        query = list(map(int, kwargs["query_token_ids"]))
        trace_id = hashlib.sha256(json.dumps(query).encode()).hexdigest()
        record = {
            "schema": "prism-concurrency-trace-v1",
            "trace_id": trace_id, "task": loader._selector_index_task,
            "prefix_tokens": int(kwargs["prefix_tokens"]),
            "suffix_tokens": len(query),
            "block_size": int(loader._config.chunk_size),
            "budget": float(os.environ["PRISM_GAO_GLOBAL_KEEP_RATIO"]),
            "selector": os.environ.get("PRISM_GAO_SELECTOR_MODE"),
            "reuse_policy": os.environ.get("PRISM_GAO_PERIOD_POLICY"),
            "source_precision": os.environ.get("PRISM_GAO_PRECISION_MODE"),
            "layers": [layers[key] for key in sorted(layers)],
        }
        # Original response/accuracy timers and close have already finished.
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        result[3]["prism_gao_trace_id"] = trace_id
        return result

    FlexGenLayerLoader.configure_impress_layer = configure
    runner.greedy_flexgen_completion = completion
