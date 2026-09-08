"""Run the PRISM CLI with optional process-local PRISM-Gao hooks."""

from __future__ import annotations

import os


def _install_sample_offset() -> None:
    """Use a disjoint slice without changing the original dataset loader."""
    offset = int(os.environ.get("PRISM_GAO_SAMPLE_OFFSET", "0") or "0")
    if offset < 0:
        raise ValueError("PRISM_GAO_SAMPLE_OFFSET must be non-negative")
    if offset == 0:
        return
    from contiguous_fuxian import flexgen_qwen_reprefill

    original = flexgen_qwen_reprefill._load_bundle_records

    def shifted(bundle_dir, tasks, samples_per_task):
        rows = []
        for task in tasks:
            available = original(bundle_dir, [task], int(samples_per_task) + offset)
            selected = available[offset : offset + int(samples_per_task)]
            if len(selected) != int(samples_per_task):
                raise ValueError("task " + repr(task) + " lacks records for sample offset " + str(offset))
            rows.extend(selected)
        return rows

    flexgen_qwen_reprefill._load_bundle_records = shifted


def main() -> int:
    selector_mode = os.environ.get("PRISM_GAO_SELECTOR_MODE", "exact").strip().lower()
    if selector_mode == "torch":
        # True Stage-A baseline: do not patch QuantizedKeyIndex.load_layer or the
        # Qwen selector scorer.
        if os.environ.get("PRISM_GAO_PRECISION_MODE", "").strip().lower() not in {
            "",
            "off",
            "none",
        }:
            raise ValueError("torch selector mode cannot enable Stage-C hooks")
        if os.environ.get("PRISM_GAO_PERIOD_POLICY", "original").strip().lower() not in {
            "",
            "original",
            "v1",
        } or int(os.environ.get("PRISM_GAO_FIXED_PERIOD", "0") or "0") != 0:
            raise ValueError("torch selector mode is only for the untouched Stage-A baseline")
    else:
        if selector_mode not in {"exact", "direct"}:
            raise ValueError("PRISM_GAO_SELECTOR_MODE must be torch, exact, or direct")
        from .fused_selector_integration import install

        install()

    _install_sample_offset()
    from .concurrency_hooks import install as install_concurrency_hooks
    install_concurrency_hooks()
    from contiguous_fuxian.flexgen_qwen_reprefill import main as original_main

    return int(original_main())


if __name__ == "__main__":
    raise SystemExit(main())
