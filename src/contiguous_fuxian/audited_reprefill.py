"""Explicit fixed-P8 entrypoint over the existing PRISM GPU/data path.

No GPU results are bundled with this module. Historical CLI defaults are left
unchanged. A ContextVar passes run options to the existing selector calls;
the original decoder consumes decision.period to schedule real anchor layers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .promixed import selection_experiment


BASE_SNAPSHOT = "5dd35b501a3a4025dda5c81b6aa348d19af4f7a5"


def token_hash(ids) -> str:
    h = hashlib.sha256()
    for token in ids:
        h.update(int(token).to_bytes(4, "little", signed=False))
    return h.hexdigest()


def validate_prefix_identity(records, tokenizer, metadata: dict[str, dict]) -> None:
    """Reject mismatched row prefixes before executing any new GPU benchmark."""
    seen = {}
    for row in records:
        task = str(row["task"])
        text = str(row["prefix_text"])
        if task in seen and seen[task] != text:
            raise ValueError(f"task {task} contains multiple different prefixes")
        if task in seen:
            continue
        seen[task] = text
        ids = tokenizer(text, add_special_tokens=False).input_ids
        info = metadata[task]
        if len(ids) != int(info["prefix_tokens"]) or token_hash(ids) != info["token_hash"]:
            raise ValueError(f"task {task}: bundle prefix does not match stored KV token hash")


def checked_summary(summary: dict, *, strategy: str, coverage_enabled: bool,
                    anchor_preload: bool, trace_enabled: bool) -> dict:
    """Annotate actual overrides, and fail if fixed-P8 did not take effect."""
    for task, metrics in summary["tasks"].items():
        if not math.isclose(float(metrics["mean_promixed_period"]), 8.0, abs_tol=1e-9):
            raise RuntimeError(f"{task}: decoder did not report fixed P8 decisions")
        if any(float(metrics.get(f"mean_promixed_p{p}_decisions", 0)) != 0 for p in (1, 2, 4)):
            raise RuntimeError(f"{task}: adaptive decisions leaked into fixed-P8 run")
    runtime = summary["runtime"]
    old_variant = runtime.get("runtime_variant", "")
    if "+promixed-gqa-adaptive-p8" not in old_variant:
        raise RuntimeError("unexpected historical runtime variant; integration needs review")
    runtime["runtime_variant"] = old_variant.replace(
        "+promixed-gqa-adaptive-p8", f"+promixed-gqa-fixed-p8+selection-{strategy}")
    runtime["selection_override"] = dict(fixed_period=8, strategy=strategy,
        coverage_enabled=coverage_enabled, anchor_only_selector_preload=anchor_preload)
    runtime["historical_fusion_active"] = strategy == "legacy"
    runtime["round_robin_active"] = strategy == "legacy" and coverage_enabled
    runtime["adaptive_reuse_active"] = False
    runtime["latency_contains_selector_trace"] = trace_enabled
    runtime["audit_base_snapshot"] = BASE_SNAPSHOT
    runtime["actual_payload_quantization"] = "none; FP16 storage, BF16 model compute"
    runtime["ssd_byte_counter_semantics"] = (
        "application-estimated bytes attributed to disk-tier sources, NOT measured NVMe bytes")
    return summary


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "bundle-dir", "store-root", "flexgen-root", "flexgen-kv-dir", "output-dir"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--tasks", default="trec")
    p.add_argument("--store-tasks", default="sst2,subj,trec,rte")
    p.add_argument("--samples-per-task", type=int, default=128,
                   help="Positive count; use a large positive value for all rows, never -1")
    p.add_argument("--keep-ratio", type=float, default=0.1)
    p.add_argument("--selector-index-dir", help="Existing INT4 index; omission uses original FP16 Pcache path")
    p.add_argument("--layer-budget-profile", help="Omit for uniform layer budget; no retuning by test labels")
    p.add_argument("--selection", choices=("legacy", "balanced", "mean", "normalized_mean", "max"), default="legacy")
    p.add_argument("--coverage", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--anchor-preload", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trace", action="store_true", help="Diagnostic only: records score matrices; invalid for headline latency")
    p.add_argument("--gpu-cache-mb", type=float, default=55)
    p.add_argument("--cpu-cache-mb", type=float, default=131)
    p.add_argument("--warmup-passes", type=int, default=1)
    p.add_argument("--warmup-samples-per-task", type=int, default=32)
    p.add_argument("--dry-run", action="store_true", help="Validate metadata and print configuration; no GPU, no file writes")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples_per_task <= 0:
        raise ValueError("samples-per-task must be positive; -1 otherwise silently drops the last row")
    if not math.isfinite(args.keep_ratio) or not 0 < args.keep_ratio <= 1:
        raise ValueError("keep-ratio must be in (0, 1]")
    for value in (args.gpu_cache_mb, args.cpu_cache_mb):
        if not math.isfinite(value) or value < 0:
            raise ValueError("cache capacity must be finite and non-negative")
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    store_tasks = [x.strip() for x in args.store_tasks.split(",") if x.strip()]
    if not tasks or len(set(tasks)) != len(tasks) or len(set(store_tasks)) != len(store_tasks) or not set(tasks) <= set(store_tasks):
        raise ValueError("tasks must be unique and included in the persisted store task order")
    if args.warmup_passes < 0 or args.warmup_samples_per_task <= 0:
        raise ValueError("invalid warmup configuration")
    metadata = {task: json.loads((Path(args.store_root) / task / "metadata.json").read_text()) for task in store_tasks}
    geometries = {(int(v['layers']), int(v['kv_heads']), int(v['head_dim'])) for v in metadata.values()}
    if len(geometries) != 1:
        raise ValueError("store task geometries disagree")
    layers, heads, _ = next(iter(geometries))
    model_config = json.loads((Path(args.model_path) / "config.json").read_text())
    if (layers != int(model_config['num_hidden_layers']) or heads != int(model_config['num_key_value_heads'])
            or model_config.get('model_type') != 'qwen2'):
        raise ValueError("this audited entrypoint requires a matching Qwen2 model/store")
    groups = int(model_config['num_attention_heads']) // heads
    records = []
    for task in tasks:
        data = [json.loads(line) for line in (Path(args.bundle_dir) / f"{task}.jsonl").read_text().splitlines() if line.strip()]
        records.extend(data[:args.samples_per_task])
    if not records:
        raise ValueError("empty evaluation set")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite any existing run: {output}")
    spec = dict(base_snapshot=BASE_SNAPSHOT, fixed_period=8, selection=args.selection,
        coverage_enabled=args.coverage, anchors=list(range(0, layers, 8)),
        selector_backend='k4-preloaded' if args.selector_index_dir else 'fp16-pcache',
        layer_budget='profile' if args.layer_budget_profile else 'uniform',
        anchor_only_preload=args.anchor_preload, samples=len(records),
        note='fixed P8 shares score decisions/indices, never KV values across layers')
    if args.dry_run:
        print(json.dumps(spec, indent=2))
        return 0
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    validate_prefix_identity(records, tokenizer, metadata)
    from .flexgen_pcache import FlexGenPcacheConfig
    from .flexgen_qwen_reprefill import run_flexgen_reprefill
    output.mkdir(parents=True, exist_ok=False)
    (output / 'audit_run_spec.json').write_text(json.dumps(spec, indent=2))
    plan_path = output / 'online_shape_plan.json'
    plan_path.write_text(json.dumps({'metadata': {'method': 'impress', 'chunk_size': 16, 'keep_ratio': args.keep_ratio}}))
    profile_path = Path(args.layer_budget_profile) if args.layer_budget_profile else output / 'uniform_layer_budget.json'
    if not args.layer_budget_profile:
        profile_path.write_text(json.dumps(dict(schema_version=1, model=args.model_path,
            target_mean_ratio=args.keep_ratio, layer_ratios=[args.keep_ratio] * layers,
            calibration={'method': 'uniform-control-no-label-calibration'})))
    trace = [] if args.trace else None
    with selection_experiment(fixed_period=8, strategy=args.selection, coverage_enabled=args.coverage,
                              trace=trace, preload_anchors_only=args.anchor_preload):
        summary = run_flexgen_reprefill(
            model_path=args.model_path, bundle_dir=args.bundle_dir, store_root=args.store_root,
            plan_path=plan_path, tasks=tasks, store_tasks=store_tasks,
            samples_per_task=args.samples_per_task, output_dir=output, max_tokens=1,
            accuracy_scoring='label_continuation_loglikelihood', device='cuda', dtype='bfloat16',
            flexgen_config=FlexGenPcacheConfig(flexgen_root=Path(args.flexgen_root),
                kv_dir=Path(args.flexgen_kv_dir), chunk_size=16, gpu_cache_mb=args.gpu_cache_mb,
                cpu_cache_mb=args.cpu_cache_mb, cache_type='CKLFU',
                selector_kv_head_ids=tuple(range(heads)),
                selector_index_dir=Path(args.selector_index_dir) if args.selector_index_dir else None,
                reuse_existing=True),
            period_size=8, subperiod_size=4, expected_keep_ratio=args.keep_ratio,
            warmup_passes=args.warmup_passes, warmup_samples_per_task=args.warmup_samples_per_task,
            online_selection=True, probe_query_heads=tuple(g * groups for g in range(heads)),
            impress_async_prefetch=True, impress_selection_block_size=16,
            layer_budget_profile=profile_path, exact_layer_block_budget=True,
            impress_selection_period_size=8, impress_known_period_prefetch=True,
            impress_value_ordered_prefetch=True, impress_value_prefetch_budget_scale=0.9,
            promixed_gqa_selection=True, promixed_adaptive_coverage=False,
            defer_cache_score_updates=False)
    summary = checked_summary(summary, strategy=args.selection, coverage_enabled=args.coverage,
                              anchor_preload=args.anchor_preload, trace_enabled=args.trace)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    if trace is not None:
        # Includes warm-up and per-budget function calls. It is NOT a per-layer,
        # per-request trace and must not be treated as independent model samples.
        with (output / 'selector_invocation_trace.jsonl').open('w') as f:
            for row in trace:
                f.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
