#!/usr/bin/env python3
"""Calibrate FP16 per-layer KV retention budgets on the real sparse runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from contiguous_fuxian.flexgen_pcache import FlexGenPcacheConfig, FlexGenPcacheStore
from contiguous_fuxian.flexgen_qwen_reprefill import (
    _load_model_and_tokenizer,
    _load_plan_metadata,
    build_online_layer_plan,
    greedy_flexgen_completion,
    parse_head_ids,
    validate_impress_block_mode,
    validate_online_selector_mapping,
    validate_plan_keep_ratio,
)
from contiguous_fuxian.layer_budget import (
    build_ranked_three_level_profile,
    profile_to_json_dict,
)
from contiguous_fuxian.sparse_qwen_reprefill import (
    _load_bundle_records,
    read_store_info,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logit_divergence(reference: Any, candidate: Any) -> dict[str, float | int]:
    import torch

    reference = reference.float()
    candidate = candidate.float()
    reference_logp = torch.log_softmax(reference, dim=-1)
    candidate_logp = torch.log_softmax(candidate, dim=-1)
    midpoint_logp = torch.logaddexp(reference_logp, candidate_logp) - math.log(2.0)
    reference_p = reference_logp.exp()
    candidate_p = candidate_logp.exp()
    kl = torch.sum(reference_p * (reference_logp - candidate_logp))
    js = 0.5 * (
        torch.sum(reference_p * (reference_logp - midpoint_logp))
        + torch.sum(candidate_p * (candidate_logp - midpoint_logp))
    )
    cosine = torch.nn.functional.cosine_similarity(
        reference.unsqueeze(0),
        candidate.unsqueeze(0),
    )[0]
    return {
        "kl_reference_to_candidate": max(0.0, float(kl.item())),
        "js_divergence": max(0.0, float(js.item())),
        "logit_mse": float(torch.mean((reference - candidate) ** 2).item()),
        "cosine_distance": max(0.0, 1.0 - float(cosine.item())),
        "top1_changed": int(reference.argmax().item() != candidate.argmax().item()),
    }


def _mean(rows: Sequence[dict[str, float | int]], key: str) -> float:
    return math.fsum(float(row[key]) for row in rows) / max(1, len(rows))


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    tasks = tuple(task.strip().lower() for task in args.tasks.split(",") if task.strip())
    if not tasks:
        raise ValueError("at least one calibration task is required")
    if not 0.0 < args.perturbed_ratio < args.target_mean_ratio <= 1.0:
        raise ValueError(
            "perturbed ratio must be positive and lower than the target mean ratio"
        )
    probe_query_heads = parse_head_ids(args.probe_query_heads)
    selector_kv_head_ids = parse_head_ids(args.selector_kv_head_ids)
    metadata = _load_plan_metadata(args.plan)
    target_ratio = validate_plan_keep_ratio(metadata, args.target_mean_ratio)
    method = str(metadata.get("method", ""))
    if method != "impress":
        raise ValueError("layer calibration requires an online IMPRESS/HyperInfer plan")
    chunk_size = int(metadata.get("chunk_size", 0))
    validate_impress_block_mode(
        method=method,
        online_selection=True,
        block_size=args.selection_block_size,
        physical_chunk_size=chunk_size,
        reorder_enabled=False,
    )

    rows = _load_bundle_records(args.bundle_dir, tasks, args.samples_per_task)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "calibration_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")

    model, tokenizer = _load_model_and_tokenizer(
        args.model_path,
        dtype=args.dtype,
        device=args.device,
    )
    layers = int(model.config.num_hidden_layers)
    validate_online_selector_mapping(
        model.config,
        method=method,
        selector_kv_head_ids=selector_kv_head_ids,
        probe_query_heads=probe_query_heads,
    )
    config = FlexGenPcacheConfig(
        flexgen_root=Path(args.flexgen_root),
        kv_dir=Path(args.flexgen_kv_dir),
        chunk_size=chunk_size,
        gpu_cache_mb=args.gpu_cache_mb,
        cpu_cache_mb=args.cpu_cache_mb,
        cache_type=args.cache_type,
        prefetch_time_budget=args.prefetch_time_budget,
        selector_kv_head_ids=selector_kv_head_ids,
        reuse_existing=True,
    )
    store = FlexGenPcacheStore(config)
    for task in tasks:
        store.add_task(store_root=args.store_root, task=task)

    per_layer: list[list[dict[str, float | int | str]]] = [
        [] for _ in range(layers)
    ]
    references: list[dict[str, Any]] = []
    uniform_ratios = (target_ratio,) * layers

    def run_logits(
        row: dict[str, Any],
        layer_ratios: Sequence[float],
    ) -> tuple[Any, float, dict[str, Any]]:
        task = str(row["task"])
        info = read_store_info(args.store_root, task)
        if info.layers != layers:
            raise ValueError(
                f"task {task} cache has {info.layers} layers; model has {layers}"
            )
        layer_plan = build_online_layer_plan(
            layers=layers,
            prefix_tokens=info.prefix_tokens,
            chunk_size=chunk_size,
        )
        loader = store.new_layer_loader(
            task=task,
            layer_plan=layer_plan,
            method=method,
            online_selection=True,
            keep_ratio=target_ratio,
            layer_keep_ratios=layer_ratios,
            probe_query_heads=probe_query_heads,
            similarity_alpha=args.similarity_alpha,
            impress_async_prefetch=args.async_prefetch,
            impress_period_prefetch_size=1,
        )
        query_ids = tokenizer(
            str(row["query_text"]),
            add_special_tokens=False,
        ).input_ids
        captured: list[Any] = []
        _, ttft_ms, _, metrics = greedy_flexgen_completion(
            model=model,
            tokenizer=tokenizer,
            query_token_ids=query_ids,
            prefix_tokens=info.prefix_tokens,
            loader=loader,
            max_tokens=1,
            period_size=args.period_size,
            subperiod_size=args.subperiod_size,
            impress_selection_block_size=args.selection_block_size,
            impress_period_prefetch_size=1,
            first_token_logits_out=captured,
        )
        if len(captured) != 1:
            raise RuntimeError("first-token calibration did not capture exactly one logit row")
        metrics["prefix_tokens"] = info.prefix_tokens
        return captured[0], ttft_ms, metrics

    try:
        for request_index, row in enumerate(rows):
            reference, reference_ttft, reference_metrics = run_logits(
                row,
                uniform_ratios,
            )
            reference_record = {
                "uid": str(row["uid"]),
                "task": str(row["task"]),
                "ttft_ms": reference_ttft,
                "top1_token": int(reference.argmax().item()),
                "effective_mean_keep_ratio": float(
                    reference_metrics["effective_mean_keep_ratio"]
                ),
                "selected_tokens_by_layer": list(
                    reference_metrics["selected_tokens_by_layer"]
                ),
            }
            references.append(reference_record)
            with progress_path.open("a", encoding="utf-8") as progress:
                progress.write(
                    json.dumps({"event": "reference", **reference_record}) + "\n"
                )
                progress.flush()
            print(
                json.dumps(
                    {
                        "event": "reference",
                        "request": request_index + 1,
                        "requests": len(rows),
                        **reference_record,
                    }
                ),
                flush=True,
            )

            for layer in range(layers):
                perturbed = list(uniform_ratios)
                perturbed[layer] = args.perturbed_ratio
                candidate, candidate_ttft, candidate_metrics = run_logits(
                    row,
                    perturbed,
                )
                divergence = _logit_divergence(reference, candidate)
                reference_selected = int(
                    reference_metrics["selected_tokens_by_layer"][layer]
                )
                candidate_selected = int(
                    candidate_metrics["selected_tokens_by_layer"][layer]
                )
                prefix_tokens = int(candidate_metrics["prefix_tokens"])
                removed_tokens = max(0, reference_selected - candidate_selected)
                record: dict[str, float | int | str] = {
                    "uid": str(row["uid"]),
                    "task": str(row["task"]),
                    "layer": layer,
                    "ttft_ms": candidate_ttft,
                    "effective_mean_keep_ratio": float(
                        candidate_metrics["effective_mean_keep_ratio"]
                    ),
                    "reference_layer_selected_tokens": reference_selected,
                    "candidate_layer_selected_tokens": candidate_selected,
                    "removed_tokens": removed_tokens,
                    "removed_fraction_of_prefix": removed_tokens
                    / max(1, prefix_tokens),
                    "actual_drop_succeeded": int(removed_tokens > 0),
                    **divergence,
                }
                per_layer[layer].append(record)
                with progress_path.open("a", encoding="utf-8") as progress:
                    progress.write(
                        json.dumps({"event": "perturbation", **record}) + "\n"
                    )
                    progress.flush()
                print(
                    json.dumps(
                        {
                            "event": "perturbation",
                            "request": request_index + 1,
                            "requests": len(rows),
                            "layer": layer,
                            "layers": layers,
                            "js_divergence": record["js_divergence"],
                        }
                    ),
                    flush=True,
                )
                del candidate
                torch.cuda.empty_cache()
            del reference
            torch.cuda.empty_cache()
    finally:
        store.close()

    layer_rows = []
    sensitivity_scores = []
    for layer, measurements in enumerate(per_layer):
        summary = {
            "layer": layer,
            "mean_js_divergence": _mean(measurements, "js_divergence"),
            "mean_kl_reference_to_candidate": _mean(
                measurements,
                "kl_reference_to_candidate",
            ),
            "mean_logit_mse": _mean(measurements, "logit_mse"),
            "mean_cosine_distance": _mean(measurements, "cosine_distance"),
            "top1_flip_rate": _mean(measurements, "top1_changed"),
            "mean_removed_tokens": _mean(measurements, "removed_tokens"),
            "mean_removed_fraction_of_prefix": _mean(
                measurements,
                "removed_fraction_of_prefix",
            ),
            "actual_drop_success_rate": _mean(
                measurements,
                "actual_drop_succeeded",
            ),
            "measurements": measurements,
        }
        layer_rows.append(summary)

    normalized_scores = [
        (
            float(row["mean_js_divergence"])
            / float(row["mean_removed_fraction_of_prefix"])
            if float(row["mean_removed_fraction_of_prefix"]) >= 0.05
            else 0.0
        )
        for row in layer_rows
    ]
    resolved_scores = [
        score
        for score, row in zip(normalized_scores, layer_rows)
        if float(row["actual_drop_success_rate"]) >= 0.5
        and float(row["mean_removed_fraction_of_prefix"]) >= 0.05
    ]
    unresolved_floor = max(resolved_scores, default=0.0) + 1.0
    for normalized_score, row in zip(normalized_scores, layer_rows):
        unresolved = (
            float(row["actual_drop_success_rate"]) < 0.5
            or float(row["mean_removed_fraction_of_prefix"]) < 0.05
        )
        profile_score = (
            unresolved_floor
            + (1.0 - float(row["actual_drop_success_rate"]))
            + normalized_score
            if unresolved
            else normalized_score
        )
        row["normalized_js_per_removed_prefix_fraction"] = normalized_score
        row["fallback_or_no_drop_unresolved"] = unresolved
        row["profile_sensitivity_score"] = profile_score
        sensitivity_scores.append(profile_score)

    generated_at = datetime.now(timezone.utc).isoformat()
    sensitivity_payload = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "model": args.model_path,
        "protocol": {
            "name": "single-layer-fp16-retention-ablation-v1",
            "reference_ratio": target_ratio,
            "perturbed_ratio": args.perturbed_ratio,
            "score": (
                "first-token Jensen-Shannon divergence normalized by actual "
                "removed KV; no-drop fallback layers are excluded from low budget"
            ),
            "selection": "unchanged HyperInfer online importance selector",
            "selection_block_size": args.selection_block_size,
            "storage_precision": "float16",
            "async_prefetch": args.async_prefetch,
            "predictive_period_prefetch_size": 1,
            "tasks": list(tasks),
            "samples_per_task": args.samples_per_task,
            "request_uids": [str(row["uid"]) for row in rows],
        },
        "references": references,
        "layers": layer_rows,
    }
    sensitivity_path = output / "layer_sensitivity.json"
    sensitivity_path.write_text(
        json.dumps(sensitivity_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    sensitivity_sha256 = _sha256(sensitivity_path)

    calibration_metadata = {
        "method": "single-layer-fp16-retention-ablation-v1",
        "generated_at_utc": generated_at,
        "reference_ratio": target_ratio,
        "perturbed_ratio": args.perturbed_ratio,
        "score": "fallback_aware_js_per_actual_removed_prefix_fraction",
        "fallback_low_budget_policy": (
            "layers with <50% successful perturbations or <5% mean actual "
            "prefix removal rank above all low-budget candidates"
        ),
        "selection": "HyperInfer online importance, unchanged",
        "selection_block_size": args.selection_block_size,
        "storage_precision": "float16",
        "tasks": list(tasks),
        "samples_per_task": args.samples_per_task,
        "request_uids": [str(row["uid"]) for row in rows],
        "sensitivity_artifact": sensitivity_path.name,
        "sensitivity_sha256": sensitivity_sha256,
        "profile_delta": args.profile_delta,
        "extreme_fraction": args.extreme_fraction,
    }
    profile = build_ranked_three_level_profile(
        sensitivity_scores,
        target_mean_ratio=target_ratio,
        delta=args.profile_delta,
        model=args.model_path,
        calibration=calibration_metadata,
        extreme_fraction=args.extreme_fraction,
    )
    profile_path = output / "layer_budget_profile.json"
    profile_path.write_text(
        json.dumps(
            profile_to_json_dict(profile),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "sensitivity_path": str(sensitivity_path),
        "sensitivity_sha256": sensitivity_sha256,
        "profile_path": str(profile_path),
        "profile_sha256": _sha256(profile_path),
        "target_mean_ratio": profile.target_mean_ratio,
        "layer_ratios": list(profile.layer_ratios),
        "ranked_layers_most_to_least_sensitive": sorted(
            range(layers),
            key=lambda layer: (-sensitivity_scores[layer], layer),
        ),
    }
    (output / "calibration_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a three-level FP16 layer retention profile using the "
            "actual HyperInfer selector and 16-token storage path."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--flexgen-root", required=True)
    parser.add_argument("--flexgen-kv-dir", required=True)
    parser.add_argument("--tasks", default="sst2,subj,trec,rte")
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--target-mean-ratio", type=float, default=0.5)
    parser.add_argument("--perturbed-ratio", type=float, default=0.25)
    parser.add_argument("--profile-delta", type=float, default=0.25)
    parser.add_argument("--extreme-fraction", type=float, default=0.25)
    parser.add_argument("--selection-block-size", type=int, default=16)
    parser.add_argument("--period-size", type=int, default=8)
    parser.add_argument("--subperiod-size", type=int, default=4)
    parser.add_argument("--probe-query-heads", default="0,1,2")
    parser.add_argument("--selector-kv-head-ids", default="0")
    parser.add_argument("--similarity-alpha", type=float, default=0.6)
    parser.add_argument(
        "--async-prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gpu-cache-mb", type=float, default=55.0)
    parser.add_argument("--cpu-cache-mb", type=float, default=131.0)
    parser.add_argument("--cache-type", choices=("LRU", "LFU", "CKLFU"), default="LRU")
    parser.add_argument("--prefetch-time-budget", type=float, default=10_000.0)
    args = parser.parse_args()

    result = calibrate(args)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
