"""Generate and validate IMPRESS importance-based physical token reorderings."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .paper_plan_generator import _average_heads, _head_prefix_scores, load_bundle_records
from .sparse_qwen_reprefill import PrefixStoreInfo, _token_hash, read_store_info


SCHEMA_VERSION = 1


def invert_permutation(physical_to_logical: Sequence[int]) -> list[int]:
    """Return a logical-to-physical inverse after strict permutation validation."""

    order = [int(token) for token in physical_to_logical]
    if sorted(order) != list(range(len(order))):
        raise ValueError("reorder row must be a permutation of every prefix token")
    inverse = [0] * len(order)
    for physical, logical in enumerate(order):
        inverse[logical] = physical
    return inverse


def importance_order(scores: Sequence[float]) -> list[int]:
    """Pack tokens by descending finite non-negative importance, then token ID."""

    normalized = [float(score) for score in scores]
    if not normalized:
        raise ValueError("importance scores must not be empty")
    if any(not math.isfinite(score) or score < 0 for score in normalized):
        raise ValueError("importance scores must be finite and non-negative")
    return sorted(range(len(normalized)), key=lambda token: (-normalized[token], token))


def reorder_manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validated_task_mapping(
    payload: Mapping[str, Any],
    *,
    task: str,
    info: PrefixStoreInfo,
) -> tuple[list[list[int]], list[list[int]]]:
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported IMPRESS reorder manifest schema")
    if payload.get("method") != "impress":
        raise ValueError("reorder manifest is not for IMPRESS")
    tasks = payload.get("tasks")
    if not isinstance(tasks, Mapping) or task not in tasks:
        raise KeyError(f"reorder manifest has no task {task!r}")
    entry = tasks[task]
    if not isinstance(entry, Mapping):
        raise ValueError(f"invalid reorder entry for {task}")
    expected = {
        "prefix_tokens": info.prefix_tokens,
        "layers": info.layers,
        "token_hash": info.token_hash,
    }
    for key, value in expected.items():
        if entry.get(key) != value:
            raise ValueError(
                f"reorder {task} {key} {entry.get(key)!r} does not match {value!r}"
            )
    rows = entry.get("physical_to_logical")
    if not isinstance(rows, list) or len(rows) != info.layers:
        raise ValueError(f"reorder {task} must contain {info.layers} layer rows")
    physical_to_logical = []
    logical_to_physical = []
    for layer, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != info.prefix_tokens:
            raise ValueError(
                f"reorder {task} layer {layer} must contain {info.prefix_tokens} tokens"
            )
        normalized = [int(token) for token in row]
        inverse = invert_permutation(normalized)
        physical_to_logical.append(normalized)
        logical_to_physical.append(inverse)
    return physical_to_logical, logical_to_physical


def load_task_reorder(
    path: str | Path,
    *,
    task: str,
    info: PrefixStoreInfo,
) -> tuple[list[list[int]], list[list[int]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("IMPRESS reorder manifest must be a JSON object")
    return _validated_task_mapping(payload, task=task, info=info)


def generate_reorder_manifest(
    *,
    model_path: str,
    bundle_dir: str | Path,
    store_root: str | Path,
    tasks: Sequence[str],
    samples_per_task: int,
    sample_offset: int,
    device: str,
    dtype: str,
    allow_gpu: bool,
    max_prompt_tokens: int,
) -> dict[str, Any]:
    """Average full-attention importance over a history trace and rank tokens."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda") and not allow_gpu:
        raise ValueError("cuda requires --allow-gpu on a shared server")
    if samples_per_task <= 0:
        raise ValueError("samples_per_task must be positive")
    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device).eval()
    rows = load_bundle_records(bundle_dir, tasks, sample_offset + samples_per_task)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["task"]), []).append(row)

    task_payloads: dict[str, Any] = {}
    torch.set_grad_enabled(False)
    for task in tasks:
        available_rows = grouped.get(task, [])
        task_rows = available_rows[sample_offset : sample_offset + samples_per_task]
        if len(task_rows) != samples_per_task:
            raise ValueError(
                f"{task} has {len(task_rows)} history requests, expected {samples_per_task}"
            )
        prefix_text = str(task_rows[0]["prefix_text"])
        if any(str(row["prefix_text"]) != prefix_text for row in task_rows):
            raise ValueError(f"{task} history does not use one shared prefix")
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
        info = read_store_info(store_root, task)
        if len(prefix_ids) != info.prefix_tokens or _token_hash(prefix_ids) != info.token_hash:
            raise ValueError(f"{task} bundle prefix does not match the persisted KV store")
        if len(prefix_ids) > max_prompt_tokens:
            raise ValueError(f"{task} prefix exceeds max_prompt_tokens")
        prefix_inputs = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            prefix_output = model(input_ids=prefix_inputs, use_cache=True)
        prefix_cache = prefix_output.past_key_values
        accumulated = [[0.0] * info.prefix_tokens for _ in range(info.layers)]

        for row in task_rows:
            query_ids = tokenizer(
                str(row["query_text"]), add_special_tokens=False
            ).input_ids
            if not query_ids or info.prefix_tokens + len(query_ids) > max_prompt_tokens:
                raise ValueError(f"invalid query length for {row['uid']}")
            query_inputs = torch.tensor([query_ids], dtype=torch.long, device=device)
            mask = torch.ones(
                (1, info.prefix_tokens + len(query_ids)), dtype=torch.long, device=device
            )
            previous_impl = getattr(model.config, "_attn_implementation", None)
            if hasattr(model, "set_attn_implementation"):
                model.set_attn_implementation("eager")
            else:
                model.config._attn_implementation = "eager"
            query_cache = copy.deepcopy(prefix_cache)
            try:
                with torch.inference_mode():
                    output = model(
                        input_ids=query_inputs,
                        past_key_values=query_cache,
                        attention_mask=mask,
                        output_attentions=True,
                        use_cache=False,
                    )
            finally:
                if previous_impl is not None:
                    if hasattr(model, "set_attn_implementation"):
                        model.set_attn_implementation(previous_impl)
                    else:
                        model.config._attn_implementation = previous_impl
            layer_heads = _head_prefix_scores(
                output.attentions, info.prefix_tokens, len(query_ids)
            )
            if len(layer_heads) != info.layers:
                raise RuntimeError(f"{task} attention layer count changed")
            for layer, heads in enumerate(layer_heads):
                scores = _average_heads(heads)
                for token, score in enumerate(scores):
                    accumulated[layer][token] += float(score)
            del output, query_cache, query_inputs, mask
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        physical_to_logical = [importance_order(scores) for scores in accumulated]
        for row in physical_to_logical:
            invert_permutation(row)
        task_payloads[task] = {
            "prefix_tokens": info.prefix_tokens,
            "layers": info.layers,
            "token_hash": info.token_hash,
            "history_uids": [str(row["uid"]) for row in task_rows],
            "physical_to_logical": physical_to_logical,
        }
        del prefix_output, prefix_cache, prefix_inputs
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return {
        "schema_version": SCHEMA_VERSION,
        "method": "impress",
        "mapping": "physical positions sorted by descending mean full-attention token importance",
        "model_path": model_path,
        "model_compute_dtype": dtype,
        "bundle_dir": str(bundle_dir),
        "samples_per_task": samples_per_task,
        "sample_offset": sample_offset,
        "tasks": task_payloads,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--tasks", default="sst2,subj,trec,rte")
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    args = parser.parse_args()
    tasks = [task.strip().lower() for task in args.tasks.split(",") if task.strip()]
    payload = generate_reorder_manifest(
        model_path=args.model_path,
        bundle_dir=args.bundle_dir,
        store_root=args.store_root,
        tasks=tasks,
        samples_per_task=args.samples_per_task,
        sample_offset=args.sample_offset,
        device=args.device,
        dtype=args.dtype,
        allow_gpu=args.allow_gpu,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": reorder_manifest_sha256(output),
                "tasks": list(payload["tasks"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
