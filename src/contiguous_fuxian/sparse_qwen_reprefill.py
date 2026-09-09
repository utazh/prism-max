"""Qwen sparse re-prefill runner with physically removed per-layer KV entries.

This runner is deliberately separate from the generic LMCache connector.  It
stores the full shared prefix KV cache offline, then reconstructs only the
selected contiguous physical chunks for each layer.  The decoder uses a
layer-specific causal mask, so dropped KV entries are absent from attention
rather than represented by zero vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .lmcache_plan import policy_request_id
from .paper_client import percentile95, prediction_is_correct


BYTES_PER_BFLOAT16 = 2


@dataclass(frozen=True)
class PrefixStoreInfo:
    task: str
    prefix_tokens: int
    kv_heads: int
    head_dim: int
    layers: int
    token_hash: str

    @property
    def bytes_per_token_per_tensor(self) -> int:
        return self.kv_heads * self.head_dim * BYTES_PER_BFLOAT16


def selected_chunk_indices(tiers: Sequence[str]) -> list[int]:
    """Return sorted physical chunk indices that are retained by a plan row."""

    return [index for index, tier in enumerate(tiers) if str(tier).lower() != "drop"]


def contiguous_spans(indices: Iterable[int]) -> list[tuple[int, int]]:
    """Coalesce sorted chunk IDs into half-open ranges for sequential reads."""

    ordered = sorted(set(int(index) for index in indices))
    if not ordered:
        return []
    spans = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        spans.append((start, previous + 1))
        start = previous = index
    spans.append((start, previous + 1))
    return spans


def _token_hash(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _load_bundle_records(
    bundle_dir: str | Path,
    tasks: Sequence[str],
    samples_per_task: int,
) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        path = Path(bundle_dir) / f"{task}.jsonl"
        task_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not task_rows:
            raise ValueError(f"no task records in {path}")
        rows.extend(task_rows[:samples_per_task])
    return rows


def _task_prefixes(rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    prefixes: dict[str, str] = {}
    for row in rows:
        task = str(row["task"])
        prefix = str(row["prefix_text"])
        existing = prefixes.setdefault(task, prefix)
        if existing != prefix:
            raise ValueError(f"{task} does not have one shared prefix")
    return prefixes


def _task_dir(store_root: str | Path, task: str) -> Path:
    return Path(store_root) / task


def _metadata_path(store_root: str | Path, task: str) -> Path:
    return _task_dir(store_root, task) / "metadata.json"


def _tensor_path(store_root: str | Path, task: str, layer: int, kind: str) -> Path:
    return _task_dir(store_root, task) / f"layer_{layer:02d}_{kind}.bf16"


def _write_prefix_tensor(path: Path, tensor) -> None:
    """Write [batch, heads, tokens, dim] as token-major raw BF16 bytes."""

    import torch

    token_major = tensor.detach().to(torch.bfloat16).permute(0, 2, 1, 3).contiguous().view(torch.uint8).cpu()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(token_major.numpy().tobytes())
    temporary.replace(path)


def _info_from_payload(payload: MappingLike) -> PrefixStoreInfo:
    return PrefixStoreInfo(
        task=str(payload["task"]),
        prefix_tokens=int(payload["prefix_tokens"]),
        kv_heads=int(payload["kv_heads"]),
        head_dim=int(payload["head_dim"]),
        layers=int(payload["layers"]),
        token_hash=str(payload["token_hash"]),
    )


# Keeps the standard-library-facing annotations light; torch is loaded only in GPU commands.
MappingLike = dict[str, Any]


def read_store_info(store_root: str | Path, task: str) -> PrefixStoreInfo:
    return _info_from_payload(json.loads(_metadata_path(store_root, task).read_text(encoding="utf-8")))


def _store_matches(store_root: str | Path, task: str, token_hash: str) -> bool:
    metadata_path = _metadata_path(store_root, task)
    if not metadata_path.exists():
        return False
    try:
        info = read_store_info(store_root, task)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if info.token_hash != token_hash:
        return False
    return all(
        _tensor_path(store_root, task, layer, kind).exists()
        for layer in range(info.layers)
        for kind in ("key", "value")
    )


def prepare_prefix_store(
    *,
    model_path: str,
    bundle_dir: str | Path,
    store_root: str | Path,
    tasks: Sequence[str],
    samples_per_task: int,
    device: str,
    dtype: str,
    overwrite: bool = False,
) -> dict[str, PrefixStoreInfo]:
    """Compute shared prefixes once and persist every layer's full KV cache."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not device.startswith("cuda"):
        raise ValueError("the sparse Qwen runner is designed for CUDA execution")
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device).eval()
    rows = _load_bundle_records(bundle_dir, tasks, samples_per_task)
    summaries: dict[str, PrefixStoreInfo] = {}

    for task, prefix in _task_prefixes(rows).items():
        token_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        token_hash = _token_hash(token_ids)
        if not overwrite and _store_matches(store_root, task, token_hash):
            summaries[task] = read_store_info(store_root, task)
            continue
        task_dir = _task_dir(store_root, task)
        task_dir.mkdir(parents=True, exist_ok=True)
        prefix_inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = model(input_ids=prefix_inputs, use_cache=True)
        cache = output.past_key_values
        first_layer = cache.layers[0]
        info = PrefixStoreInfo(
            task=task,
            prefix_tokens=len(token_ids),
            kv_heads=int(first_layer.keys.shape[1]),
            head_dim=int(first_layer.keys.shape[-1]),
            layers=len(cache.layers),
            token_hash=token_hash,
        )
        for layer, cache_layer in enumerate(cache.layers):
            _write_prefix_tensor(_tensor_path(store_root, task, layer, "key"), cache_layer.keys)
            _write_prefix_tensor(_tensor_path(store_root, task, layer, "value"), cache_layer.values)
        _metadata_path(store_root, task).write_text(
            json.dumps(
                {
                    "task": info.task,
                    "prefix_tokens": info.prefix_tokens,
                    "kv_heads": info.kv_heads,
                    "head_dim": info.head_dim,
                    "layers": info.layers,
                    "token_hash": info.token_hash,
                    "layout": "token,kv_head,head_dim",
                    "dtype": "bfloat16",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summaries[task] = info
        del output, cache, prefix_inputs
        torch.cuda.empty_cache()
    return summaries


def _read_layer_chunks(
    path: Path,
    info: PrefixStoreInfo,
    chunk_size: int,
    chunk_indices: Sequence[int],
):
    """Read selected token-major chunks, keeping their original relative order."""

    import torch

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    parts = []
    fd = os.open(path, os.O_RDONLY)
    try:
        for start_chunk, end_chunk in contiguous_spans(chunk_indices):
            token_start = start_chunk * chunk_size
            token_end = min(info.prefix_tokens, end_chunk * chunk_size)
            if token_start >= token_end:
                continue
            byte_offset = token_start * info.bytes_per_token_per_tensor
            byte_length = (token_end - token_start) * info.bytes_per_token_per_tensor
            payload = os.pread(fd, byte_length, byte_offset)
            if len(payload) != byte_length:
                raise RuntimeError(f"short read from {path}: {len(payload)} != {byte_length}")
            raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
            token_major = raw.view(torch.bfloat16).reshape(
                1, token_end - token_start, info.kv_heads, info.head_dim
            )
            parts.append(token_major.permute(0, 2, 1, 3).contiguous())
    finally:
        os.close(fd)
    if not parts:
        return torch.empty((1, info.kv_heads, 0, info.head_dim), dtype=torch.bfloat16)
    return torch.cat(parts, dim=2)


def materialize_sparse_cache(
    *,
    store_root: str | Path,
    task: str,
    layer_plan: Sequence[Sequence[str]],
    chunk_size: int,
    device: str,
    model_config,
) -> tuple[Any, dict[str, float]]:
    """Read selected chunks from SSD and build a per-layer variable-length cache."""

    import torch
    from transformers.cache_utils import DynamicCache

    info = read_store_info(store_root, task)
    if len(layer_plan) != info.layers:
        raise ValueError(f"{task} plan layers {len(layer_plan)} != stored layers {info.layers}")
    io_start = time.perf_counter()
    cpu_pairs = []
    selected_tokens = []
    for layer, tiers in enumerate(layer_plan):
        selected = selected_chunk_indices(tiers)
        key = _read_layer_chunks(_tensor_path(store_root, task, layer, "key"), info, chunk_size, selected)
        value = _read_layer_chunks(
            _tensor_path(store_root, task, layer, "value"), info, chunk_size, selected
        )
        cpu_pairs.append((key, value))
        selected_tokens.append(int(key.shape[2]))
    io_end = time.perf_counter()
    gpu_pairs = [(key.to(device), value.to(device)) for key, value in cpu_pairs]
    torch.cuda.synchronize()
    transfer_end = time.perf_counter()
    cache = DynamicCache(ddp_cache_data=gpu_pairs, config=model_config)
    torch.cuda.synchronize()
    cache_end = time.perf_counter()
    return cache, {
        "ssd_read_ms": (io_end - io_start) * 1000,
        "h2d_ms": (transfer_end - io_end) * 1000,
        "cache_build_ms": (cache_end - transfer_end) * 1000,
        "selected_tokens_min": min(selected_tokens),
        "selected_tokens_max": max(selected_tokens),
        "selected_tokens_mean": sum(selected_tokens) / max(1, len(selected_tokens)),
        "selected_kv_bytes": sum(selected_tokens)
        * info.bytes_per_token_per_tensor
        * 2,
    }


def _layer_causal_mask(*, query_tokens: int, past_tokens: int, dtype, device):
    import torch

    mask = torch.full(
        (1, 1, query_tokens, past_tokens + query_tokens),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    mask[:, :, :, :past_tokens] = 0
    for query_index in range(query_tokens):
        mask[:, :, query_index, past_tokens : past_tokens + query_index + 1] = 0
    return mask


def sparse_decoder_logits(model, input_ids, cache, position_start: int):
    """Run Qwen one layer at a time with masks sized to each sparse cache."""

    import torch

    core = model.model
    hidden_states = core.embed_tokens(input_ids)
    query_tokens = int(input_ids.shape[1])
    cache_position = torch.arange(
        position_start, position_start + query_tokens, device=input_ids.device
    )
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = core.rotary_emb(hidden_states, position_ids)
    for layer_index, decoder_layer in enumerate(core.layers[: core.config.num_hidden_layers]):
        attention_mask = _layer_causal_mask(
            query_tokens=query_tokens,
            past_tokens=cache.get_seq_length(layer_index),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    return model.lm_head(core.norm(hidden_states))


def greedy_sparse_completion(
    *,
    model,
    tokenizer,
    query_token_ids: Sequence[int],
    prefix_tokens: int,
    cache,
    max_tokens: int,
) -> tuple[str, float, float]:
    """Generate a short label completion and return TTFT plus total latency."""

    import torch

    device = next(model.parameters()).device
    query = torch.tensor([list(query_token_ids)], dtype=torch.long, device=device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        logits = sparse_decoder_logits(model, query, cache, prefix_tokens)
    torch.cuda.synchronize()
    first_token_time = time.perf_counter()
    generated = [int(logits[0, -1].argmax().item())]
    for step in range(1, max_tokens):
        if tokenizer.eos_token_id is not None and generated[-1] == tokenizer.eos_token_id:
            break
        next_input = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = sparse_decoder_logits(
                model,
                next_input,
                cache,
                prefix_tokens + len(query_token_ids) + step - 1,
            )
        generated.append(int(logits[0, -1].argmax().item()))
    torch.cuda.synchronize()
    end = time.perf_counter()
    return tokenizer.decode(generated, skip_special_tokens=True), (first_token_time - start) * 1000, (end - start) * 1000


def _load_layer_plan(plan_path: str | Path, uid: str) -> tuple[list[list[str]], dict[str, Any]]:
    payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    request_id = policy_request_id(uid)
    plan = payload.get("layer_request_prefixes", {}).get(request_id)
    if not isinstance(plan, list) or not plan:
        raise ValueError(f"no layer plan for {request_id} in {plan_path}")
    return plan, payload.get("metadata", {})


def run_sparse_reprefill(
    *,
    model_path: str,
    bundle_dir: str | Path,
    store_root: str | Path,
    plan_path: str | Path,
    tasks: Sequence[str],
    samples_per_task: int,
    output_dir: str | Path,
    max_tokens: int,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    """Run one sparse physical-chunk policy against prepared prefix stores."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device).eval()
    rows = _load_bundle_records(bundle_dir, tasks, samples_per_task)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scored = []

    for row in rows:
        task = str(row["task"])
        info = read_store_info(store_root, task)
        layer_plan, plan_metadata = _load_layer_plan(plan_path, str(row["uid"]))
        chunk_size = int(plan_metadata.get("chunk_size", 0))
        if chunk_size <= 0:
            raise ValueError("plan metadata must declare a positive chunk_size")
        query_ids = tokenizer(str(row["query_text"]), add_special_tokens=False).input_ids
        cache, materialization = materialize_sparse_cache(
            store_root=store_root,
            task=task,
            layer_plan=layer_plan,
            chunk_size=chunk_size,
            device=device,
            model_config=model.config,
        )
        prediction, forward_ttft_ms, forward_latency_ms = greedy_sparse_completion(
            model=model,
            tokenizer=tokenizer,
            query_token_ids=query_ids,
            prefix_tokens=info.prefix_tokens,
            cache=cache,
            max_tokens=max_tokens,
        )
        ttft_ms = materialization["ssd_read_ms"] + materialization["h2d_ms"] + materialization["cache_build_ms"] + forward_ttft_ms
        correct = prediction_is_correct(prediction, str(row["answer"]))
        scored.append(
            {
                "uid": row["uid"],
                "task": task,
                "answer": row["answer"],
                "prediction": prediction,
                "correct": correct,
                "ttft_ms": ttft_ms,
                "latency_ms": materialization["ssd_read_ms"]
                + materialization["h2d_ms"]
                + materialization["cache_build_ms"]
                + forward_latency_ms,
                **materialization,
                "forward_ttft_ms": forward_ttft_ms,
            }
        )
        del cache
        torch.cuda.empty_cache()

    by_task = {}
    for task in tasks:
        task_rows = [row for row in scored if row["task"] == task]
        ttfts = [float(row["ttft_ms"]) for row in task_rows]
        by_task[task] = {
            "samples": len(task_rows),
            "accuracy": sum(row["correct"] for row in task_rows) / max(1, len(task_rows)),
            "mean_ttft_ms": sum(ttfts) / max(1, len(ttfts)),
            "p95_ttft_ms": percentile95(ttfts),
            "mean_selected_kv_bytes": sum(row["selected_kv_bytes"] for row in task_rows)
            / max(1, len(task_rows)),
        }
    ttfts = [float(row["ttft_ms"]) for row in scored]
    summary = {
        "model_path": model_path,
        "plan": str(plan_path),
        "tasks": by_task,
        "overall": {
            "samples": len(scored),
            "accuracy": sum(row["correct"] for row in scored) / max(1, len(scored)),
            "mean_ttft_ms": sum(ttfts) / max(1, len(ttfts)),
            "p95_ttft_ms": percentile95(ttfts),
        },
        "measurement": "SSD chunk read plus H2D transfer plus per-layer sparse Qwen re-prefill",
    }
    (output / "scored_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in scored), encoding="utf-8"
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run true sparse per-layer Qwen re-prefill from SSD KV chunks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common_arguments(command_parser):
        command_parser.add_argument("--model-path", default="/data1/llm/Qwen/Qwen2.5-7B-Instruct")
        command_parser.add_argument("--bundle-dir", required=True)
        command_parser.add_argument("--store-root", required=True)
        command_parser.add_argument("--tasks", default="sst2,subj,trec,rte")
        command_parser.add_argument("--samples-per-task", type=int, default=4)
        command_parser.add_argument("--device", default="cuda")
        command_parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")

    prepare_parser = subparsers.add_parser("prepare")
    common_arguments(prepare_parser)
    prepare_parser.add_argument("--overwrite", action="store_true")

    run_parser = subparsers.add_parser("run")
    common_arguments(run_parser)
    run_parser.add_argument("--plan", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--max-tokens", type=int, default=4)

    args = parser.parse_args()
    tasks = [task.strip().lower() for task in args.tasks.split(",") if task.strip()]
    if args.command == "prepare":
        result = prepare_prefix_store(
            model_path=args.model_path,
            bundle_dir=args.bundle_dir,
            store_root=args.store_root,
            tasks=tasks,
            samples_per_task=args.samples_per_task,
            device=args.device,
            dtype=args.dtype,
            overwrite=args.overwrite,
        )
        print(json.dumps({task: info.__dict__ for task, info in result.items()}, indent=2))
        return 0
    result = run_sparse_reprefill(
        model_path=args.model_path,
        bundle_dir=args.bundle_dir,
        store_root=args.store_root,
        plan_path=args.plan,
        tasks=tasks,
        samples_per_task=args.samples_per_task,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        device=args.device,
        dtype=args.dtype,
    )
    print(json.dumps(result["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
