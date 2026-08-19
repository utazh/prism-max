"""Qwen2.5 attention probe for ContiguousKV reproduction.

This optional script requires `torch` and `transformers`. It loads a local
Qwen-style causal LM, extracts per-layer attention over a prompt, converts
token scores to ContiguousChunk scores, and writes a layer/chunk plan.
"""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
from typing import Any

from .core import (
    AttentionGuidedCache,
    build_layer_chunk_plan,
    contiguous_chunk_scores,
    jaccard_similarity,
    select_period_chunks,
)
from .simulator import KVShape, LayerRequest, compare_impress_contiguous


DEFAULT_PROMPTS = [
    (
        "Classify the sentiment. Answer with positive or negative.\n\n"
        "Sentence: a warm , funny , engaging film\nAnswer: positive\n\n"
        "Sentence: a dull and predictable story\nAnswer: negative\n\n"
        "Sentence: the acting is sincere and the pacing is brisk\nAnswer:"
    ),
    (
        "Question type classification. Answer with abbreviation.\n\n"
        "Question: What city is the Eiffel Tower in?\nAnswer: LOC\n\n"
        "Question: Who wrote Hamlet?\nAnswer: HUM\n\n"
        "Question: How many planets are in the solar system?\nAnswer:"
    ),
]


def _period_heads(layer_chunks: list[set[int]], period_size: int) -> list[set[int]]:
    return [set(layer_chunks[idx]) for idx in range(0, len(layer_chunks), period_size)]


def _mean_adjacent_jaccard(period_sets: list[set[int]]) -> float:
    if len(period_sets) < 2:
        return 1.0
    return sum(
        jaccard_similarity(period_sets[idx - 1], period_sets[idx])
        for idx in range(1, len(period_sets))
    ) / (len(period_sets) - 1)


def _load_prompts(path: str | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(line.rstrip("\n"))
    if not rows:
        raise ValueError(f"no prompts found in {path}")
    return rows


def split_token_ids_for_prefill_query(
    token_ids: list[int],
    query_tail_tokens: int,
) -> tuple[list[int], list[int]]:
    """Split token ids into a prefix cache segment and a query tail."""

    if not token_ids:
        raise ValueError("token_ids must not be empty")
    if query_tail_tokens <= 0:
        raise ValueError("query_tail_tokens must be positive")
    split_at = max(1, len(token_ids) - query_tail_tokens)
    return token_ids[:split_at], token_ids[split_at:]


def run_probe(
    model_path: str,
    output: str | Path,
    prompts_path: str | None = None,
    max_prompts: int = 1,
    contiguous_chunk_size: int = 16,
    keep_ratio: float = 0.05,
    period_size: int = 8,
    subperiod_size: int = 4,
    max_prompt_tokens: int = 2048,
    attention_mode: str = "prefill_query",
    query_tail_tokens: int = 1,
    attn_implementation: str = "sdpa",
    query_attn_implementation: str = "eager",
    dtype: str = "bfloat16",
    device: str = "cpu",
    allow_gpu: bool = False,
    chunk_load_ms: float = 0.08,
    compute_ms: float = 1.0,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda") and not allow_gpu:
        raise ValueError("cuda device requires explicit allow_gpu=True")
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    if attention_mode not in {"prefill_query", "full"}:
        raise ValueError("attention_mode must be 'prefill_query' or 'full'")
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    records = []
    for prompt_idx, prompt in enumerate(_load_prompts(prompts_path)[:max_prompts]):
        token_ids = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        prompt_tokens = len(token_ids)
        if prompt_tokens > max_prompt_tokens:
            raise ValueError(
                f"prompt {prompt_idx} has {prompt_tokens} tokens, exceeding "
                f"max_prompt_tokens={max_prompt_tokens}"
            )
        if attention_mode == "prefill_query":
            prefix_token_ids, query_token_ids = split_token_ids_for_prefill_query(
                token_ids,
                query_tail_tokens,
            )
            prefix_len = len(prefix_token_ids)
            query_len = len(query_token_ids)
            prefix_inputs = torch.tensor([prefix_token_ids], dtype=torch.long, device=device)
            query_inputs = torch.tensor([query_token_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones(
                (1, prefix_len + query_len),
                dtype=torch.long,
                device=device,
            )
            with torch.inference_mode():
                prefix_out = model(input_ids=prefix_inputs, use_cache=True)
                previous_impl = getattr(model.config, "_attn_implementation", None)
                if hasattr(model, "set_attn_implementation"):
                    model.set_attn_implementation(query_attn_implementation)
                else:
                    model.config._attn_implementation = query_attn_implementation
                try:
                    out = model(
                        input_ids=query_inputs,
                        past_key_values=prefix_out.past_key_values,
                        attention_mask=attention_mask,
                        output_attentions=True,
                        use_cache=False,
                    )
                finally:
                    if previous_impl is not None:
                        if hasattr(model, "set_attn_implementation"):
                            model.set_attn_implementation(previous_impl)
                        else:
                            model.config._attn_implementation = previous_impl
        else:
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            prefix_len = int(inputs.input_ids.shape[1])
            query_len = 1
            with torch.inference_mode():
                out = model(**inputs, output_attentions=True, use_cache=False)
        num_chunks = ceil(prefix_len / contiguous_chunk_size)
        keep_chunks = max(1, ceil(num_chunks * keep_ratio))
        layer_chunk_scores = []
        for attn in out.attentions:
            # Shape is [batch, heads, query_pos, key_pos]. Use the final query's
            # distribution over prefix keys, averaged across heads.
            final_query_prefix = attn[0, :, -1, :prefix_len].float().mean(dim=0)
            scores = final_query_prefix.detach().cpu().tolist()
            layer_chunk_scores.append(
                contiguous_chunk_scores(scores, contiguous_chunk_size, prefix_len)
            )
        selected = select_period_chunks(
            layer_chunk_scores,
            keep_chunks=keep_chunks,
            period_size=period_size,
        )
        cache = AttentionGuidedCache(capacity=max(1, keep_chunks * max(1, subperiod_size)))
        evictions = 0
        for layer_idx, chunks in enumerate(selected):
            for chunk in chunks:
                evictions += len(
                    cache.touch(
                        (layer_idx, chunk),
                        layer_chunk_scores[layer_idx][chunk],
                    )
                )
        plan = build_layer_chunk_plan(num_chunks, selected)
        period_sets = _period_heads(selected, period_size)
        system_simulation = compare_impress_contiguous(
            [
                LayerRequest(layer=layer_idx, chunks=set(chunks))
                for layer_idx, chunks in enumerate(selected)
            ],
            shape=KVShape(
                num_layers=len(selected),
                prefix_tokens=prefix_len,
                contiguous_chunk_size=contiguous_chunk_size,
            ),
            impress_chunk_size=64,
            budget_ratio=keep_ratio,
            chunk_load_ms=chunk_load_ms,
            compute_ms=compute_ms,
            period_size=period_size,
            subperiod_size=subperiod_size,
        )
        system_simulation["speedup_vs_impress"] = (
            system_simulation["impress"]["estimated_ms"]
            / max(1e-9, system_simulation["contiguous"]["estimated_ms"])
        )
        records.append({
            "prompt_index": prompt_idx,
            "prompt_tokens": prompt_tokens,
            "prefix_tokens": prefix_len,
            "query_tokens": query_len,
            "num_chunks": num_chunks,
            "keep_chunks": keep_chunks,
            "mean_adjacent_period_jaccard": _mean_adjacent_jaccard(period_sets),
            "attention_cache": {
                "policy": "S=I*F",
                "capacity": cache.capacity,
                "resident_count": len(cache.resident_chunks()),
                "evictions": evictions,
            },
            "system_simulation": system_simulation,
            "layer_plan": plan,
        })
        if attention_mode == "prefill_query":
            del out, prefix_out, prefix_inputs, query_inputs, attention_mask
        else:
            del out, inputs
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    payload = {
        "model_path": model_path,
        "contiguous_chunk_size": contiguous_chunk_size,
        "keep_ratio": keep_ratio,
        "period_size": period_size,
        "subperiod_size": subperiod_size,
        "max_prompt_tokens": max_prompt_tokens,
        "attention_mode": attention_mode,
        "query_tail_tokens": query_tail_tokens,
        "attn_implementation": attn_implementation,
        "query_attn_implementation": query_attn_implementation,
        "chunk_load_ms": chunk_load_ms,
        "compute_ms": compute_ms,
        "records": records,
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe Qwen2.5 attention for ContiguousKV.")
    parser.add_argument("--model-path", default="/data1/llm/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", default="src/contiguous_fuxian/results/qwen_attention_probe.json")
    parser.add_argument("--prompts-path")
    parser.add_argument("--max-prompts", type=int, default=1)
    parser.add_argument("--contiguous-chunk-size", type=int, default=16)
    parser.add_argument("--keep-ratio", type=float, default=0.05)
    parser.add_argument("--period-size", type=int, default=8)
    parser.add_argument("--subperiod-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--attention-mode", choices=("prefill_query", "full"), default="prefill_query")
    parser.add_argument("--query-tail-tokens", type=int, default=1)
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--query-attn-implementation", choices=("eager",), default="eager")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--chunk-load-ms", type=float, default=0.08)
    parser.add_argument("--compute-ms", type=float, default=1.0)
    return parser


def validate_probe_args(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not args.allow_gpu:
        raise ValueError("Use --allow-gpu when --device is cuda on a shared server.")
    if args.max_prompt_tokens <= 0:
        raise ValueError("--max-prompt-tokens must be positive")
    if args.query_tail_tokens <= 0:
        raise ValueError("--query-tail-tokens must be positive")
    if args.chunk_load_ms < 0 or args.compute_ms < 0:
        raise ValueError("--chunk-load-ms and --compute-ms must be non-negative")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_probe_args(args)
    payload = run_probe(**vars(args))
    print(json.dumps({
        "records": len(payload["records"]),
        "first_mean_adjacent_period_jaccard": payload["records"][0]["mean_adjacent_period_jaccard"]
        if payload["records"] else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
