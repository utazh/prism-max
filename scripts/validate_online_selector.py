#!/usr/bin/env python3
"""Validate the online Qwen selector against eager full attention."""

from __future__ import annotations

import argparse
import json
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from contiguous_fuxian.core import contiguous_chunk_scores, select_top_chunks
from contiguous_fuxian.flexgen_qwen_reprefill import qwen_online_prefix_head_scores


def _input_ids(tokenizer, total_tokens: int, device: str) -> torch.Tensor:
    text = (
        "Contiguous KV cache selection must preserve rotary positions and grouped query "
        "attention while loading only the critical prefix blocks. "
    )
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    repeats = math.ceil(total_tokens / max(1, len(token_ids)))
    return torch.tensor([((token_ids * repeats)[:total_tokens])], device=device)


def _max_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    error = (actual.float() - expected.float()).abs()
    return float(error.max().item()), float(error.mean().item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix-tokens", type=int, default=48)
    parser.add_argument("--query-tokens", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--keep-ratio", type=float, default=0.25)
    parser.add_argument("--probe-query-heads", default="0,1,2")
    parser.add_argument(
        "--pcache-storage-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="Round-trip selector keys through the configured Pcache payload dtype.",
    )
    parser.add_argument("--max-absolute-error", type=float, default=0.01)
    args = parser.parse_args()
    if args.prefix_tokens <= 0 or args.query_tokens <= 0 or args.chunk_size <= 0:
        raise ValueError("token counts and chunk size must be positive")
    if not 0 < args.keep_ratio <= 1:
        raise ValueError("keep ratio must be in (0, 1]")
    probe_heads = tuple(int(item) for item in args.probe_query_heads.split(","))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(args.device).eval()
    total_tokens = args.prefix_tokens + args.query_tokens
    input_ids = _input_ids(tokenizer, total_tokens, args.device)

    with torch.inference_mode():
        full_output = model(
            input_ids,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        prefix_cache = DynamicCache(config=model.config)
        model(
            input_ids[:, : args.prefix_tokens],
            past_key_values=prefix_cache,
            use_cache=True,
            return_dict=True,
        )
        core = model.model
        decoder_layer = core.layers[0]
        query_ids = input_ids[:, args.prefix_tokens :]
        hidden_states = core.embed_tokens(query_ids)
        positions = torch.arange(args.prefix_tokens, total_tokens, device=args.device).unsqueeze(0)
        position_embeddings = core.rotary_emb(hidden_states, positions)
        storage_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[args.pcache_storage_dtype]
        selector_keys = (
            prefix_cache.layers[0]
            .keys[0]
            .permute(1, 0, 2)
            .contiguous()
            .to(storage_dtype)
        )

        online_all = qwen_online_prefix_head_scores(
            decoder_layer=decoder_layer,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            selector_keys=selector_keys,
            query_heads=None,
        )
        reference_all = full_output.attentions[0][
            0, :, args.prefix_tokens :, : args.prefix_tokens
        ].sum(dim=1).cpu()

        groups = model.config.num_attention_heads // model.config.num_key_value_heads
        probe_kv_heads = [head // groups for head in probe_heads]
        unique_probe_kv_heads = list(dict.fromkeys(probe_kv_heads))
        online_probe = qwen_online_prefix_head_scores(
            decoder_layer=decoder_layer,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            selector_keys=selector_keys[:, probe_kv_heads, :],
            query_heads=probe_heads,
        )
        online_probe_deduplicated = qwen_online_prefix_head_scores(
            decoder_layer=decoder_layer,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            selector_keys=selector_keys[:, unique_probe_kv_heads, :],
            query_heads=probe_heads,
            selector_kv_head_ids=unique_probe_kv_heads,
        )
        reference_probe = reference_all[list(probe_heads)]

    all_max, all_mean = _max_error(online_all, reference_all)
    probe_max, probe_mean = _max_error(online_probe, reference_probe)
    deduplicated_probe_max, deduplicated_probe_mean = _max_error(
        online_probe_deduplicated, reference_probe
    )
    duplicate_dedup_max, duplicate_dedup_mean = _max_error(
        online_probe_deduplicated, online_probe
    )
    online_chunks = contiguous_chunk_scores(
        online_all.mean(dim=0).tolist(), args.chunk_size, args.prefix_tokens
    )
    reference_chunks = contiguous_chunk_scores(
        reference_all.mean(dim=0).tolist(), args.chunk_size, args.prefix_tokens
    )
    keep_chunks = max(1, math.ceil(len(online_chunks) * args.keep_ratio))
    online_selected = sorted(select_top_chunks(online_chunks, keep_chunks))
    reference_selected = sorted(select_top_chunks(reference_chunks, keep_chunks))
    report = {
        "model": args.model_path,
        "dtype": "bfloat16",
        "pcache_storage_dtype": args.pcache_storage_dtype,
        "prefix_tokens": args.prefix_tokens,
        "query_tokens": args.query_tokens,
        "all_heads_max_absolute_error": all_max,
        "all_heads_mean_absolute_error": all_mean,
        "probe_heads": list(probe_heads),
        "probe_kv_heads": probe_kv_heads,
        "unique_probe_kv_heads": unique_probe_kv_heads,
        "probe_max_absolute_error": probe_max,
        "probe_mean_absolute_error": probe_mean,
        "deduplicated_probe_max_absolute_error": deduplicated_probe_max,
        "deduplicated_probe_mean_absolute_error": deduplicated_probe_mean,
        "duplicated_vs_deduplicated_max_absolute_error": duplicate_dedup_max,
        "duplicated_vs_deduplicated_mean_absolute_error": duplicate_dedup_mean,
        "online_selected_chunks": online_selected,
        "reference_selected_chunks": reference_selected,
        "selected_chunks_match": online_selected == reference_selected,
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    if max(all_max, probe_max, deduplicated_probe_max) > args.max_absolute_error:
        raise RuntimeError("online selector differs from eager attention beyond tolerance")
    if duplicate_dedup_max != 0.0:
        raise RuntimeError("deduplicating GQA selector keys changed probe scores")
    if online_selected != reference_selected:
        raise RuntimeError("online selector and eager attention chose different chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
