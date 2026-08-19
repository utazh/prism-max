"""Generate matched ContiguousKV and IMPRESS plans for paper-style task bundles."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from .core import contiguous_chunk_scores, select_period_chunks
from .lmcache_plan import policy_request_id


def top_indices(scores: Sequence[float], keep: int) -> set[int]:
    keep = max(0, min(len(scores), keep))
    return set(sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))[:keep])


def jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def mean_pairwise_jaccard(sets: Sequence[set[int]]) -> float:
    if len(sets) < 2:
        return 1.0
    values = [jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    return sum(values) / len(values)


def impress_similarity_threshold(total_tokens: int, keep_tokens: int, alpha: float) -> float:
    """Compute IMPRESS's data-dependent probe-head similarity threshold.

    IMPRESS derives the expected Jaccard value of two independently sampled
    ``keep_tokens`` sets and raises it to an empirical exponent of 0.6.  This
    avoids applying a fixed threshold to every KV retention ratio.
    """

    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    if not 0 < keep_tokens <= total_tokens:
        raise ValueError("keep_tokens must be in (0, total_tokens]")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    expected_jaccard = keep_tokens / (2 * total_tokens - keep_tokens)
    return expected_jaccard**alpha


def impress_probe_token_selection(
    head_scores: Sequence[Sequence[float]],
    keep_tokens: int,
    probe_heads: Sequence[int],
    similarity_alpha: float,
) -> tuple[set[int], bool, float]:
    """Reimplement the paper-visible IMPRESS probe-head selection path."""

    selected, _, used_probe, similarity = impress_probe_token_selection_with_ranking(
        head_scores,
        keep_tokens,
        probe_heads,
        similarity_alpha,
    )
    return selected, used_probe, similarity


def impress_probe_token_selection_with_ranking(
    head_scores: Sequence[Sequence[float]],
    keep_tokens: int,
    probe_heads: Sequence[int],
    similarity_alpha: float,
) -> tuple[set[int], list[int], bool, float]:
    """Return the unchanged IMPRESS set plus a value-ranked prefetch order."""

    valid_heads = [head for head in probe_heads if 0 <= head < len(head_scores)]
    if not valid_heads:
        valid_heads = [0]
    probe_sets = [top_indices(head_scores[head], keep_tokens) for head in valid_heads]
    similarity = mean_pairwise_jaccard(probe_sets)
    threshold = impress_similarity_threshold(len(head_scores[0]), keep_tokens, similarity_alpha)
    mean_scores = [
        sum(float(head_scores[head][token]) for head in valid_heads)
        / len(valid_heads)
        for token in range(len(head_scores[0]))
    ]
    if similarity < threshold:
        selected = set(range(len(head_scores[0])))
        priority = sorted(
            selected,
            key=lambda token: (-mean_scores[token], token),
        )
        return selected, priority, False, similarity
    counts: dict[int, int] = {}
    for selected in probe_sets:
        for token in selected:
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts, key=lambda token: (-counts[token], token))
    selected = set(ranked[:keep_tokens])
    priority = sorted(
        selected,
        key=lambda token: (-counts[token], -mean_scores[token], token),
    )
    return selected, priority, True, similarity


def tokens_to_chunks(tokens: Iterable[int], chunk_size: int, length: int) -> set[int]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if length <= 0:
        return set()
    last_chunk = math.ceil(length / chunk_size) - 1
    return {
        min(last_chunk, max(0, int(token) // chunk_size))
        for token in tokens
        if 0 <= int(token) < length
    }


def build_sparse_layer_plan(
    layer_chunks: Sequence[set[int]],
    num_chunks: int,
    selected_tiers: str | Sequence[str] = "int8",
) -> list[list[str]]:
    """Build layer plans using the same codecs used when LMCache warms KV.

    The existing LMCache MP storage path writes layer 0 as INT8 and the
    remaining layers as INT4 by default. Retrieval must request the exact
    same object codec or the selected layer/chunk appears as an L2 miss.
    """

    if isinstance(selected_tiers, str):
        tiers = [selected_tiers] * len(layer_chunks)
    else:
        tiers = [str(tier) for tier in selected_tiers]
    if len(tiers) != len(layer_chunks):
        raise ValueError("selected_tiers must provide one codec per layer")
    if any(tier not in {"int8", "int4"} for tier in tiers):
        raise ValueError("selected_tiers must contain only int8 or int4")
    return [
        [tiers[layer] if chunk in selected else "drop" for chunk in range(num_chunks)]
        for layer, selected in enumerate(layer_chunks)
    ]


def build_lmcache_payload(
    method: str,
    plans: dict[str, list[list[str]]],
    records_meta: list[dict[str, Any]],
    metadata: dict[str, Any],
    layer_token_selections: dict[str, list[list[int]]] | None = None,
    layer_chunk_attention_scores: dict[str, list[list[float]]] | None = None,
) -> dict[str, Any]:
    request_prefixes: dict[str, list[str]] = {}
    for request_id, layer_plan in plans.items():
        request_prefixes[request_id] = [
            "base" if any(row[chunk] != "drop" for row in layer_plan) else "drop"
            for chunk in range(len(layer_plan[0]))
        ]
    tiers = Counter(tier for layer_plan in plans.values() for row in layer_plan for tier in row)
    payload = {
        "default": ["base"],
        "request_prefixes": request_prefixes,
        "layer_default": [],
        "layer_request_prefixes": plans,
        "metadata": {
            "method": method,
            "runtime_tier_counts": dict(tiers),
            "records": records_meta,
            **metadata,
        },
    }
    if layer_token_selections is not None:
        payload["layer_token_selections"] = layer_token_selections
    if layer_chunk_attention_scores is not None:
        payload["layer_chunk_attention_scores"] = layer_chunk_attention_scores
    return payload


def load_bundle_records(bundle_dir: str | Path, tasks: Sequence[str], samples_per_task: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = Path(bundle_dir)
    for task in tasks:
        path = root / f"{task}.jsonl"
        task_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not task_rows:
            raise ValueError(f"no records in {path}")
        rows.extend(task_rows[:samples_per_task])
    return rows


def _head_prefix_scores(
    attentions: Any,
    prefix_len: int,
    query_len: int,
) -> list[list[list[float]]]:
    """Aggregate each head's prefix attention across all query suffix tokens."""

    scores = []
    for layer, attention in enumerate(attentions):
        if attention is None:
            raise RuntimeError("query attention was not returned; eager attention is required")
        query_to_prefix = attention[0, :, -query_len:, :prefix_len].float().sum(dim=1).detach().cpu()
        finite = query_to_prefix.isfinite()
        if not bool(finite.all()):
            invalid = int((~finite).sum().item())
            raise RuntimeError(
                f"layer {layer} query-to-prefix attention contains {invalid} non-finite values; "
                "use bfloat16 for Qwen2.5 plan generation"
            )
        scores.append(query_to_prefix.tolist())
    return scores


def _average_heads(head_scores: Sequence[Sequence[float]]) -> list[float]:
    if not head_scores:
        return []
    values = [0.0] * len(head_scores[0])
    for head in head_scores:
        for index, value in enumerate(head):
            values[index] += float(value)
    return [value / len(head_scores) for value in values]


def generate_matched_plans(
    *,
    model_path: str,
    bundle_dir: str | Path,
    tasks: Sequence[str],
    samples_per_task: int,
    keep_ratio: float,
    contiguous_chunk_size: int,
    impress_chunk_size: int,
    period_size: int,
    subperiod_size: int,
    probe_heads: Sequence[int],
    similarity_alpha: float,
    int8_layers: Sequence[int],
    max_prompt_tokens: int,
    device: str,
    allow_gpu: bool,
    dtype: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one attention observation per query and emit matched runtime plans."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda") and not allow_gpu:
        raise ValueError("cuda requires --allow-gpu on a shared server")
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    if subperiod_size <= 0:
        raise ValueError("subperiod_size must be positive")
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device).eval()

    rows = load_bundle_records(bundle_dir, tasks, samples_per_task)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["task"]), []).append(row)

    contiguous_plans: dict[str, list[list[str]]] = {}
    contiguous_score_plans: dict[str, list[list[float]]] = {}
    impress_plans: dict[str, list[list[str]]] = {}
    impress_token_plans: dict[str, list[list[int]]] = {}
    contiguous_meta: list[dict[str, Any]] = []
    impress_meta: list[dict[str, Any]] = []

    torch.set_grad_enabled(False)
    for task, task_rows in grouped.items():
        prefix_text = str(task_rows[0]["prefix_text"])
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
        if len(prefix_ids) > max_prompt_tokens:
            raise ValueError(f"{task} prefix has {len(prefix_ids)} tokens, exceeds {max_prompt_tokens}")
        prefix_inputs = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            prefix_output = model(input_ids=prefix_inputs, use_cache=True)
        prefix_len = len(prefix_ids)
        prefix_cache = prefix_output.past_key_values
        for row in task_rows:
            query_ids = tokenizer(str(row["query_text"]), add_special_tokens=False).input_ids
            if not query_ids:
                raise ValueError(f"query is empty for {row['uid']}")
            if prefix_len + len(query_ids) > max_prompt_tokens:
                raise ValueError(f"{row['uid']} exceeds max_prompt_tokens")
            query_inputs = torch.tensor([query_ids], dtype=torch.long, device=device)
            mask = torch.ones((1, prefix_len + len(query_ids)), dtype=torch.long, device=device)
            previous_impl = getattr(model.config, "_attn_implementation", None)
            if hasattr(model, "set_attn_implementation"):
                model.set_attn_implementation("eager")
            else:
                model.config._attn_implementation = "eager"
            query_cache = copy.deepcopy(prefix_cache)
            try:
                with torch.inference_mode():
                    query_output = model(
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

            if int(prefix_cache.get_seq_length()) != prefix_len:
                raise RuntimeError("shared prefix cache was mutated while generating a query plan")

            layer_heads = _head_prefix_scores(query_output.attentions, prefix_len, len(query_ids))
            layer_codecs = [
                "int8" if layer in set(int8_layers) else "int4"
                for layer in range(len(layer_heads))
            ]
            contiguous_chunk_count = math.ceil(prefix_len / contiguous_chunk_size)
            contiguous_keep = max(1, math.ceil(contiguous_chunk_count * keep_ratio))
            contiguous_scores = [
                contiguous_chunk_scores(_average_heads(heads), contiguous_chunk_size, prefix_len)
                for heads in layer_heads
            ]
            contiguous_selected = select_period_chunks(
                contiguous_scores, keep_chunks=contiguous_keep, period_size=period_size
            )
            impress_keep_tokens = max(1, math.ceil(prefix_len * keep_ratio))
            impress_selected_tokens = []
            impress_probe_used = []
            impress_similarity = []
            for heads in layer_heads:
                selected, used_probe, similarity = impress_probe_token_selection(
                    heads, impress_keep_tokens, probe_heads, similarity_alpha
                )
                impress_selected_tokens.append(selected)
                impress_probe_used.append(used_probe)
                impress_similarity.append(similarity)
            impress_chunk_count = math.ceil(prefix_len / impress_chunk_size)
            impress_selected = [
                tokens_to_chunks(selected, impress_chunk_size, prefix_len)
                for selected in impress_selected_tokens
            ]

            request_id = policy_request_id(str(row["uid"]))
            contiguous_plans[request_id] = build_sparse_layer_plan(
                contiguous_selected, contiguous_chunk_count, layer_codecs
            )
            contiguous_score_plans[request_id] = [
                [float(score) for score in scores] for scores in contiguous_scores
            ]
            impress_plans[request_id] = build_sparse_layer_plan(
                impress_selected, impress_chunk_count, layer_codecs
            )
            impress_token_plans[request_id] = [
                sorted(selected) for selected in impress_selected_tokens
            ]
            common = {
                "uid": row["uid"],
                "request_id": request_id,
                "task": task,
                "prefix_tokens": prefix_len,
                "query_tokens": len(query_ids),
                "layers": len(layer_heads),
            }
            contiguous_meta.append(
                {
                    **common,
                    "chunks": contiguous_chunk_count,
                    "kept_chunks_per_layer": contiguous_keep,
                    "period_size": period_size,
                }
            )
            impress_meta.append(
                {
                    **common,
                    "chunks": impress_chunk_count,
                    "keep_tokens_per_layer": impress_keep_tokens,
                    "mean_probe_jaccard": sum(impress_similarity) / max(1, len(impress_similarity)),
                    "probe_selected_layers": sum(impress_probe_used),
                    "probe_fallback_layers": len(impress_probe_used) - sum(impress_probe_used),
                    "similarity_threshold": impress_similarity_threshold(
                        prefix_len, impress_keep_tokens, similarity_alpha
                    ),
                }
            )
            del query_output, query_cache, query_inputs, mask
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        del prefix_output, prefix_inputs
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    base_metadata = {
        "model_path": model_path,
        "model_compute_dtype": dtype,
        "tasks": list(tasks),
        "samples_per_task": samples_per_task,
        "keep_ratio": keep_ratio,
        "selection_source": "Qwen2.5 query-suffix attention over paper-style shared prefixes",
        "query_cache_isolated": True,
        "lmcache_int8_layers": list(int8_layers),
        "lmcache_selected_layer_codecs": "int8 for configured sensitive layers; int4 otherwise",
    }
    contiguous = build_lmcache_payload(
        "contigkv",
        contiguous_plans,
        contiguous_meta,
        {
            **base_metadata,
            "chunk_size": contiguous_chunk_size,
            "period_size": period_size,
            "subperiod_size": subperiod_size,
        },
        layer_chunk_attention_scores=contiguous_score_plans,
    )
    impress = build_lmcache_payload(
        "impress",
        impress_plans,
        impress_meta,
        {
            **base_metadata,
            "chunk_size": impress_chunk_size,
            "probe_heads": list(probe_heads),
            "similarity_alpha": similarity_alpha,
        },
        layer_token_selections=impress_token_plans,
    )
    return contiguous, impress


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate matched ContiguousKV and IMPRESS LMCache plans.")
    parser.add_argument("--model-path", default="/data1/llm/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--tasks", default="sst2,subj,trec,rte")
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--keep-ratio", type=float, required=True)
    parser.add_argument("--contiguous-output", required=True)
    parser.add_argument("--impress-output", required=True)
    parser.add_argument("--contiguous-chunk-size", type=int, default=16)
    parser.add_argument("--impress-chunk-size", type=int, default=64)
    parser.add_argument("--period-size", type=int, default=8)
    parser.add_argument("--subperiod-size", type=int, default=4)
    parser.add_argument("--probe-heads", default="0,1,2")
    parser.add_argument("--similarity-alpha", type=float, default=0.6)
    parser.add_argument("--int8-layers", default="0")
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    tasks = [task.strip().lower() for task in args.tasks.split(",") if task.strip()]
    probe_heads = [int(value) for value in args.probe_heads.split(",") if value.strip()]
    int8_layers = [int(value) for value in args.int8_layers.split(",") if value.strip()]
    contiguous, impress = generate_matched_plans(
        model_path=args.model_path,
        bundle_dir=args.bundle_dir,
        tasks=tasks,
        samples_per_task=args.samples_per_task,
        keep_ratio=args.keep_ratio,
        contiguous_chunk_size=args.contiguous_chunk_size,
        impress_chunk_size=args.impress_chunk_size,
        period_size=args.period_size,
        subperiod_size=args.subperiod_size,
        probe_heads=probe_heads,
        similarity_alpha=args.similarity_alpha,
        int8_layers=int8_layers,
        max_prompt_tokens=args.max_prompt_tokens,
        device=args.device,
        allow_gpu=args.allow_gpu,
        dtype=args.dtype,
    )
    for path, payload in ((Path(args.contiguous_output), contiguous), (Path(args.impress_output), impress)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "contiguous_tiers": contiguous["metadata"]["runtime_tier_counts"],
        "impress_tiers": impress["metadata"]["runtime_tier_counts"],
        "records": len(contiguous["metadata"]["records"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
