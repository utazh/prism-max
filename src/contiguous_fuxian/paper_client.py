"""Serve paper-style shared-prefix bundles against one LMCache precision policy."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .lmcache_plan import policy_request_id


def normalize_label(value: str) -> str:
    return " ".join("".join(char if char.isalnum() or char == "_" else " " for char in value.lower()).split())


def prediction_is_correct(prediction: str, answer: str) -> bool:
    expected = normalize_label(answer)
    observed = normalize_label(prediction)
    return (
        observed == expected
        or observed.startswith(expected + " ")
        or expected in observed.split()
    )


def continuation_token_ids(
    tokenizer: Any,
    labels: Sequence[str],
    *,
    continuation_prefix: str = "",
) -> dict[str, tuple[int, ...]]:
    """Tokenize complete candidate labels as prompt continuations."""

    normalized = tuple(str(label) for label in labels)
    if not normalized:
        raise ValueError("label-logit scoring requires at least one label")
    if len(set(normalized)) != len(normalized):
        raise ValueError("label-logit scoring requires unique labels")
    token_ids: dict[str, tuple[int, ...]] = {}
    for label in normalized:
        ids = tuple(
            int(token)
            for token in tokenizer(
                continuation_prefix + label,
                add_special_tokens=False,
            ).input_ids
        )
        if not ids:
            raise ValueError(f"label {label!r} has no continuation tokens")
        token_ids[label] = ids
    return token_ids


def label_token_ids(
    tokenizer: Any,
    labels: Sequence[str],
    *,
    continuation_prefix: str = "",
) -> dict[str, tuple[int, ...]]:
    """Tokenize labels for the legacy one-step first-token scorer."""

    token_ids = continuation_token_ids(
        tokenizer,
        labels,
        continuation_prefix=continuation_prefix,
    )
    first_tokens = [ids[0] for ids in token_ids.values()]
    if len(set(first_tokens)) != len(first_tokens):
        raise ValueError(
            "one-step label-logit scoring requires labels with distinct first tokens"
        )
    return token_ids


def predict_from_label_logits(
    first_token_logits: Any,
    tokenizer: Any,
    labels: Sequence[str],
    *,
    continuation_prefix: str = "",
) -> tuple[str, dict[str, float], dict[str, tuple[int, ...]]]:
    """Choose the legal label whose first continuation token has highest logit."""

    token_ids = label_token_ids(
        tokenizer,
        labels,
        continuation_prefix=continuation_prefix,
    )
    scores = {
        label: float(first_token_logits[ids[0]].item())
        for label, ids in token_ids.items()
    }
    prediction = max(token_ids, key=lambda label: scores[label])
    return prediction, scores, token_ids


def predict_from_label_token_logprobs(
    token_logprobs: Mapping[str, Sequence[float]],
) -> tuple[str, dict[str, float]]:
    """Choose the candidate with the highest mean continuation log-probability."""

    if not token_logprobs:
        raise ValueError("label continuation scoring requires at least one candidate")
    scores: dict[str, float] = {}
    for label, values in token_logprobs.items():
        row = tuple(float(value) for value in values)
        if not row:
            raise ValueError(f"label {label!r} has no token log-probabilities")
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"label {label!r} has a non-finite token log-probability")
        scores[str(label)] = sum(row) / len(row)
    prediction = max(scores, key=scores.__getitem__)
    return prediction, scores


def percentile95(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _load_rows(bundle_dir: str | Path, tasks: list[str], samples_per_task: int) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        path = Path(bundle_dir) / f"{task}.jsonl"
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows.extend(lines[:samples_per_task])
    return rows


def _kv_params(policy: str, uid: str, skip_save: bool) -> dict[str, Any]:
    params = {
        "lmcache.precision_policy": policy,
        "lmcache.policy_request_id": policy_request_id(uid),
    }
    if skip_save:
        params["lmcache.skip_save"] = True
    return params


def wait_server(base_url: str, timeout_s: float) -> None:
    import requests

    deadline = time.monotonic() + timeout_s
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            response = requests.get(base_url.rstrip("/") + "/models", timeout=5)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:160]}"
        except Exception as exc:  # pragma: no cover - depends on server startup timing
            last_error = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"server did not become ready: {last_error}")


def stream_completion(
    base_url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    timeout_s: float,
    request_id: str,
    kv_transfer_params: dict[str, Any],
) -> dict[str, Any]:
    import requests

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "request_id": request_id,
        "kv_transfer_params": kv_transfer_params,
    }
    start = time.perf_counter()
    first_token = None
    pieces: list[str] = []
    response = requests.post(
        base_url.rstrip("/") + "/completions", json=payload, stream=True, timeout=timeout_s
    )
    try:
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            raw = raw_line[5:].strip()
            if raw == "[DONE]":
                break
            event = json.loads(raw)
            text = event.get("choices", [{}])[0].get("text", "") or ""
            if text and first_token is None:
                first_token = time.perf_counter()
            pieces.append(text)
    finally:
        response.close()
    end = time.perf_counter()
    return {
        "text": "".join(pieces),
        "ttft_ms": None if first_token is None else (first_token - start) * 1000,
        "latency_ms": (end - start) * 1000,
    }


def run_paper_client(
    *,
    base_url: str,
    model: str,
    bundle_dir: str | Path,
    tasks: list[str],
    samples_per_task: int,
    policy: str,
    output_dir: str | Path,
    max_tokens: int = 4,
    request_timeout_s: float = 300.0,
    server_timeout_s: float = 900.0,
    post_warm_sleep_s: float = 0.0,
) -> dict[str, Any]:
    """Warm each matching prefix then measure its query re-prefill TTFT."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(bundle_dir, tasks, samples_per_task)
    wait_server(base_url, server_timeout_s)
    scored = []
    warm = []
    for index, row in enumerate(rows):
        uid = str(row["uid"])
        warm_result = stream_completion(
            base_url,
            model,
            str(row["prefix_text"]),
            max_tokens=1,
            timeout_s=request_timeout_s,
            request_id=f"cmpl-{uid}-warm",
            kv_transfer_params=_kv_params(policy, uid, skip_save=False),
        )
        warm.append({"uid": uid, "task": row["task"], **warm_result})
        if post_warm_sleep_s > 0:
            time.sleep(post_warm_sleep_s)
        score_result = stream_completion(
            base_url,
            model,
            str(row["prefix_text"]) + str(row["query_text"]),
            max_tokens=max_tokens,
            timeout_s=request_timeout_s,
            request_id=f"cmpl-{uid}-score",
            kv_transfer_params=_kv_params(policy, uid, skip_save=True),
        )
        correct = prediction_is_correct(score_result["text"], str(row["answer"]))
        scored.append(
            {
                "uid": uid,
                "task": row["task"],
                "answer": row["answer"],
                "prediction": score_result["text"],
                "correct": correct,
                "ttft_ms": score_result["ttft_ms"],
                "latency_ms": score_result["latency_ms"],
            }
        )
        print(f"score {index + 1}/{len(rows)} {uid} correct={correct} ttft_ms={score_result['ttft_ms']}", flush=True)

    by_task = {}
    for task in tasks:
        task_rows = [row for row in scored if row["task"] == task]
        ttfts = [row["ttft_ms"] for row in task_rows if row["ttft_ms"] is not None]
        by_task[task] = {
            "samples": len(task_rows),
            "accuracy": sum(row["correct"] for row in task_rows) / max(1, len(task_rows)),
            "mean_ttft_ms": sum(ttfts) / max(1, len(ttfts)),
            "p95_ttft_ms": percentile95(ttfts),
        }
    ttfts = [row["ttft_ms"] for row in scored if row["ttft_ms"] is not None]
    summary = {
        "policy": policy,
        "model": model,
        "tasks": by_task,
        "overall": {
            "samples": len(scored),
            "accuracy": sum(row["correct"] for row in scored) / max(1, len(scored)),
            "mean_ttft_ms": sum(ttfts) / max(1, len(ttfts)),
            "p95_ttft_ms": percentile95(ttfts),
        },
        "measurement": "prefix warmed per request, then score request TTFT",
    }
    (output / "warm_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in warm), encoding="utf-8"
    )
    (output / "scored_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in scored), encoding="utf-8"
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one paper-aligned LMCache serving policy.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen2.5-7b")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--tasks", default="sst2,subj,trec,rte")
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--policy", choices=("paper-contigkv", "paper-impress"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--server-timeout-s", type=float, default=900.0)
    parser.add_argument("--post-warm-sleep-s", type=float, default=0.0)
    args = parser.parse_args()
    summary = run_paper_client(
        base_url=args.base_url,
        model=args.model,
        bundle_dir=args.bundle_dir,
        tasks=[task.strip().lower() for task in args.tasks.split(",") if task.strip()],
        samples_per_task=args.samples_per_task,
        policy=args.policy,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        request_timeout_s=args.request_timeout_s,
        server_timeout_s=args.server_timeout_s,
        post_warm_sleep_s=args.post_warm_sleep_s,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
