"""Build paper-style few-shot shared-prefix tasks without a datasets dependency."""

from __future__ import annotations

import argparse
import csv
import json
import random
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TaskSpec:
    name: str
    fewshot_examples: int
    paper_prefix_tokens: int


PAPER_TASKS = {
    "sst2": TaskSpec("sst2", fewshot_examples=100, paper_prefix_tokens=3800),
    "subj": TaskSpec("subj", fewshot_examples=110, paper_prefix_tokens=4400),
    "trec": TaskSpec("trec", fewshot_examples=120, paper_prefix_tokens=5000),
    "rte": TaskSpec("rte", fewshot_examples=80, paper_prefix_tokens=6000),
}

SOURCES = {
    "sst2": "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
    "rte": "https://dl.fbaipublicfiles.com/glue/data/RTE.zip",
    "subj": "https://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz",
    "trec_train": "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
    "trec_test": "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label",
}

QWEN_USER_PREFIX = "<|im_start|>user\n"
QWEN_ASSISTANT_PREFIX = "<|im_end|>\n<|im_start|>assistant\n"


def _clean_text(value: str) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def _download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def ensure_paper_datasets(data_root: str | Path) -> Path:
    """Download official raw inputs required by the four paper tasks."""

    root = Path(data_root)
    raw = root / "raw"
    _download(SOURCES["sst2"], raw / "SST-2.zip")
    _download(SOURCES["rte"], raw / "RTE.zip")
    _download(SOURCES["subj"], raw / "rotten_imdb.tar.gz")
    _download(SOURCES["trec_train"], raw / "train_5500.label")
    _download(SOURCES["trec_test"], raw / "TREC_10.label")

    if not (root / "SST-2").exists():
        with zipfile.ZipFile(raw / "SST-2.zip") as archive:
            archive.extractall(root)
    if not (root / "RTE").exists():
        with zipfile.ZipFile(raw / "RTE.zip") as archive:
            archive.extractall(root)
    subj_dir = root / "subj"
    if not subj_dir.exists():
        subj_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(raw / "rotten_imdb.tar.gz", "r:gz") as archive:
            archive.extractall(subj_dir)
    return root


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _read_trec(path: Path) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="latin-1").splitlines():
        label, text = line.split(" ", 1)
        rows.append({"text": _clean_text(text), "label": label.split(":", 1)[0]})
    return rows


def _find_member(root: Path, filename: str) -> Path:
    candidates = sorted(root.rglob(filename))
    if not candidates:
        raise FileNotFoundError(f"{filename} was not found under {root}")
    return candidates[0]


def _read_subj(root: Path) -> list[dict[str, str]]:
    subjective = _find_member(root, "quote.tok.gt9.5000")
    objective = _find_member(root, "plot.tok.gt9.5000")
    return [
        *({"text": _clean_text(line), "label": "subjective"} for line in subjective.read_text(encoding="latin-1").splitlines()),
        *({"text": _clean_text(line), "label": "objective"} for line in objective.read_text(encoding="latin-1").splitlines()),
    ]


def load_task_rows(task: str, data_root: str | Path, seed: int = 42) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return deterministic train/evaluation row splits for one paper task."""

    root = Path(data_root)
    task = task.lower()
    if task == "sst2":
        train = [
            {"text": _clean_text(row["sentence"]), "label": "positive" if row["label"] == "1" else "negative"}
            for row in _read_tsv(root / "SST-2" / "train.tsv")
        ]
        evaluation = [
            {"text": _clean_text(row["sentence"]), "label": "positive" if row["label"] == "1" else "negative"}
            for row in _read_tsv(root / "SST-2" / "dev.tsv")
        ]
        return train, evaluation
    if task == "rte":
        def convert(row: dict[str, str]) -> dict[str, str]:
            label = str(row["label"]).strip()
            return {
                "premise": _clean_text(row["sentence1"]),
                "hypothesis": _clean_text(row["sentence2"]),
                "label": "entailment" if label == "entailment" else "not_entailment",
            }

        return (
            [convert(row) for row in _read_tsv(root / "RTE" / "train.tsv")],
            [convert(row) for row in _read_tsv(root / "RTE" / "dev.tsv")],
        )
    if task == "trec":
        return _read_trec(root / "raw" / "train_5500.label"), _read_trec(root / "raw" / "TREC_10.label")
    if task == "subj":
        rows = _read_subj(root / "subj")
        rng = random.Random(seed)
        rng.shuffle(rows)
        split = int(len(rows) * 0.9)
        return rows[:split], rows[split:]
    raise ValueError(f"unsupported task: {task}")


def format_example(task: str, row: dict[str, str], include_label: bool) -> str:
    """Use a stable prompt template for the paper's few-shot classification workload."""

    task = task.lower()
    if task == "sst2":
        answer = f" {row['label']}" if include_label else ""
        return f"Sentence: {row['text']}\nSentiment:{answer}\n\n"
    if task == "subj":
        answer = f" {row['label']}" if include_label else ""
        return f"Sentence: {row['text']}\nSubjectivity:{answer}\n\n"
    if task == "trec":
        answer = f" {row['label']}" if include_label else ""
        return (
            "Labels: ABBR abbreviation, DESC description, ENTY entity, HUM human, "
            "LOC location, NUM numeric.\n"
            f"Question: {row['text']}\nType:{answer}\n\n"
        )
    if task == "rte":
        answer = f" {row['label']}" if include_label else ""
        return (
            f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}\n"
            f"Relation:{answer}\n\n"
        )
    raise ValueError(f"unsupported task: {task}")


def task_header(task: str) -> str:
    task = task.lower()
    headers = {
        "sst2": "Classify sentiment as positive or negative.\n\n",
        "subj": "Classify each sentence as subjective or objective.\n\n",
        "trec": "Classify each question as ABBR, DESC, ENTY, HUM, LOC, or NUM.\n\n",
        "rte": "Decide whether the premise entails the hypothesis. Answer entailment or not_entailment.\n\n",
    }
    return headers[task]


def stratified_sample(rows: Iterable[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    """Pick deterministic label-balanced few-shot rows without replacement."""

    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(str(row["label"]), []).append(row)
    if not groups:
        raise ValueError("cannot sample an empty dataset")
    rng = random.Random(seed)
    labels = sorted(groups)
    for group in groups.values():
        rng.shuffle(group)
    sampled = []
    offsets = {label: 0 for label in labels}
    for index in range(count):
        label = labels[index % len(labels)]
        if offsets[label] >= len(groups[label]):
            raise ValueError(f"not enough rows for label {label}")
        sampled.append(groups[label][offsets[label]])
        offsets[label] += 1
    return sampled


def length_matched_stratified_sample(
    task: str,
    rows: Iterable[dict[str, str]],
    count: int,
    target_prefix_tokens: int,
    token_count,
    seed: int,
) -> list[dict[str, str]]:
    """Pick balanced demonstrations whose formatted length approaches the paper table.

    The paper publishes target prefix lengths but not the exact sampled examples.
    This deterministic approximation preserves the published demonstration count
    and label balance while choosing examples near the needed per-example length.
    """

    grouped: dict[str, list[tuple[dict[str, str], int]]] = {}
    for row in rows:
        label = str(row["label"])
        grouped.setdefault(label, []).append((row, int(token_count(format_example(task, row, True)))) )
    if not grouped:
        raise ValueError("cannot sample an empty dataset")
    labels = sorted(grouped)
    quotas = {label: 0 for label in labels}
    for index in range(count):
        quotas[labels[index % len(labels)]] += 1
    header_tokens = int(token_count(task_header(task)))
    desired = max(1.0, (target_prefix_tokens - header_tokens) / count)
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for label in labels:
        candidates = list(grouped[label])
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: abs(item[1] - desired))
        quota = quotas[label]
        if len(candidates) < quota:
            raise ValueError(f"not enough rows for label {label}")
        selected.extend(row for row, _ in candidates[:quota])
    rng.shuffle(selected)
    return selected


def build_task_records(
    task: str,
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    *,
    seed: int = 42,
    eval_samples: int = 4,
    token_count=None,
    qwen_chat_format: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Build one shared prefix and deterministic query records for a task."""

    spec = PAPER_TASKS[task]
    if token_count is None:
        fewshots = stratified_sample(train_rows, spec.fewshot_examples, seed)
    else:
        fewshots = length_matched_stratified_sample(
            task,
            train_rows,
            spec.fewshot_examples,
            spec.paper_prefix_tokens,
            token_count,
            seed,
        )
    prompt_body = task_header(task) + "".join(format_example(task, row, True) for row in fewshots)
    prefix = QWEN_USER_PREFIX + prompt_body if qwen_chat_format else prompt_body
    rng = random.Random(seed)
    shuffled_eval = list(eval_rows)
    rng.shuffle(shuffled_eval)
    records = []
    for index, row in enumerate(shuffled_eval[:eval_samples]):
        records.append(
            {
                "uid": f"{task}-{index}",
                "task": task,
                "prefix_text": prefix,
                "query_text": (
                    format_example(task, row, False) + QWEN_ASSISTANT_PREFIX
                    if qwen_chat_format
                    else format_example(task, row, False)
                ),
                "answer": row["label"],
                "labels": sorted({str(item["label"]) for item in train_rows}),
            }
        )
    return prefix, records


def write_paper_task_bundles(
    output_dir: str | Path,
    data_root: str | Path,
    tasks: Iterable[str],
    *,
    seed: int = 42,
    eval_samples: int = 4,
    tokenizer_path: str | None = None,
    qwen_chat_format: bool = True,
) -> dict[str, dict[str, Any]]:
    """Write JSONL task bundles consumed by the paper plan and serving runners."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    token_count = None
    if tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, trust_remote_code=True
        )
        token_count = lambda text: len(tokenizer(text, add_special_tokens=False).input_ids)
    metadata: dict[str, dict[str, Any]] = {}
    for task in tasks:
        normalized = task.lower()
        train_rows, eval_rows = load_task_rows(normalized, data_root, seed)
        prefix, records = build_task_records(
            normalized,
            train_rows,
            eval_rows,
            seed=seed,
            eval_samples=eval_samples,
            token_count=token_count,
            qwen_chat_format=qwen_chat_format,
        )
        path = output / f"{normalized}.jsonl"
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        spec = PAPER_TASKS[normalized]
        metadata[normalized] = {
            "fewshot_examples": spec.fewshot_examples,
            "paper_prefix_tokens": spec.paper_prefix_tokens,
            "records": len(records),
            "prefix_characters": len(prefix),
            "actual_prefix_tokens": token_count(prefix) if token_count else None,
            "qwen_chat_format": qwen_chat_format,
            "bundle": str(path),
        }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper-style SST-2, SUBJ, TREC, and RTE task bundles.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", default="sst2,subj,trec,rte")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--qwen-chat-format", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    tasks = [task.strip().lower() for task in args.tasks.split(",") if task.strip()]
    unknown = sorted(set(tasks) - set(PAPER_TASKS))
    if unknown:
        raise ValueError(f"unsupported tasks: {', '.join(unknown)}")
    if not args.skip_download:
        ensure_paper_datasets(args.data_root)
    result = write_paper_task_bundles(
        args.output_dir,
        args.data_root,
        tasks,
        seed=args.seed,
        eval_samples=args.eval_samples,
        tokenizer_path=args.tokenizer,
        qwen_chat_format=args.qwen_chat_format,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
