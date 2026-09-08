"""Build the small three-stage JSON summary and Markdown review report."""

from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path("/home/panzihang/src/prism_max/src/prism_gao")
RESULTS = ROOT / "results"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(run: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run / "scored_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def average(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if field in row]
    return statistics.fmean(values) if values else 0.0


def differences(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, int]:
    left = {row["uid"]: row for row in baseline}
    right = {row["uid"]: row for row in candidate}
    if set(left) != set(right):
        raise RuntimeError("record UID sets differ")
    fields = (
        "layer_token_selection_sha256",
        "prediction",
        "generation_prediction",
        "correct",
    )
    return {
        field: sum(left[uid].get(field) != right[uid].get(field) for uid in left)
        for field in fields
    }


def run_row(run: Path, mode: str, budget: str) -> dict[str, Any]:
    payload = load(run / "summary.json")
    task = next(iter(payload["tasks"]))
    summary = payload["tasks"][task]
    rows = load_records(run)
    result = {
        "task": task,
        "budget": budget,
        "mode": mode,
        "samples": int(summary["samples"]),
        "accuracy": float(summary["accuracy"]),
        "mean_ttft_ms": float(summary["mean_ttft_ms"]),
        "p95_ttft_ms": float(summary["p95_ttft_ms"]),
        "selector_calls": float(summary.get("mean_selector_calls", 0.0)),
        "mean_period": float(summary.get("mean_promixed_period", 0.0)),
        "run": str(run),
    }
    for field in (
        "prism_gao_fp16_blocks",
        "prism_gao_int8_blocks",
        "prism_gao_promoted_int8_blocks",
        "prism_gao_int8_runs_before",
        "prism_gao_int8_runs_after",
        "prism_gao_payload_read_bytes",
        "prism_gao_payload_pread_calls",
        "prism_gao_payload_read_ms",
        "prism_gao_materialize_ms",
        "prism_gao_payload_byte_ratio",
    ):
        result[field] = average(rows, field)
    return result


def collect_stage_b() -> list[dict[str, Any]]:
    output = []
    pattern = str(RESULTS / "stage_b" / "*" / "*" / "summary.json")
    for path in sorted(glob.glob(pattern)):
        run = Path(path).parent
        budget = run.name.split("_", 1)[0].removeprefix("k")
        mode = run.name.rsplit("_", 2)[-2]
        output.append(run_row(run, mode, budget))
    return output


def collect_stage_c() -> list[dict[str, Any]]:
    output = []
    pattern = str(RESULTS / "stage_c" / "*" / "*" / "summary.json")
    for path in sorted(glob.glob(pattern)):
        run = Path(path).parent
        if not run.name.endswith("_64"):
            continue
        budget = run.name.split("_", 1)[0].removeprefix("k")
        mode = run.name.rsplit("_", 2)[-2]
        if mode in {"fp16", "naive", "coalesced"}:
            output.append(run_row(run, mode, budget))
    return output


def main() -> int:
    stage_a = {
        "trec": load(RESULTS / "trec64" / "ab_exact.json"),
        "subj": load(RESULTS / "subj64" / "ab_exact.json"),
    }
    direct_root = RESULTS / "trec64" / "trec"
    torch_run = direct_root / "k010_promixed_k4_torch_64"
    direct_run = direct_root / "k010_promixed_k4_fused_64"
    direct_audit = differences(
        load_records(torch_run),
        load_records(direct_run),
    )
    stage_b = collect_stage_b()
    period_order = {"adaptive": 0, "p1": 1, "p4": 2, "p8": 3}
    stage_b.sort(
        key=lambda row: (
            row["task"],
            row["budget"],
            period_order[row["mode"]],
        )
    )
    stage_c = collect_stage_c()
    precision_order = {"fp16": 0, "naive": 1, "coalesced": 2}
    stage_c.sort(
        key=lambda row: (
            row["task"], row["budget"], precision_order[row["mode"]]
        )
    )
    stage_c_audits = []
    groups = sorted({(row["task"], row["budget"]) for row in stage_c})
    for task, budget in groups:
        runs = {
            row["mode"]: Path(row["run"])
            for row in stage_c
            if row["task"] == task and row["budget"] == budget
        }
        if "fp16" not in runs:
            continue
        baseline = load_records(runs["fp16"])
        for mode in ("naive", "coalesced"):
            if mode in runs:
                stage_c_audits.append(
                    {
                        "task": task,
                        "budget": budget,
                        "candidate": mode,
                        **differences(baseline, load_records(runs[mode])),
                    }
                )

    fp16_reference_audits = []
    for task, budget in (("trec", "010"), ("subj", "025")):
        b_run = (
            RESULTS
            / "stage_b"
            / task
            / f"k{budget}_promixed_exact_adaptive_64"
        )
        c_run = (
            RESULTS
            / "stage_c"
            / task
            / f"k{budget}_promixed_exact_fp16_64"
        )
        fp16_reference_audits.append(
            {
                "task": task,
                **differences(load_records(b_run), load_records(c_run)),
            }
        )
    break_even_path = RESULTS / "stage_c" / "break_even.json"
    break_even = load(break_even_path)
    payload = {
        "scope": "64 samples per cell, one order, no multi-round ABBA",
        "stage_a_exact": stage_a,
        "stage_a_direct_audit": direct_audit,
        "stage_b": stage_b,
        "stage_c_break_even": break_even,
        "stage_c": stage_c,
        "stage_c_audits": stage_c_audits,
        "stage_c_fp16_reference_audits": fp16_reference_audits,
    }
    json_path = RESULTS / "three_stage_summary.json"
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# PRISM 三阶段最小实验报告（供 GPT 审核）",
        "",
        "## 实验边界",
        "",
        "本轮只做筛选实验：每个 cell 64 条，1 次 warmup（16 条），"
        "单一运行顺序；没有跑完整数据集，也没有做多轮 ABBA。"
        "所有新增实现和结果均位于 " + str(ROOT)
        + "，原 src/contiguous_fuxian 未修改。",
        "",
        "## 实现审计",
        "",
        "- A：packed INT4 codes/scales 直接进入 GPU；exact 只融合 unpack/"
        "dequant，仍沿用原 BF16 matmul。",
        "- B：固定周期钩子只覆盖 decision.period，不改当次 selected_blocks "
        "与 priority_blocks。",
        "- C：重要性只分配精度，标签写回原物理 block 顺序；短 INT8 run "
        "提升为 FP16，K/V 由融合 materialize 恢复原顺序。",
        "",
        "## 阶段 A：INT4 selector",
        "",
        "| task/budget | accuracy torch/exact | mean TTFT torch→exact | "
        "selector load torch→exact | exact 审计差异 |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, budget in (("trec", "k010"), ("subj", "k025")):
        result = stage_a[task]
        metric = result["metrics"]
        audit_total = sum(result["record_differences"].values())
        lines.append(
            f"| {task}/{budget} | "
            f"{metric['accuracy']['baseline']:.6f}/"
            f"{metric['accuracy']['candidate']:.6f} | "
            f"{metric['mean_ttft_ms']['baseline']:.2f}→"
            f"{metric['mean_ttft_ms']['candidate']:.2f} ms "
            f"({metric['mean_ttft_ms']['delta_percent']:.2f}%) | "
            f"{metric['mean_selector_load_ms']['baseline']:.2f}→"
            f"{metric['mean_selector_load_ms']['candidate']:.2f} ms | "
            f"{audit_total} |"
        )
    lines += [
        "",
        "结论：exact 路径可作为默认实现。它在两个 64 条实验中保持选择、"
        "预测和正确性完全一致，mean TTFT 分别下降 4.19% 和 7.15%。",
        "",
        "direct 路径不应作为正式实现：TREC 64 条中选择哈希变化 "
        f"{direct_audit['layer_token_selection_sha256']} 条，分类变化 "
        f"{direct_audit['prediction']} 条，correctness 变化 "
        f"{direct_audit['correct']} 条。",
        "",
        "## 阶段 B：固定周期与自适应周期",
        "",
        "| task/budget | mode | accuracy | mean TTFT | p95 | selector calls | "
        "mean period |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in stage_b:
        lines.append(
            f"| {row['task']}/k{row['budget']} | {row['mode']} | "
            f"{row['accuracy']:.6f} | {row['mean_ttft_ms']:.2f} ms | "
            f"{row['p95_ttft_ms']:.2f} ms | {row['selector_calls']:.2f} | "
            f"{row['mean_period']:.2f} |"
        )
    lines += [
        "",
        "B 阶段说明固定周期是明确的速度/准确率控制变量。P1 的 selector "
        "调用最多且明显最慢；P8 通常最快，但不保证准确率最高。因此当前不应"
        "仅凭 TTFT 把自适应策略替换成固定 P8。",
        "具体地，TREC/k010 的 P8 同时比 adaptive 快 61.37 ms 且多对 3/64 条，"
        "TREC/k050 的 P4 也同时更快且多对 2/64 条。",
        "但 SUBJ/k025 的 P8 虽快 7.84 ms，却少对 12/64 条；当前自适应阈值"
        "还不是稳定 Pareto 最优，需要按任务/不确定度重新校准。",
        "",
        "## 阶段 C：16/8/drop 与物理 run 合并",
        "",
        "break-even 微基准采用 5% 安全边际，最小可盈利 INT8 连续 run 为 "
        + str(break_even["min_profitable_int8_run_blocks"]) + " 个 block。",
        "",
        "新 FP16 reader 对照 Stage B adaptive：两个任务的选择、预测和正确性差异均为 0/64。",
        "",
        "| task/budget | mode | accuracy | mean TTFT | payload MiB | "
        "byte ratio | read ms | materialize ms | pread | "
        "FP16/INT8/promoted blocks |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stage_c:
        lines.append(
            f"| {row['task']}/k{row['budget']} | {row['mode']} | "
            f"{row['accuracy']:.6f} | {row['mean_ttft_ms']:.2f} ms | "
            f"{row['prism_gao_payload_read_bytes'] / 1048576:.2f} | "
            f"{row['prism_gao_payload_byte_ratio']:.3f} | "
            f"{row['prism_gao_payload_read_ms']:.2f} | "
            f"{row['prism_gao_materialize_ms']:.2f} | "
            f"{row['prism_gao_payload_pread_calls']:.1f} | "
            f"{row['prism_gao_fp16_blocks']:.1f}/"
            f"{row['prism_gao_int8_blocks']:.1f}/"
            f"{row['prism_gao_promoted_int8_blocks']:.1f} |"
        )
    lines += [
        "",
        "run 合并机制本身有效：相对 naive，coalesced 将 SUBJ/TREC 的 pread "
        "分别减少 36.3%/43.2%，mean TTFT 再下降 4.2%/1.3%。",
        "准确率方面，SUBJ 与 naive 相同且比 FP16 多对 1/64；TREC 比 naive "
        "多对 1/64，但仍比 FP16 少对 2/64。",
        "因此 C 证明了碎片治理有收益，但尚未证明混合精度在所有任务上无损。",
        "",
        "### C 阶段逐请求审计",
        "",
        "| task/budget | candidate vs FP16 | selection hash diff | "
        "prediction diff | correct diff |",
        "|---|---|---:|---:|---:|",
    ]
    for row in stage_c_audits:
        lines.append(
            f"| {row['task']}/k{row['budget']} | {row['candidate']} | "
            f"{row['layer_token_selection_sha256']} | "
            f"{row['prediction']} | {row['correct']} |"
        )
    lines += [
        "",
        "C 阶段是同步、无预取的隔离物理读取对照。低精度会改变后续 hidden "
        "state，进而反馈到在线 selector，所以报告同时列出选择哈希和预测差异，"
        "不能把字节下降直接等价为无损加速。",
        "",
        "## 请 GPT 重点审核",
        "",
        "1. 是否接受 Stage A exact 为当前默认 selector，并拒绝 direct 正式结果。",
        "2. Stage B 结果应只用于调阈值，还是足以改动自适应复用规则。",
        "3. Stage C 在当前结果下是否值得扩展到 16/8/4；若 16/8 没有稳定"
        "正收益，不建议立即扩大实现。",
        "4. 审核通过后再跑完整数据集和多轮 ABBA；本报告不把筛选结果包装成"
        "正式论文结论。",
        "",
        "机器可读汇总：" + str(json_path),
    ]
    report_path = ROOT / "GPT_REVIEW_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
