"""Summarize the bounded concurrency study without treating KV replay as TTFT."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from .benchmark_concurrency import load_traces, percentile, read_jsonl

MODES = ("fp16", "naive", "coalesced")
LABELS = {"fp16": "FP16", "naive": "16/8 未合并", "coalesced": "16/8 短 run 合并"}


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["concurrency"], row["mode"])].append(row)
    summary = {}
    for (c, mode), cells in groups.items():
        assert len(cells) == 2 and {r["round"] for r in cells} == {0, 1}
        requests = [r for cell in cells for r in cell["requests"]]
        latencies = [r["latency_ms"] for r in requests]
        total_s = sum(r["wall_s"] for r in cells)
        summary[f"{c}/{mode}"] = {
            "concurrency": c, "mode": mode, "samples_including_repeats": len(requests),
            "unique_requests": len({r["uid"] for r in requests}),
            "mean_ms": statistics.mean(latencies),
            "p50_ms": percentile(latencies, .5),
            "p95_ms": percentile(latencies, .95),
            "requests_per_s": len(requests) / total_s,
            "mean_logical_mib": statistics.mean(r["read_bytes"] for r in requests) / 2**20,
            "mean_pread_calls": statistics.mean(r["pread_calls"] for r in requests),
            "logical_bytes": sum(r["logical_read_bytes"] for r in cells),
            "storage_bytes": sum(r["process_storage_read_bytes"] for r in cells),
            "mean_host_wall_ms": statistics.mean(r["host_wall_ms"] for r in requests),
            "mean_materialize_ms": statistics.mean(r["materialize_event_ms"] for r in requests),
            "round_mean_ms": [r["mean_ms"] for r in sorted(cells, key=lambda r: r["round"])],
            "round_throughput": [r["requests_per_s"] for r in sorted(cells, key=lambda r: r["round"])],
        }
    for row in summary.values():
        base = summary[f'{row["concurrency"]}/fp16']
        row["latency_change_pct"] = (row["mean_ms"] / base["mean_ms"] - 1) * 100
        row["throughput_change_pct"] = (row["requests_per_s"] / base["requests_per_s"] - 1) * 100
        row["byte_ratio"] = row["logical_bytes"] / base["logical_bytes"]
        row["pread_ratio"] = row["mean_pread_calls"] / base["mean_pread_calls"]
    return summary


def table(summary):
    lines = [
        "| 并发 | 方法 | 平均 ms | P95 ms | 请求/s | 延迟变化 | 吞吐变化 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for c in sorted({r["concurrency"] for r in summary.values()}):
        for mode in MODES:
            r = summary[f"{c}/{mode}"]
            lines.append(
                f'| {c} | {LABELS[mode]} | {r["mean_ms"]:.2f} | {r["p95_ms"]:.2f} | '
                f'{r["requests_per_s"]:.3f} | {r["latency_change_pct"]:+.2f}% | '
                f'{r["throughput_change_pct"]:+.2f}% |')
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    package = Path(__file__).resolve().parent
    if package not in root.parents:
        raise ValueError("report must stay in prism_gao")
    sources = {
        "thread_buffered": ("replay_buffered_warm.jsonl", 18),
        "thread_direct": ("replay_direct.jsonl", 18),
        "process_buffered": ("process_buffered.jsonl", 6),
        "process_direct": ("process_direct.jsonl", 6),
    }
    data, metadata = {}, {}
    for name, (file, count) in sources.items():
        rows = read_jsonl(root / file)
        assert len(rows) == count + 1, f"incomplete results: {file}"
        metadata[name] = rows[0]["metadata"]
        data[name] = aggregate(rows[1:])
        assert metadata[name]["verification"]["all_byte_exact"]
        if "buffered" in name:
            assert all(r["process_storage_read_bytes"] == 0 for r in rows[1:]), "cache state not uniform"
        else:
            assert all(.99 <= r["storage_to_logical_ratio"] <= 1.02 for r in rows[1:]), "direct I/O evidence missing"
    assert len({m["trace_sha256"] for m in metadata.values()}) == 1
    verified = load_traces(root, 8)
    audit = {"verified_trace_selection_hashes": len(verified),
             "direct_read_byte_exact_checks_per_benchmark_invocation": 24,
             "identical_trace_digest_all_experiments": True,
             "original_source_verified": all(
                 line.endswith(": OK") for line in
                 (root / "original_source_verified.txt").read_text().splitlines())}
    old_root = package / "results/stage_c_v4_v4_20260902_0210/forward"
    mismatches = defaultdict(int)
    compared = 0
    for task in ("sst2", "subj", "trec", "rte"):
        old = {r["uid"]: r for r in read_jsonl(
            old_root / task / "k050_fp16_pipeline_forward/scored_records.jsonl")}
        for row in read_jsonl(root / "capture" / task / "capture_fp16/scored_records.jsonl"):
            compared += 1
            for field in ("prediction", "correct", "layer_token_selection_sha256"):
                mismatches[field] += int(row[field] != old[row["uid"]][field])
    audit["capture_vs_v4_records"] = compared
    audit["capture_vs_v4_mismatches"] = dict(mismatches)
    tests = (root / "tests.txt").read_text()
    report = {"scope": "KV readiness concurrency diagnostic; not end-to-end TTFT",
              "metadata": metadata, "results": data, "audit": audit, "tests": tests}
    (root / "concurrency_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    t = data["thread_direct"]["4/coalesced"]
    p = data["process_direct"]["4/coalesced"]
    single = data["thread_direct"]["1/coalesced"]
    lines = [
        "# PRISM-Gao 并发分级精度最小实验审查报告",
        "",
        "## 结论",
        "",
        f'在真实 SSD 直读的 KV 链路上出现了小幅收益：并发 4 的单进程线程版，'
        f'短 run 合并相对 FP16 平均延迟变化 {t["latency_change_pct"]:+.2f}%，'
        f'吞吐变化 {t["throughput_change_pct"]:+.2f}%。独立进程对照分别为 '
        f'{p["latency_change_pct"]:+.2f}% / {p["throughput_change_pct"]:+.2f}%。',
        f'单请求仍为负收益（合并方案延迟 {single["latency_change_pct"]:+.2f}%）。'
        "因此不能概括为“并发一定使混合精度更快”，更不能据此声称端到端 TTFT 已经改善。",
        "",
        "四独立进程下，未合并的 16/8 也获益且略快于固定阈值合并方案。"
        "因此目前支持的是分级精度在 I/O 受压时有潜力，尚不能证明 run 合并总是最优；阈值需要按负载校准。",
        "",
        "## 实验边界与公平性",
        "",
        "- 只修改/新增服务器 src/prism_gao 下的诊断代码；原 contiguous_fuxian 顶层源码哈希全部复核，git diff 无已跟踪改动；未生成本地备份。",
        "- GPU 2，RTX 3090；payload 位于 /dev/sda4 ext4 SATA SSD。GPU 0/1 的既有任务未改动，GPU 3 未使用。",
        "- SST2/SUBJ/TREC/RTE 的 prefix 长度为 3811/4412/4940/6011 token，28 层、4 KV heads、head_dim=128、block=16、group=32。",
        "- 采集每任务 8 条真实查询（offset=16，另有 2 条 warmup）；回放每任务前 4 条，共 16 条唯一请求。两种顺序各一次，每个 cell 32 次观测，不是 32 条独立请求。",
        "- A 固定 exact + resident selector，B 固定 V4 profiled reuse，预算 k050。所有精度变体回放完全相同的逐层 selected/priority blocks；32/32 采集记录与实际选择 SHA256 一致。",
        "- FP16：所有 selected blocks 为 FP16；naive：重要性最高 25% FP16，其余 INT8；coalesced：短于 2 blocks 的 INT8 run 升为 FP16。drop、预算和物理 token 顺序保持不变。",
        "- 成本 gate 在回放中明确不使用，这是 ungated coalescer 的性能消融。没有调低 gate 门槛宣称原 V4 策略已获益，原默认策略未替换。",
        "- 主实验是一个进程、1/2/4 个 KV 请求线程；补充实验是 4 个独立进程，共享同一 SSD/GPU，但各有 CUDA context。它们均无模型 attention 计算，不是四个完整模型副本。请求固定分配到 lane，短任务结束后不迁移其他 lane 的请求，因此收尾阶段并发会下降。",
        "- 单请求从首层 host read 开始，到最后一层 H2D + materialize 完成；最多保留两层在途 GPU payload，源/目的张量均保留到 event 完成。同步只发生在此微基准回收边界，未改正式异步流水线。",
        "- 吞吐按总完成请求数 / 同批 wall time 计算；延迟为闭环服务时间，P95 为 32 次观测合并后的线性插值分位数，不包含开放到达队列等待，也不是 SLO capacity。",
        "- buffered 统一预热三种策略的完整工作集，记录实际 storage bytes=0；direct 使用只读 O_DIRECT，512B 对齐 bounce buffer，不清系统 page cache、不写 KV 文件，不人为限速。",
        "",
        "## 1. 单进程多线程：缓存已热",
        "",
    ] + table(data["thread_buffered"]) + [
        "",
        "## 2. 单进程多线程：SSD 直读",
        "",
    ] + table(data["thread_direct"]) + [
        "",
        "## 3. 四独立进程对照",
        "",
        "### 缓存已热",
        "",
    ] + table(data["process_buffered"]) + [
        "",
        "### SSD 直读",
        "",
    ] + table(data["process_direct"]) + [
        "",
        "## 4. 为什么收益没有达到字节压缩比例",
        "",
        "| 方法 | 每请求逻辑 MiB | 每请求 pread 次数 | 字节比例 | pread 比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        r = data["thread_direct"][f"4/{mode}"]
        lines.append(f'| {LABELS[mode]} | {r["mean_logical_mib"]:.2f} | '
                     f'{r["mean_pread_calls"]:.1f} | {r["byte_ratio"]:.4f} | {r["pread_ratio"]:.3f} |')
    lines += [
        "",
        "INT8 确实降低了实际磁盘字节量，但 codes/scales 分文件、精度切换和短 run 增加了小读取次数。"
        "因此本机既有字节/排队收益，也有 IOPS、系统调用和 host packing 代价。SSD busy 接近 100% 不能单独证明纯带宽已经饱和。",
        f'并发 4 的合并方案中，线程版平均 host read/packing 为 {t["mean_host_wall_ms"]:.2f} ms，'
        f'H2D+materialize GPU event 合计约 {t["mean_materialize_ms"]:.2f} ms。'
        "后者包含传输和物化，不是独立反量化计时；可判断这一回放的主要阻力在读取端，而不能把损失全部归因于反量化。",
        "",
        "线程与独立进程的缓存结果差异说明 Python 线程争用会影响并发结论；独立进程对照同时引入了多个 CUDA context，不能把两者的全部差值精确归因于 GIL。",
        "",
        "## 5. 审计、排除项和不能下的结论",
        "",
        "- 初始 replay_buffered.jsonl 的首轮 naive/C1 有约 362.8 MiB 真实磁盘读取，冷热状态不一致。完整原始日志保留，仅作为预实验；正式缓存表使用统一预热重跑的 replay_buffered_warm.jsonl。",
        "- 单请求 smoke_direct.jsonl 仅用于检查 O_DIRECT 能运行及数据一致性，不混入主表。",
        "- SSD 直读各 cell 的进程实际磁盘读取量与逻辑量误差小于 2%；对齐尾部放大计入 physical bytes。全盘 diskstats 包含其他用户活动，只作辅助证据。",
        "- 这里刻意固定选择并省略 attention，使被计算掩盖的 I/O 尽量暴露。这证明的是“在 KV 供给成为瓶颈时的链路潜力”，不是对上一版端到端负收益的直接因果证明。",
        "- 四个 prefix 仍是已有 4K–6K 数据，O_DIRECT 是明确的冷/非驻留存储诊断。未伪造更长 prefix，未把复制字节冒充真实长上下文查询。",
        "- 没有重测混合精度的生成准确率，没有证明 selection 在真实低精度反馈下仍相同，没有完成在线 serving、TTFT/P95/SLO 或完整数据集验证。",
        "- 两种运行顺序是小样本方向确认，不是多轮 ABBA 或统计显著性结论；不应根据这 16 条请求的测试准确率调整策略。",
        "",
        "## 6. 最小下一步",
        "",
        "保留现有默认实现。下一轮优先在共享 SSD 的真实多请求端到端 runner 中对照 FP16、未合并和合并三种方案，"
        "增加 resolve 前 host-ready/GPU-ready 和真正 consumer-stream wait 的测量；不要用请求末尾 event.query() 冒充预取命中。"
        "只有端到端延迟/吞吐和独立质量样本一起过关后，才考虑负载感知启用 mixed；本轮不再改 A/B。",
        "",
        "## 测试与文件",
        "",
        f'- 采集对 V4 基线核对：{compared} 条，字段差异 {dict(mismatches)}。',
        f'- 原源码哈希通过：{audit["original_source_verified"]}；direct 原始字节一致性检查：24/24。',
        "",
        "```text",
        tests.strip(),
        "```",
        "",
        "- 原始结果：replay_direct.jsonl、replay_buffered_warm.jsonl、process_direct.jsonl、process_buffered.jsonl。",
        "- 汇总：concurrency_summary.json；记录：capture/、各 driver.log、replay_environment.txt。",
        "- 代码：benchmark_concurrency.py、concurrency_io.py、concurrency_hooks.py、tests/test_concurrency.py；run_fused_selector.py 仅增加可选 hook 入口。",
        "- 复跑使用新的 run-root/output 路径，避免覆盖原数据。capture/replay/process/validation 的 shell 脚本及其参数保存在 src/prism_gao。",
        "",
        "所有数据和报告留在服务器，本轮没有打包或拷贝到本地桌面。",
    ]
    text = "\n".join(lines) + "\n"
    (root / "GPT_REVIEW_REPORT_CONCURRENCY.md").write_text(text, encoding="utf-8")
    print(json.dumps({"report": str(root / "GPT_REVIEW_REPORT_CONCURRENCY.md"),
                      "thread_direct_c4": t, "process_direct_c4": p, "audit": audit},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
