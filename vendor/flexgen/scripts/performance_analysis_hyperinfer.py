import os
import re
import csv
from collections import defaultdict

# 1. 从文件名里解析参数 -----------------------------------------

def parse_config_from_filename(filename: str) -> dict:
    """从文件名中解析出各种参数，返回一个 dict"""
    cfg = {}

    patterns = {
        "full_load":        r"full_load(True|False)",
        "sele_load":        r"sele_load(True|False)",
        "sele_load_by_p":   r"sele_load_by_p(True|False)",
        "sele_percent":     r"sele_percent\[(\d+)\]",
        "sim_thred":        r"sim_thred([0-9.]+)",
        "cache_type":       r"cache_type([A-Za-z0-9_]+)",
        "disk_type":        r"disk_type([A-Za-z0-9_]+)",
        "prefix_aware_inf": r"prefix_aware_inf(True|False)",
        "ro":               r"ro(True|False)",
        "cksize":           r"cksize(\d+)",
        "no_prefetch":      r"no_prefetch(True|False)",
        "recompute":        r"recompute(True|False)",
    }

    # 先按原有规则解析各种布尔 / 数值参数
    for key, pat in patterns.items():
        m = re.search(pat, filename)
        if not m:
            continue
        val = m.group(1)
        if val in ("True", "False"):
            cfg[key] = (val == "True")
        elif key in ("sele_percent", "cksize"):
            cfg[key] = int(val)
        elif key == "sim_thred":
            cfg[key] = float(val)
        else:
            cfg[key] = val

    # ===== fo-模型-数据集-xxx-full_load... 的规则提取 model / dataset =====
    base = os.path.basename(filename)

    # 找到 "-full" 的位置（包括 full_load / full_xxx 等）
    idx = base.find("-full")
    if idx != -1:
        prefix = base[:idx]          # 例如: 'fo-6.7b-copa-expand'
    else:
        # 万一没有 "-full"，就用去掉后缀的整个名字
        prefix = os.path.splitext(base)[0]

    parts = prefix.split("-")

    # 期望格式: fo-模型-数据集-xxx...
    # parts: ['fo', '<model>', '<dataset>', 'xxx', ...]
    if len(parts) >= 3:
        model = parts[1]                  # 第二段 = 模型
        dataset = "-".join(parts[2:])     # 第三段开始到结尾 = 数据集-xxx
        cfg["model"] = model
        cfg["dataset"] = dataset
    else:
        cfg.setdefault("model", "UNKNOWN_MODEL")
        cfg.setdefault("dataset", "UNKNOWN_DATASET")

    return cfg


# 2. 根据参数组合映射到“技术”名称 -------------------------------

def infer_method_from_config(cfg: dict) -> str:
    """依据参数组合归类到某个技术（method 名字）"""

    cache_type   = cfg.get("cache_type", None)
    sele_load    = cfg.get("sele_load", None)         # True / False
    disk_type    = cfg.get("disk_type", None)         # KV_Division / Chunk / ...
    ro           = cfg.get("ro", None)                # True / False
    recompute    = cfg.get("recompute", None)
    sele_percent = cfg.get("sele_percent", None)
    full_load    = cfg.get("full_load", None)
    no_prefetch  = cfg.get("no_prefetch", None)
    reorder      = cfg.get("ro", None)
    

    if recompute is True:
        return "recompute"
    
    if full_load is True:
        if sele_percent == 100:
            return "as_like"
        else:
            return "generate_mapping_list"

    if full_load is False and sele_load is False:
        if cache_type == "LRU":
            return "as+h2o+lru"
        elif cache_type == "LFU":
            return "as+h2o+lfu"

    if sele_load is True:
        if no_prefetch is True:
            if reorder is True:
                if cache_type == "CKLFU":
                    return "impress"
                elif cache_type == "LRU":
                    return "impress - cklfu"
            else:
                return "impress - cklfu - reorder"
        else:
            if reorder is True:
                if cache_type == "CKLFU":
                    return "hyperinfer"
                elif cache_type == "LRU":
                    return "hyperinfer - cklfu"
            else:
                return "hyperinfer - cklfu - reorder"

    # 兜底
    return "UNKNOWN"


# 3. 递归收集日志文件 --------------------------------------------

def extract_log_files(root_paths, suffix=".log"):
    log_files = []
    for path in root_paths:
        if os.path.isfile(path) and path.endswith(suffix):
            log_files.append(path)
        elif os.path.isdir(path):
            for name in os.listdir(path):
                full = os.path.join(path, name)
                log_files.extend(extract_log_files([full], suffix=suffix))
    return log_files


# 4. 从日志里抽各种时间指标 --------------------------------------

def parse_time_metrics_from_log(path: str) -> dict:
    """
    从日志中解析出各种时间字段，比如:
    suffix time:526.26,all compute:0,load:0,sele:593.74,load_head:0
    get key:976.34,get value:733.10
    p99:0.3096
    返回一个 dict, key 已经转成 snake_case，如 suffix_time / all_compute / get_key / get_value / p99
    """
    metrics: dict[str, float] = {}

    # 兼容旧格式: "total time: xxx"
    total_re = re.compile(r"total time:\s*([0-9.]+)")

    # 通用 KV 解析: 形如 "xxx:123.45"
    kv_re = re.compile(r"([A-Za-z_ ]+):([0-9.]+)")

    with open(path, "r") as f:
        for line in f:
            # 旧 total time
            m_total = total_re.search(line)
            if m_total:
                try:
                    metrics["total_time"] = float(m_total.group(1))
                except ValueError:
                    pass

            # 新的 key:value,key:value,...
            for k, v in kv_re.findall(line):
                key = k.strip().replace(" ", "_")  # "suffix time" -> "suffix_time"
                try:
                    metrics[key] = float(v)
                except ValueError:
                    continue

    return metrics


def compute_time_for_method(method: str, metrics: dict) -> float | None:
    """
    根据 method + metrics，算一个“总时间”出来。
    不同方法可以用不同公式，在这里写分支逻辑。
    """

    if not metrics:
        return None

    if method.startswith("recompute"):
        # 这里假设你想用 all_compute
        return metrics.get("all_compute")

    if method.startswith("impress"):
        return (
            metrics.get("suffix_time")
            + metrics.get("sele")
            + metrics.get("load_head", 0.0)
            + metrics.get("load")
        )
    
    if method.startswith("hyperinfer"):
        # hyperinfer: 同上
        return (
            metrics.get("suffix_time")
            + metrics.get("sele")
            + metrics.get("load_head", 0.0)
            + metrics.get("load")
        )

    if method == "as_like":
        # as_like: suffix_time + load
        return metrics.get("suffix_time") + metrics.get("load")

    if method in ("as+h2o+lru", "as+h2o+lfu"):
        # as+h2o: suffix_time + sele + get_key + get_value
        return (
            metrics.get("suffix_time", 0.0)
            + metrics.get("sele", 0.0)
            + metrics.get("get_key", 0.0)
            + metrics.get("get_value", 0.0)
        )

    # 兜底：算不出来就 None
    return None

def load_time_for_method(method: str, metrics: dict) -> float | None:
    """
    根据 method + metrics，算一个“总时间”出来。
    不同方法可以用不同公式，在这里写分支逻辑。
    """

    if not metrics:
        return None

    if method.startswith("recompute"):
        # 这里假设你想用 all_compute
        return 0.0

    if method.startswith("impress"):
        return (
            
            metrics.get("load")
        )
    
    if method.startswith("hyperinfer"):
        # hyperinfer: 同上
        return (
            0.0
            + metrics.get("load")
        )

    if method == "as_like":
        # as_like: suffix_time + load
        return metrics.get("load")

    if method in ("as+h2o+lru", "as+h2o+lfu"):
        # as+h2o: suffix_time + sele + get_key + get_value
        return (
            0.0
            + metrics.get("get_key", 0.0)
            + metrics.get("get_value", 0.0)
        )

    # 兜底：算不出来就 None
    return None


def compute_oracle_time_for_impress(metrics: dict) -> float | None:
    """
    计算 impress 的 oracle time: suffix_time + load_head + sele + ideal_load
    ideal_load = get_key + get_value
    """
    if not metrics:
        return None


    oracle_time = metrics.get('orcale_time', 0.0)
    return oracle_time

def compute_oracle_load_time_for_impress(metrics: dict) -> float | None:
    """
    计算 impress 的 oracle time: suffix_time + load_head + sele + ideal_load
    ideal_load = get_key + get_value
    """
    if not metrics:
        return None


    oracle_time = metrics.get('orcale_load_time', 0.0)

    return oracle_time
# 5. 主流程：逐条写入 CSV ----------------------------------------
if __name__ == "__main__":
    log_dir = "../logs"
    logs = extract_log_files([log_dir], suffix=".log")
    print(f"找到日志文件数: {len(logs)}")

    rows = []  # 先把所有条目存起来

    for path in logs:
        fname = os.path.basename(path)
        cfg = parse_config_from_filename(fname)
        method = infer_method_from_config(cfg)
        model = cfg.get("model", "UNKNOWN_MODEL")
        dataset = cfg.get("dataset", "UNKNOWN_DATASET")

        metrics = parse_time_metrics_from_log(path)
        total_time = compute_time_for_method(method, metrics)
        load_time = load_time_for_method(method, metrics)
        if method == 'impress':
            oracle_time = compute_oracle_time_for_impress(metrics)
            orcale_load_time = compute_oracle_load_time_for_impress(metrics)
            rows.append({
                "filename": fname,
                "method": 'oracle',
                "model": model,
                "dataset": dataset,
                "time": oracle_time,
                "load_time": orcale_load_time,
            })
        if total_time is None:
            print(f"[WARN] {path} 没算出 total_time, metrics={metrics}，跳过")
            continue

        rows.append({
            "filename": fname,
            "method": method,
            "model": model,
            "dataset": dataset,
            "time": total_time,
            "load_time": load_time if load_time is not None else 0.0,
        })

    # 按 数据集 -> 模型 -> 方法 排序
    rows.sort(key=lambda r: (r["dataset"], r["model"], r["method"]))

    # 要输出的 csv 文件名
    csv_path = "log_times.csv"

    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ "dataset", "model", "method", "time" , 'load_time'])
        for r in rows:
            writer.writerow([
                r["dataset"],
                r["model"],
                r["method"],
                r["time"],
                r['load_time']
            ])

    print(f"已写入排序后的 CSV: {csv_path}")

