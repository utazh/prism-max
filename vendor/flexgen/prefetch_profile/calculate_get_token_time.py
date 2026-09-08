import re
from collections import defaultdict
import statistics
import numpy as np

def parse_device_token_stats(file_path):
    """解析日志文件，按设备统计单个token获取时间"""
    
    device_stats = {
        'disk': {' key ': [], 'value': []},
        'cpu': {' key ': [], 'value': []},
        'cuda:0': {' key ': [], 'value': []}
    }
    
    # 正则表达式匹配日志行
    pattern = r'get ( key |value) time: ([\d.]+) s, token_id len: (\d+), device: (\w+:?\d*)'
    
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(pattern, line.strip())
            if match:
                data_type, time_str, token_len_str, device = match.groups()
                
                time_val = float(time_str)
                token_len = int(token_len_str)
                
                # 计算单个token的平均时间
                if token_len > 0:
                    per_token_time = time_val / token_len
                    
                    # 按设备分类存储
                    if device in device_stats:
                        device_stats[device][data_type].append(per_token_time)
    
    return device_stats

def analyze_device_stats(device_stats):
    """分析统计结果"""
    
    print("=== 单个Token获取时间统计 (按设备分类) ===\n")
    
    for device, data in device_stats.items():
        if not data[' key '] and not data['value']:
            continue
            
        print(f"📍 设备: {device.upper()}")
        print("-" * 50)
        
        for data_type in [' key ', 'value']:
            times = data[data_type]
            if not times:
                continue
                
            mean_time = statistics.mean(times)
            median_time = statistics.median(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            min_time = min(times)
            max_time = max(times)

            # 波动性分析指标
            cv = (std_time / mean_time) * 100 if mean_time > 0 else 0  # 变异系数(%)
            range_val = max_time - min_time  # 极差
            iqr = calculate_iqr(times)  # 四分位距
            
            # 分位数
            p90 = np.percentile(times, 90)
            p95 = np.percentile(times, 95)
            p99 = np.percentile(times, 99)
            
            # 异常值检测
            outliers_count = count_outliers(times)
            outliers_ratio = (outliers_count / len(times)) * 100 if times else 0
            
            print(f"  {data_type.upper()} 缓存:")
            print(f"    📊 样本数量: {len(times)}")
            print(f"    ⏱️  平均时间: {mean_time*1000:.4f} ms/token")
            print(f"    📈 中位数时间: {median_time*1000:.4f} ms/token")
            print(f"    📉 标准差: {std_time*1000:.4f} ms")
            print(f"    🔺 最大值: {max_time*1000:.4f} ms/token")
            print(f"    🔻 最小值: {min_time*1000:.4f} ms/token")
            # 波动性指标
            print(f"    📊 变异系数: {cv:.2f}% {'🟢(稳定)' if cv < 20 else '🟡(中等)' if cv < 50 else '🔴(波动大)'}")
            print(f"    📏 极差: {range_val*1000:.4f} ms/token")
            print(f"    📐 四分位距: {iqr*1000:.4f} ms/token")
            print(f"    🎯 90%分位数: {p90*1000:.4f} ms/token")
            print(f"    🎯 95%分位数: {p95*1000:.4f} ms/token")
            print(f"    🎯 99%分位数: {p99*1000:.4f} ms/token")
            print(f"    ⚠️  异常值: {outliers_count} 个 ({outliers_ratio:.2f}%)")
            # 波动性综合评估
            stability_score = evaluate_stability(cv, outliers_ratio)
            print(f"    🏆 稳定性评级: {stability_score}")
            print()
        
        print()

def calculate_iqr(data):
    """计算四分位距"""
    q75 = np.percentile(data, 75)
    q25 = np.percentile(data, 25)
    return q75 - q25

def count_outliers(data):
    """使用IQR方法检测异常值"""
    q75 = np.percentile(data, 75)
    q25 = np.percentile(data, 25)
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return len(outliers)

def evaluate_stability(cv, outliers_ratio):
    """综合评估数据稳定性"""
    if cv < 10 and outliers_ratio < 1:
        return "🟢 非常稳定"
    elif cv < 20 and outliers_ratio < 3:
        return "🟢 稳定"
    elif cv < 35 and outliers_ratio < 5:
        return "🟡 中等波动"
    elif cv < 50 and outliers_ratio < 10:
        return "🟠 波动较大"
    else:
        return "🔴 波动很大"

# 使用示例
if __name__ == "__main__":
    # 分析你的日志文件
    file_path = "get_token_profile.txt"
    
    # 生成统计报告
    device_stats = parse_device_token_stats(file_path)
    analyze_device_stats(device_stats)