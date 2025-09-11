"""AutoGPT 排序算法性能基准测试脚本。

本脚本用于测试和比较不同排序算法的性能表现，生成详细的基准测试报告。
主要测试冒泡排序和快速排序在不同数据规模下的执行时间。

功能特性:
    - 多种数据规模的性能测试
    - 精确的时间测量（毫秒级）
    - CSV 格式的结果输出
    - 自动创建结果目录

测试算法:
    - BubbleSort: 冒泡排序算法
    - QuickSort: 快速排序算法

输出格式:
    CSV 文件包含数据规模和各算法的执行时间（毫秒）

使用场景:
    - 算法性能分析
    - 性能回归测试
    - 算法优化验证
    - 教学演示
"""

import os
import random
import time
from pathlib import Path

# 添加项目根目录到 Python 路径，以便导入算法模块
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.sorting.basic.bubble_sort import BubbleSort
from algorithms.sorting.basic.quick_sort import QuickSort

# 基准测试结果文件路径
# 结果将保存在项目根目录的 benchmarks 文件夹中
RESULT_FILE = Path(__file__).resolve().parent.parent / "benchmarks" / "sorting_benchmark.txt"


def benchmark() -> None:
    """执行排序算法基准测试。
    
    测试不同规模数据下各种排序算法的性能表现，
    并将结果保存为 CSV 格式的文件。
    
    测试配置:
        - 数据规模: 50, 100, 200 个元素
        - 数据类型: 随机整数
        - 测量精度: 毫秒级
        
    输出格式:
        CSV 文件，包含以下列：
        - size: 数据规模
        - bubble_sort_ms: 冒泡排序执行时间（毫秒）
        - quick_sort_ms: 快速排序执行时间（毫秒）
        
    性能测量:
        使用 time.perf_counter() 进行高精度时间测量，
        确保测试结果的准确性和可重现性。
    """
    # 定义测试的数据规模
    sizes = [50, 100, 200]
    
    # 初始化结果列表，包含 CSV 头部
    lines = ["size,bubble_sort_ms,quick_sort_ms"]
    
    # 对每个数据规模进行测试
    for n in sizes:
        # 生成随机测试数据
        # 使用 sample 确保数据唯一性，范围是数据规模的两倍
        data = random.sample(range(n * 2), n)
        
        # 测试冒泡排序性能
        start = time.perf_counter()
        BubbleSort().execute(data)
        bubble_ms = (time.perf_counter() - start) * 1000  # 转换为毫秒
        
        # 测试快速排序性能
        start = time.perf_counter()
        QuickSort().execute(data)
        quick_ms = (time.perf_counter() - start) * 1000  # 转换为毫秒
        
        # 记录测试结果，保留3位小数
        lines.append(f"{n},{bubble_ms:.3f},{quick_ms:.3f}")
    
    # 确保结果目录存在
    RESULT_FILE.parent.mkdir(exist_ok=True)
    
    # 将结果写入文件
    RESULT_FILE.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    # 当脚本直接运行时执行基准测试
    benchmark()