"""
大脑模拟系统主入口

提供命令行接口启动大脑模拟系统。
"""

import argparse
import json
import os
import sys
import time
import webbrowser
from typing import Dict, Any

from BrainSimulationSystem.brain_simulation import BrainSimulation
from BrainSimulationSystem.visualization.visualizer import BrainVisualizer
from BrainSimulationSystem.api.brain_api import BrainAPI
from BrainSimulationSystem.config.default_config import get_config, update_config

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="大脑模拟系统")
    
    # 基本参数
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--mode", type=str, default="interactive", 
                        choices=["interactive", "batch", "api", "visualization"],
                        help="运行模式")
    
    # API模式参数
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API服务器主机地址")
    parser.add_argument("--port", type=int, default=5000, help="API服务器端口号")
    
    # 批处理模式参数
    parser.add_argument("--input", type=str, help="输入文件路径")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--duration", type=float, default=1000.0, help="模拟持续时间（毫秒）")
    parser.add_argument("--dt", type=float, default=1.0, help="时间步长（毫秒）")
    
    # 可视化参数
    parser.add_argument("--visualize", action="store_true", help="启用可视化")
    parser.add_argument("--live", action="store_true", help="启用实时可视化")
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        print(f"错误：配置文件 {config_path} 不存在")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = json.load(f)
            return config
        except json.JSONDecodeError as e:
            print(f"错误：配置文件格式不正确 - {e}")
            sys.exit(1)

def load_inputs(input_path: str) -> Dict[str, Any]:
    """
    加载输入文件
    
    Args:
        input_path: 输入文件路径
        
    Returns:
        输入数据字典
    """
    if not os.path.exists(input_path):
        print(f"错误：输入文件 {input_path} 不存在")
        sys.exit(1)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            inputs = json.load(f)
            return inputs
        except json.JSONDecodeError as e:
            print(f"错误：输入文件格式不正确 - {e}")
            sys.exit(1)

def save_results(output_path: str, results: Dict[str, Any]) -> None:
    """
    保存结果
    
    Args:
        output_path: 输出文件路径
        results: 结果数据
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 转换numpy数组为列表
    def convert_numpy(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results = convert_numpy(results)
    
    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

def run_interactive_mode(brain: BrainSimulation, visualizer: BrainVisualizer) -> None:
    """
    运行交互模式
    
    Args:
        brain: 大脑模拟系统实例
        visualizer: 可视化器实例
    """
    print("=== 大脑模拟系统交互模式 ===")
    print("输入 'help' 查看可用命令")
    
    # 启动实时可视化
    visualizer.start_live_visualization()
    
    # 命令处理函数
    def handle_command(cmd: str) -> bool:
        """处理命令"""
        cmd = cmd.strip().lower()
        
        if cmd == "help":
            print("可用命令：")
            print("  help - 显示帮助信息")
            print("  info - 显示系统信息")
            print("  start - 开始连续模拟")
            print("  stop - 停止连续模拟")
            print("  reset - 重置模拟")
            print("  step - 执行一步模拟")
            print("  run <duration> <dt> - 运行模拟")
            print("  save <filepath> - 保存状态")
            print("  load <filepath> - 加载状态")
            print("  visualize - 显示可视化")
            print("  exit - 退出程序")
        
        elif cmd == "info":
            print(f"神经元数量：{len(brain.network.neurons)}")
            print(f"突触数量：{len(brain.network.synapses)}")
            print(f"当前时间：{brain.current_time} ms")
            print(f"运行状态：{'运行中' if brain.is_running else '空闲'}")
        
        elif cmd == "start":
            dt = float(input("时间步长（毫秒）："))
            brain.start_continuous_simulation(dt)
            print("连续模拟已启动")
        
        elif cmd == "stop":
            brain.stop_continuous_simulation()
            print("连续模拟已停止")
        
        elif cmd == "reset":
            brain.reset()
            print("模拟已重置")
        
        elif cmd == "step":
            inputs = {}
            brain.step(inputs, 1.0)
            print(f"执行一步模拟，当前时间：{brain.current_time} ms")
        
        elif cmd.startswith("run"):
            parts = cmd.split()
            if len(parts) >= 3:
                duration = float(parts[1])
                dt = float(parts[2])
                print(f"运行模拟，持续时间：{duration} ms，时间步长：{dt} ms")
                brain.run([], duration, dt)
                print("模拟完成")
            else:
                print("用法：run <duration> <dt>")
        
        elif cmd.startswith("save"):
            parts = cmd.split()
            if len(parts) >= 2:
                filepath = parts[1]
                brain.save_state(filepath)
                print(f"状态已保存到 {filepath}")
            else:
                print("用法：save <filepath>")
        
        elif cmd.startswith("load"):
            parts = cmd.split()
            if len(parts) >= 2:
                filepath = parts[1]
                brain.load_state(filepath)
                print(f"状态已从 {filepath} 加载")
            else:
                print("用法：load <filepath>")
        
        elif cmd == "visualize":
            visualizer.visualize_network_structure()
            visualizer.visualize_activity()
            visualizer.visualize_cognitive_state()
        
        elif cmd == "exit":
            return False
        
        else:
            print(f"未知命令：{cmd}")
        
        return True
    
    # 命令循环
    running = True
    while running:
        try:
            cmd = input("> ")
            running = handle_command(cmd)
        except KeyboardInterrupt:
            print("\n退出程序")
            running = False
        except Exception as e:
            print(f"错误：{e}")
    
    # 停止实时可视化
    visualizer.stop_live_visualization()

def run_batch_mode(brain: BrainSimulation, args) -> None:
    """
    运行批处理模式
    
    Args:
        brain: 大脑模拟系统实例
        args: 命令行参数
    """
    print("=== 大脑模拟系统批处理模式 ===")
    
    # 加载输入
    inputs = load_inputs(args.input)
    inputs_sequence = inputs.get("inputs_sequence", [])
    
    # 运行模拟
    print(f"运行模拟，持续时间：{args.duration} ms，时间步长：{args.dt} ms")
    start_time = time.time()
    results = brain.run(inputs_sequence, args.duration, args.dt)
    end_time = time.time()
    
    # 保存结果
    if args.output:
        save_results(args.output, results)
        print(f"结果已保存到 {args.output}")
    
    print(f"模拟完成，耗时：{end_time - start_time:.2f} 秒")

def run_api_mode(brain: BrainSimulation, args) -> None:
    """
    运行API模式
    
    Args:
        brain: 大脑模拟系统实例
        args: 命令行参数
    """
    print("=== 大脑模拟系统API模式 ===")
    
    # 创建API
    api = BrainAPI(brain)
    
    # 启动API服务器
    print(f"启动API服务器，地址：{args.host}，端口：{args.port}")
    api.start(host=args.host, port=args.port)
    
    # 打开浏览器
    url = f"http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/api/info"
    print(f"API服务器已启动，访问：{url}")
    
    try:
        # 等待用户中断
        print("按 Ctrl+C 停止服务器")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n停止API服务器")
        api.stop()

def run_visualization_mode(brain: BrainSimulation, visualizer: BrainVisualizer) -> None:
    """
    运行可视化模式
    
    Args:
        brain: 大脑模拟系统实例
        visualizer: 可视化器实例
    """
    print("=== 大脑模拟系统可视化模式 ===")
    
    # 显示可视化
    visualizer.visualize_network_structure()
    visualizer.visualize_activity()
    visualizer.visualize_cognitive_state()
    
    # 启动实时可视化
    visualizer.start_live_visualization()
    
    try:
        # 等待用户中断
        print("按 Ctrl+C 停止可视化")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n停止可视化")
        visualizer.stop_live_visualization()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = get_config()
    if args.config:
        user_config = load_config(args.config)
        config = update_config(user_config)
    
    # 创建大脑模拟系统
    brain = BrainSimulation(config)
    
    # 创建可视化器
    visualizer = BrainVisualizer(brain)
    
    # 根据模式运行
    if args.mode == "interactive":
        run_interactive_mode(brain, visualizer)
    elif args.mode == "batch":
        run_batch_mode(brain, args)
    elif args.mode == "api":
        run_api_mode(brain, args)
    elif args.mode == "visualization":
        run_visualization_mode(brain, visualizer)

if __name__ == "__main__":
    main()