"""
认知系统集成测试

测试注意力系统、工作记忆和认知控制器的集成功能。
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import time

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from BrainSimulationSystem.models.attention import AttentionSystem
from BrainSimulationSystem.models.working_memory import AChModulatedWorkingMemory
from BrainSimulationSystem.models.cognitive_controller import CognitiveControllerBuilder, CognitiveState


class CognitiveSimulation:
    """认知模拟测试类"""
    
    def __init__(self):
        """初始化认知模拟"""
        # 构建认知控制器
        self.controller = self._build_controller()
        
        # 模拟数据
        self.sensory_data = {}
        self.results = []
        self.time_step = 0
        
        # 可视化数据
        self.attention_history = []
        self.memory_history = []
        self.neuromod_history = []
        self.state_history = []
        
    def _build_controller(self):
        """构建认知控制器"""
        # 配置注意力系统参数
        attention_params = {
            "ach_sensitivity": 1.2,
            "bottom_up_weight": 0.6,
            "top_down_weight": 0.4,
            "attention_span": 3
        }
        
        # 配置工作记忆参数
        memory_params = {
            "capacity": 5,
            "decay_rate": 0.03,
            "ach_sensitivity": 1.5,
            "attention_boost": 0.4,
            "encoding_factor": 1.3,
            "retrieval_threshold": 0.35,
            "interference_factor": 0.75
        }
        
        # 使用构建器创建控制器
        builder = CognitiveControllerBuilder()
        builder.with_attention_params(attention_params)
        builder.with_working_memory_params(memory_params)
        
        return builder.build()
    
    def generate_sensory_input(self, time_step: int) -> Dict[str, Any]:
        """生成模拟的感觉输入"""
        # 模拟不同的感觉输入项
        sensory_input = {
            "visual_object_1": {"shape": "circle", "color": "red", "size": 0.8},
            "visual_object_2": {"shape": "square", "color": "blue", "size": 0.5},
            "visual_object_3": {"shape": "triangle", "color": "green", "size": 0.6},
            "auditory_input": {"frequency": 440, "volume": 0.7},
            "tactile_input": {"pressure": 0.3, "location": "left_hand"}
        }
        
        # 每3个时间步改变一个对象的属性，增加动态性
        if time_step % 3 == 0:
            # 改变颜色
            colors = ["red", "blue", "green", "yellow", "purple"]
            sensory_input["visual_object_1"]["color"] = colors[time_step % len(colors)]
        
        # 每5个时间步添加一个新对象
        if time_step % 5 == 0:
            sensory_input[f"new_object_{time_step}"] = {
                "shape": "star", 
                "color": "yellow", 
                "size": 0.9,
                "novelty": 1.0  # 新对象有高显著性
            }
            
        # 每7个时间步改变声音
        if time_step % 7 == 0:
            sensory_input["auditory_input"]["frequency"] = 220 + (time_step * 20) % 880
            sensory_input["auditory_input"]["volume"] = 0.5 + 0.5 * np.sin(time_step / 10)
            
        return sensory_input
    
    def modulate_neuromodulators(self, time_step: int) -> Dict[str, float]:
        """模拟神经调质水平变化"""
        # 基础水平
        base_level = 0.5
        
        # 模拟乙酰胆碱水平变化 (注意力相关)
        # 周期性变化，模拟注意力波动
        ach_level = base_level + 0.3 * np.sin(time_step / 10)
        
        # 模拟多巴胺水平变化 (奖励相关)
        # 在特定时间点有峰值，模拟奖励事件
        dopa_level = base_level
        if time_step % 15 == 0:
            dopa_level += 0.4  # 奖励峰值
        
        # 模拟去甲肾上腺素水平变化 (警觉相关)
        # 随机波动，模拟环境变化引起的警觉性变化
        ne_level = base_level + 0.2 * np.random.random()
        
        # 模拟5-羟色胺水平变化 (情绪相关)
        # 缓慢变化，模拟情绪状态
        serotonin_level = base_level + 0.1 * np.sin(time_step / 20)
        
        return {
            "acetylcholine": max(0.0, min(1.0, ach_level)),
            "dopamine": max(0.0, min(1.0, dopa_level)),
            "norepinephrine": max(0.0, min(1.0, ne_level)),
            "serotonin": max(0.0, min(1.0, serotonin_level))
        }
    
    def run_simulation(self, steps: int = 30) -> List[Dict[str, Any]]:
        """运行认知模拟"""
        print("开始认知模拟...")
        
        for step in range(steps):
            self.time_step = step
            print(f"\n时间步 {step}:")
            
            # 生成感觉输入
            sensory_input = self.generate_sensory_input(step)
            self.sensory_data = sensory_input
            
            # 模拟神经调质水平
            neuromodulators = self.modulate_neuromodulators(step)
            
            # 准备控制器输入
            controller_input = {
                "sensory_input": sensory_input,
                "neuromodulators": neuromodulators,
                "task_goal": "identify_important_objects",
                "decision_required": step % 10 == 0,  # 每10步需要一次决策
                "response_required": step % 10 == 5,  # 每10步需要一次响应
                "response_complete": step % 10 == 6   # 响应完成
            }
            
            # 处理认知周期
            result = self.controller.process(controller_input)
            self.results.append(result)
            
            # 记录历史数据用于可视化
            self._record_history(result)
            
            # 打印关键结果
            self._print_step_results(result)
            
        print("\n模拟完成!")
        return self.results
    
    def _record_history(self, result: Dict[str, Any]) -> None:
        """记录历史数据用于可视化"""
        # 记录注意力焦点
        if "components" in result and "attention" in result["components"]:
            attention_data = result["components"]["attention"]
            if "attention_focus" in attention_data:
                self.attention_history.append(attention_data["attention_focus"])
            else:
                self.attention_history.append([])
        elif "integrated_output" in result and "focus" in result["integrated_output"]:
            self.attention_history.append(result["integrated_output"]["focus"])
        else:
            self.attention_history.append([])
            
        # 记录工作记忆状态
        if "components" in result and "working_memory" in result["components"]:
            memory_data = result["components"]["working_memory"]
            if "memory_state" in memory_data:
                self.memory_history.append(list(memory_data["memory_state"].keys()))
            else:
                self.memory_history.append([])
        elif "integrated_output" in result and "memory_content" in result["integrated_output"]:
            self.memory_history.append(list(result["integrated_output"]["memory_content"].keys()))
        else:
            self.memory_history.append([])
            
        # 记录神经调质水平
        if "neuromodulators" in result:
            self.neuromod_history.append(result["neuromodulators"])
            
        # 记录认知状态
        if "cognitive_state" in result:
            self.state_history.append(result["cognitive_state"])
    
    def _print_step_results(self, result: Dict[str, Any]) -> None:
        """打印每个时间步的关键结果"""
        print(f"认知状态: {result['cognitive_state']}")
        
        # 打印神经调质水平
        print("神经调质水平:")
        for name, level in result["neuromodulators"].items():
            print(f"  - {name}: {level:.2f}")
            
        # 打印注意力焦点
        if "integrated_output" in result and "focus" in result["integrated_output"]:
            focus = result["integrated_output"]["focus"]
            print(f"注意力焦点: {focus}")
            
        # 打印工作记忆内容
        if "integrated_output" in result and "memory_content" in result["integrated_output"]:
            memory = result["integrated_output"]["memory_content"]
            print("工作记忆内容:")
            for key in memory:
                print(f"  - {key}")
    
    def visualize_results(self) -> None:
        """可视化模拟结果"""
        plt.figure(figsize=(15, 10))
        
        # 1. 绘制神经调质水平变化
        plt.subplot(3, 1, 1)
        x = range(len(self.neuromod_history))
        for neuromod in ["acetylcholine", "dopamine", "norepinephrine", "serotonin"]:
            values = [data[neuromod] for data in self.neuromod_history]
            plt.plot(x, values, label=neuromod)
        plt.title("神经调质水平变化")
        plt.xlabel("时间步")
        plt.ylabel("水平")
        plt.legend()
        plt.grid(True)
        
        # 2. 绘制注意力焦点变化
        plt.subplot(3, 1, 2)
        attention_data = []
        all_items = set()
        for items in self.attention_history:
            all_items.update(items)
        
        all_items = sorted(list(all_items))
        item_indices = {item: i for i, item in enumerate(all_items)}
        
        for step, items in enumerate(self.attention_history):
            for item in items:
                attention_data.append((step, item_indices[item]))
        
        if attention_data:
            x, y = zip(*attention_data)
            plt.scatter(x, y, marker='o')
            plt.yticks(range(len(all_items)), all_items)
            plt.title("注意力焦点变化")
            plt.xlabel("时间步")
            plt.ylabel("关注项")
            plt.grid(True)
        
        # 3. 绘制认知状态变化
        plt.subplot(3, 1, 3)
        states = list(CognitiveState.__members__.keys())
        state_indices = {state: i for i, state in enumerate(states)}
        state_data = [state_indices[state] for state in self.state_history]
        
        plt.plot(range(len(state_data)), state_data, marker='o')
        plt.yticks(range(len(states)), states)
        plt.title("认知状态变化")
        plt.xlabel("时间步")
        plt.ylabel("状态")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("BrainSimulationSystem/tests/cognitive_simulation_results.png")
        plt.close()
        
        print("可视化结果已保存到 'BrainSimulationSystem/tests/cognitive_simulation_results.png'")


def main():
    """主函数"""
    # 创建模拟
    simulation = CognitiveSimulation()
    
    # 运行模拟
    simulation.run_simulation(steps=30)
    
    # 可视化结果
    simulation.visualize_results()


if __name__ == "__main__":
    main()