"""
大脑模拟系统主模块

整合神经元网络、突触连接、学习规则和认知过程，提供统一的接口。
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import time
import threading
import json
import os

from BrainSimulationSystem.core.network import NeuralNetwork
from BrainSimulationSystem.core.neurons import create_neuron
from BrainSimulationSystem.core.synapses import create_synapse
from BrainSimulationSystem.core.learning import create_learning_rule
from BrainSimulationSystem.models.perception import PerceptionProcess
from BrainSimulationSystem.models.attention import AttentionProcess
from BrainSimulationSystem.models.memory import MemoryProcess
from BrainSimulationSystem.models.decision import DecisionProcess
from BrainSimulationSystem.config.default_config import get_config, update_config


class BrainSimulation:
    """
    大脑模拟系统
    
    整合神经元网络、突触连接、学习规则和认知过程，提供统一的接口。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化大脑模拟系统
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        # 加载配置
        self.config = get_config()
        if config:
            self.config = update_config(config)
        
        # 创建神经网络
        self.network = NeuralNetwork(self.config["network"])
        
        # 创建学习规则
        self.learning_rules = []
        for rule_name, rule_params in self.config["network"]["learning_rules"].items():
            if rule_params.get("enabled", False):
                rule = create_learning_rule(rule_name, self.network, rule_params)
                self.learning_rules.append(rule)
        
        # 创建认知过程
        self.perception = PerceptionProcess(self.network, self.config.get("perception", {}))
        self.attention = AttentionProcess(self.network, self.config.get("attention", {}))
        self.memory = MemoryProcess(self.network, self.config.get("memory", {}))
        self.decision = DecisionProcess(self.network, self.config.get("decision", {}))
        
        # 模拟状态
        self.is_running = False
        self.simulation_thread = None
        self.current_time = 0.0
        self.simulation_results = {
            "times": [],
            "spikes": [],
            "voltages": [],
            "weights": [],
            "cognitive_states": []
        }
        
        # 事件回调
        self.event_callbacks = {}
    
    def reset(self) -> None:
        """重置模拟状态"""
        # 重置网络
        self.network.reset()
        
        # 重置模拟状态
        self.current_time = 0.0
        self.simulation_results = {
            "times": [],
            "spikes": [],
            "voltages": [],
            "weights": [],
            "cognitive_states": []
        }
    
    def step(self, inputs: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """
        执行一步模拟
        
        Args:
            inputs: 输入数据字典
            dt: 时间步长
            
        Returns:
            包含模拟结果的字典
        """
        # 更新时间
        self.current_time += dt
        
        # 处理感知输入
        perception_result = self.perception.process(inputs)
        
        # 处理注意力
        attention_result = self.attention.process({
            "perception_output": perception_result["perception_output"],
            "focus_position": inputs.get("focus_position", 0.5)
        })
        
        # 处理记忆
        memory_result = self.memory.process({
            "store": inputs.get("memory_store", None),
            "retrieve": inputs.get("memory_retrieve", None)
        })
        
        # 处理决策
        decision_result = self.decision.process({
            "options": inputs.get("decision_options", []),
            "context": inputs.get("decision_context", {}),
            "reward": inputs.get("reward", None)
        })
        
        # 执行网络更新
        network_state = self.network.step(dt)
        
        # 应用学习规则
        for rule in self.learning_rules:
            rule.update(network_state, dt)
        
        # 记录结果
        self.simulation_results["times"].append(self.current_time)
        self.simulation_results["spikes"].append(network_state["spikes"])
        self.simulation_results["voltages"].append(network_state["voltages"])
        self.simulation_results["weights"].append(network_state["weights"])
        
        # 记录认知状态
        cognitive_state = {
            "perception": perception_result,
            "attention": attention_result,
            "memory": memory_result,
            "decision": decision_result
        }
        self.simulation_results["cognitive_states"].append(cognitive_state)
        
        # 触发事件
        self._trigger_event("step", {
            "time": self.current_time,
            "network_state": network_state,
            "cognitive_state": cognitive_state
        })
        
        return {
            "time": self.current_time,
            "network_state": network_state,
            "cognitive_state": cognitive_state
        }
    
    def run(self, 
            inputs_sequence: List[Dict[str, Any]], 
            duration: float, 
            dt: float) -> Dict[str, Any]:
        """
        运行模拟
        
        Args:
            inputs_sequence: 输入序列，每个时间步的输入数据字典
            duration: 模拟持续时间
            dt: 时间步长
            
        Returns:
            包含模拟结果的字典
        """
        # 重置模拟状态
        self.reset()
        
        # 计算时间步数
        steps = int(duration / dt)
        
        # 触发事件
        self._trigger_event("simulation_start", {
            "duration": duration,
            "dt": dt,
            "steps": steps
        })
        
        # 执行模拟
        for step in range(steps):
            # 获取当前时间步的输入
            inputs = {}
            if step < len(inputs_sequence):
                inputs = inputs_sequence[step]
            
            # 执行一步模拟
            self.step(inputs, dt)
        
        # 触发事件
        self._trigger_event("simulation_end", {
            "results": self.simulation_results
        })
        
        return self.simulation_results
    
    def start_continuous_simulation(self, dt: float) -> None:
        """
        开始连续模拟
        
        Args:
            dt: 时间步长
        """
        if self.is_running:
            return
        
        self.is_running = True
        
        # 创建模拟线程
        self.simulation_thread = threading.Thread(
            target=self._continuous_simulation_loop,
            args=(dt,)
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # 触发事件
        self._trigger_event("continuous_simulation_start", {
            "dt": dt
        })
    
    def stop_continuous_simulation(self) -> None:
        """停止连续模拟"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 等待线程结束
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)
            self.simulation_thread = None
        
        # 触发事件
        self._trigger_event("continuous_simulation_stop", {
            "results": self.simulation_results
        })
    
    def _continuous_simulation_loop(self, dt: float) -> None:
        """
        连续模拟循环
        
        Args:
            dt: 时间步长
        """
        while self.is_running:
            # 执行一步模拟
            self.step({}, dt)
            
            # 控制模拟速度
            time.sleep(dt / 1000.0)  # 将毫秒转换为秒
    
    def register_event_callback(self, event_name: str, callback: Callable) -> None:
        """
        注册事件回调
        
        Args:
            event_name: 事件名称
            callback: 回调函数
        """
        if event_name not in self.event_callbacks:
            self.event_callbacks[event_name] = []
        
        self.event_callbacks[event_name].append(callback)
    
    def _trigger_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """
        触发事件
        
        Args:
            event_name: 事件名称
            event_data: 事件数据
        """
        if event_name in self.event_callbacks:
            for callback in self.event_callbacks[event_name]:
                try:
                    callback(event_data)
                except Exception as e:
                    print(f"事件回调错误: {e}")
    
    def save_state(self, filepath: str) -> None:
        """
        保存模拟状态
        
        Args:
            filepath: 文件路径
        """
        # 创建状态字典
        state = {
            "config": self.config,
            "current_time": self.current_time,
            "network_state": {
                "neuron_states": {
                    neuron_id: {
                        "voltage": neuron.voltage,
                        "spike_history": neuron.spike_history if hasattr(neuron, "spike_history") else []
                    }
                    for neuron_id, neuron in self.network.neurons.items()
                },
                "synapse_states": {
                    f"{synapse.pre_id}_{synapse.post_id}": {
                        "weight": synapse.weight
                    }
                    for synapse in self.network.synapses.values()
                }
            },
            "cognitive_state": {
                "memory": {
                    "working_memory": self.memory.working_memory,
                    "long_term_memory": self.memory.long_term_memory,
                    "memory_strengths": self.memory.memory_strengths
                },
                "decision": {
                    "action_values": self.decision.action_values,
                    "decision_history": self.decision.decision_history
                }
            }
        }
        
        # 保存到文件
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str) -> None:
        """
        加载模拟状态
        
        Args:
            filepath: 文件路径
        """
        # 从文件加载
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # 更新配置
        self.config = state["config"]
        
        # 重新创建网络
        self.network = NeuralNetwork(self.config["network"])
        
        # 更新时间
        self.current_time = state["current_time"]
        
        # 更新神经元状态
        for neuron_id, neuron_state in state["network_state"]["neuron_states"].items():
            neuron_id = int(neuron_id)
            if neuron_id in self.network.neurons:
                neuron = self.network.neurons[neuron_id]
                # 设置电压
                if hasattr(neuron, "_voltage"):
                    neuron._voltage = neuron_state["voltage"]
                elif hasattr(neuron, "_v"):
                    neuron._v = neuron_state["voltage"]
                
                # 设置脉冲历史
                if hasattr(neuron, "_spike_history"):
                    neuron._spike_history = neuron_state["spike_history"]
        
        # 更新突触状态
        for synapse_key, synapse_state in state["network_state"]["synapse_states"].items():
            pre_id, post_id = map(int, synapse_key.split("_"))
            if (pre_id, post_id) in self.network.synapses:
                synapse = self.network.synapses[(pre_id, post_id)]
                # 设置权重
                if hasattr(synapse, "_weight"):
                    synapse._weight = synapse_state["weight"]
        
        # 更新认知状态
        # 记忆
        self.memory.working_memory = state["cognitive_state"]["memory"]["working_memory"]
        self.memory.long_term_memory = state["cognitive_state"]["memory"]["long_term_memory"]
        self.memory.memory_strengths = state["cognitive_state"]["memory"]["memory_strengths"]
        
        # 决策
        self.decision.action_values = state["cognitive_state"]["decision"]["action_values"]
        self.decision.decision_history = state["cognitive_state"]["decision"]["decision_history"]