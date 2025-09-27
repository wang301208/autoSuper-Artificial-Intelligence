"""
学习和记忆模块

实现各种学习算法和记忆形成过程，包括STDP、Hebbian学习、强化学习等。
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from abc import ABC, abstractmethod

from BrainSimulationSystem.core.network import NeuralNetwork


class LearningRule(ABC):
    """学习规则基类，定义所有学习规则的通用接口"""
    
    def __init__(self, network: NeuralNetwork, params: Dict[str, Any]):
        """
        初始化学习规则
        
        Args:
            network: 神经网络实例
            params: 学习参数字典
        """
        self.network = network
        self.params = params
    
    @abstractmethod
    def update(self, state: Dict[str, Any], dt: float) -> None:
        """
        更新网络权重
        
        Args:
            state: 网络状态字典
            dt: 时间步长
        """
        pass


class STDPLearning(LearningRule):
    """
    尖峰时间依赖可塑性学习规则
    
    根据前后神经元的脉冲时间差调整突触权重
    """
    
    def __init__(self, network: NeuralNetwork, params: Dict[str, Any]):
        """
        初始化STDP学习规则
        
        Args:
            network: 神经网络实例
            params: 学习参数字典，包含以下键：
                - learning_rate: 学习率
                - a_plus: 正时间窗口幅度
                - a_minus: 负时间窗口幅度
                - tau_plus: 正时间窗口时间常数 (ms)
                - tau_minus: 负时间窗口时间常数 (ms)
                - weight_min: 最小权重
                - weight_max: 最大权重
        """
        super().__init__(network, params)
        
        # 初始化神经元痕迹
        self.pre_traces = {neuron_id: 0.0 for neuron_id in self.network.neurons}
        self.post_traces = {neuron_id: 0.0 for neuron_id in self.network.neurons}
    
    def update(self, state: Dict[str, Any], dt: float) -> None:
        """
        更新网络权重
        
        Args:
            state: 网络状态字典
            dt: 时间步长
        """
        # 获取参数
        learning_rate = self.params.get("learning_rate", 0.01)
        a_plus = self.params.get("a_plus", 0.1)
        a_minus = self.params.get("a_minus", -0.1)
        tau_plus = self.params.get("tau_plus", 20.0)  # ms
        tau_minus = self.params.get("tau_minus", 20.0)  # ms
        weight_min = self.params.get("weight_min", 0.0)
        weight_max = self.params.get("weight_max", 1.0)
        
        # 获取当前脉冲
        spikes = state.get("spikes", [])
        
        # 更新所有神经元的痕迹
        for neuron_id in self.pre_traces:
            # 指数衰减
            self.pre_traces[neuron_id] *= np.exp(-dt / tau_plus)
            self.post_traces[neuron_id] *= np.exp(-dt / tau_minus)
            
            # 如果神经元产生脉冲，增加痕迹值
            if neuron_id in spikes:
                self.pre_traces[neuron_id] += 1.0
                self.post_traces[neuron_id] += 1.0
        
        # 更新突触权重
        for (pre_id, post_id), synapse in self.network.synapses.items():
            # 计算权重变化
            dw = 0.0
            
            # 前神经元脉冲后，根据后神经元痕迹调整权重
            if pre_id in spikes:
                dw += learning_rate * a_minus * self.post_traces[post_id]
            
            # 后神经元脉冲后，根据前神经元痕迹调整权重
            if post_id in spikes:
                dw += learning_rate * a_plus * self.pre_traces[pre_id]
            
            # 更新权重
            if dw != 0:
                new_weight = synapse.weight + dw
                new_weight = max(weight_min, min(new_weight, weight_max))
                
                # 直接修改突触权重
                synapse._weight = new_weight


class HebbianLearning(LearningRule):
    """
    Hebbian学习规则
    
    根据"同时激活的神经元会增强它们之间的连接"的原则调整权重
    """
    
    def __init__(self, network: NeuralNetwork, params: Dict[str, Any]):
        """
        初始化Hebbian学习规则
        
        Args:
            network: 神经网络实例
            params: 学习参数字典，包含以下键：
                - learning_rate: 学习率
                - weight_min: 最小权重
                - weight_max: 最大权重
                - decay_rate: 权重衰减率
        """
        super().__init__(network, params)
    
    def update(self, state: Dict[str, Any], dt: float) -> None:
        """
        更新网络权重
        
        Args:
            state: 网络状态字典
            dt: 时间步长
        """
        # 获取参数
        learning_rate = self.params.get("learning_rate", 0.01)
        weight_min = self.params.get("weight_min", 0.0)
        weight_max = self.params.get("weight_max", 1.0)
        decay_rate = self.params.get("decay_rate", 0.0001)
        
        # 获取当前脉冲
        spikes = set(state.get("spikes", []))
        
        # 更新突触权重
        for (pre_id, post_id), synapse in self.network.synapses.items():
            # 如果前后神经元同时激活，增强连接
            if pre_id in spikes and post_id in spikes:
                dw = learning_rate
            # 否则，轻微衰减连接
            else:
                dw = -decay_rate
            
            # 更新权重
            new_weight = synapse.weight + dw
            new_weight = max(weight_min, min(new_weight, weight_max))
            
            # 直接修改突触权重
            synapse._weight = new_weight


class BCMLearning(LearningRule):
    """
    BCM (Bienenstock-Cooper-Munro) 学习规则
    
    一种基于后突触神经元活动的学习规则，具有稳定性和竞争性
    """
    
    def __init__(self, network: NeuralNetwork, params: Dict[str, Any]):
        """
        初始化BCM学习规则
        
        Args:
            network: 神经网络实例
            params: 学习参数字典，包含以下键：
                - learning_rate: 学习率
                - target_rate: 目标活动率
                - sliding_threshold_tau: 滑动阈值时间常数
                - weight_min: 最小权重
                - weight_max: 最大权重
        """
        super().__init__(network, params)
        
        # 初始化神经元活动率和阈值
        self.activity = {neuron_id: 0.0 for neuron_id in self.network.neurons}
        self.thresholds = {neuron_id: self.params.get("target_rate", 0.1) 
                          for neuron_id in self.network.neurons}
    
    def update(self, state: Dict[str, Any], dt: float) -> None:
        """
        更新网络权重
        
        Args:
            state: 网络状态字典
            dt: 时间步长
        """
        # 获取参数
        learning_rate = self.params.get("learning_rate", 0.01)
        target_rate = self.params.get("target_rate", 0.1)
        sliding_tau = self.params.get("sliding_threshold_tau", 1000.0)  # ms
        weight_min = self.params.get("weight_min", 0.0)
        weight_max = self.params.get("weight_max", 1.0)
        
        # 获取当前脉冲
        spikes = set(state.get("spikes", []))
        
        # 更新神经元活动率
        for neuron_id in self.activity:
            # 指数滤波更新活动率
            tau_activity = 100.0  # ms
            decay = np.exp(-dt / tau_activity)
            
            # 如果神经元产生脉冲，增加活动率
            if neuron_id in spikes:
                self.activity[neuron_id] = decay * self.activity[neuron_id] + (1.0 - decay)
            else:
                self.activity[neuron_id] *= decay
            
            # 更新滑动阈值
            threshold_decay = np.exp(-dt / sliding_tau)
            self.thresholds[neuron_id] = threshold_decay * self.thresholds[neuron_id] + \
                                        (1.0 - threshold_decay) * (self.activity[neuron_id]**2)
        
        # 更新突触权重
        for (pre_id, post_id), synapse in self.network.synapses.items():
            # 获取前神经元活动和后神经元阈值
            pre_activity = 1.0 if pre_id in spikes else 0.0
            post_activity = self.activity[post_id]
            post_threshold = self.thresholds[post_id]
            
            # BCM规则：dw = η * o * (o - θ) * x
            # 其中o是后神经元活动，θ是阈值，x是前神经元活动
            dw = learning_rate * post_activity * (post_activity - post_threshold) * pre_activity
            
            # 更新权重
            new_weight = synapse.weight + dw
            new_weight = max(weight_min, min(new_weight, weight_max))
            
            # 直接修改突触权重
            synapse._weight = new_weight


class OjaLearning(LearningRule):
    """
    Oja学习规则
    
    Hebbian学习的一种变体，具有自我正则化特性，可以防止权重无限增长
    """
    
    def __init__(self, network: NeuralNetwork, params: Dict[str, Any]):
        """
        初始化Oja学习规则
        
        Args:
            network: 神经网络实例
            params: 学习参数字典，包含以下键：
                - learning_rate: 学习率
                - weight_min: 最小权重
                - weight_max: 最大权重
        """
        super().__init__(network, params)
    
    def update(self, state: Dict[str, Any], dt: float) -> None:
        """
        更新网络权重
        
        Args:
            state: 网络状态字典
            dt: 时间步长
        """
        # 获取参数
        learning_rate = self.params.get("learning_rate", 0.01)
        weight_min = self.params.get("weight_min", 0.0)
        weight_max = self.params.get("weight_max", 1.0)
        
        # 获取当前脉冲和电压
        spikes = set(state.get("spikes", []))
        voltages = state.get("voltages", {})
        
        # 更新突触权重
        for (pre_id, post_id), synapse in self.network.synapses.items():
            # 获取前后神经元活动
            pre_activity = 1.0 if pre_id in spikes else 0.0
            post_activity = voltages.get(post_id, 0.0)
            
            # Oja规则：dw = η * y * (x - y * w)
            # 其中y是后神经元活动，x是前神经元活动，w是当前权重
            dw = learning_rate * post_activity * (pre_activity - post_activity * synapse.weight)
            
            # 更新权重
            new_weight = synapse.weight + dw
            new_weight = max(weight_min, min(new_weight, weight_max))
            
            # 直接修改突触权重
            synapse._weight = new_weight


class HomeostaticPlasticity(LearningRule):
    """
    稳态可塑性
    
    调整神经元的内在特性，使其保持在目标活动水平
    """
    
    def __init__(self, network: NeuralNetwork, params: Dict[str, Any]):
        """
        初始化稳态可塑性
        
        Args:
            network: 神经网络实例
            params: 学习参数字典，包含以下键：
                - learning_rate: 学习率
                - target_rate: 目标活动率
                - time_window: 活动率计算的时间窗口 (ms)
        """
        super().__init__(network, params)
        
        # 初始化神经元活动率
        self.activity_rates = {neuron_id: 0.0 for neuron_id in self.network.neurons}
        self.spike_counts = {neuron_id: 0 for neuron_id in self.network.neurons}
        self.time_elapsed = 0.0
    
    def update(self, state: Dict[str, Any], dt: float) -> None:
        """
        更新网络权重
        
        Args:
            state: 网络状态字典
            dt: 时间步长
        """
        # 获取参数
        learning_rate = self.params.get("learning_rate", 0.001)
        target_rate = self.params.get("target_rate", 0.01)  # 目标活动率 (spikes/ms)
        time_window = self.params.get("time_window", 1000.0)  # ms
        
        # 更新时间
        self.time_elapsed += dt
        
        # 获取当前脉冲
        spikes = state.get("spikes", [])
        
        # 更新脉冲计数
        for neuron_id in spikes:
            if neuron_id in self.spike_counts:
                self.spike_counts[neuron_id] += 1
        
        # 如果达到时间窗口，计算活动率并调整阈值
        if self.time_elapsed >= time_window:
            for neuron_id, count in self.spike_counts.items():
                # 计算活动率 (spikes/ms)
                rate = count / self.time_elapsed
                self.activity_rates[neuron_id] = rate
                
                # 获取神经元
                if neuron_id in self.network.neurons:
                    neuron = self.network.neurons[neuron_id]
                    
                    # 调整阈值
                    if hasattr(neuron, 'params') and 'threshold' in neuron.params:
                        # 如果活动率高于目标，增加阈值；否则降低阈值
                        threshold_change = learning_rate * (rate - target_rate)
                        neuron.params["threshold"] += threshold_change
                        
                        # 确保阈值在合理范围内
                        neuron.params["threshold"] = max(0.1, neuron.params["threshold"])
            
            # 重置计数和时间
            self.spike_counts = {neuron_id: 0 for neuron_id in self.network.neurons}
            self.time_elapsed = 0.0


def create_learning_rule(rule_type: str, network: NeuralNetwork, params: Dict[str, Any]) -> LearningRule:
    """
    创建指定类型的学习规则
    
    Args:
        rule_type: 学习规则类型，可选值：'stdp', 'hebbian', 'bcm', 'oja', 'homeostatic'
        network: 神经网络实例
        params: 学习参数字典
        
    Returns:
        创建的学习规则实例
        
    Raises:
        ValueError: 如果指定的学习规则类型不支持
    """
    learning_classes = {
        'stdp': STDPLearning,
        'hebbian': HebbianLearning,
        'bcm': BCMLearning,
        'oja': OjaLearning,
        'homeostatic': HomeostaticPlasticity,
    }
    
    if rule_type not in learning_classes:
        raise ValueError(f"不支持的学习规则类型: {rule_type}")
    
    return learning_classes[rule_type](network, params)