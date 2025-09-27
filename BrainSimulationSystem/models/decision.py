"""
决策过程模块

实现基于输入和内部状态做出决策的功能。
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import random

from BrainSimulationSystem.core.network import NeuralNetwork
from BrainSimulationSystem.models.cognitive_base import CognitiveProcess


class DecisionProcess(CognitiveProcess):
    """
    决策过程
    
    基于输入和内部状态做出决策
    """
    
    def __init__(self, network: NeuralNetwork, params: Dict[str, Any]):
        """
        初始化决策过程
        
        Args:
            network: 神经网络实例
            params: 参数字典，包含以下键：
                - decision_type: 决策类型
                - temperature: 决策温度（用于探索-利用权衡）
                - threshold: 决策阈值
        """
        super().__init__(network, params)
        
        # 决策历史
        self.decision_history = []
        
        # 奖励历史
        self.reward_history = []
        
        # 动作价值估计
        self.action_values = {}
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理决策
        
        Args:
            inputs: 输入数据字典，包含以下键：
                - options: 决策选项
                - context: 决策上下文
                - reward: 上一次决策的奖励（可选）
                
        Returns:
            包含处理结果的字典
        """
        # 获取参数
        decision_type = self.params.get("decision_type", "softmax")
        temperature = self.params.get("temperature", 1.0)
        threshold = self.params.get("threshold", 0.5)
        
        # 获取决策选项
        options = inputs.get("options", [])
        if not options:
            return {"decision": None, "confidence": 0.0}
        
        # 获取上下文
        context = inputs.get("context", {})
        context_key = str(hash(str(context)))
        
        # 处理奖励（如果有）
        if "reward" in inputs:
            self._process_reward(inputs["reward"], context_key)
        
        # 做出决策
        if decision_type == "softmax":
            # Softmax决策：基于价值的概率选择
            decision, confidence = self._softmax_decision(options, context_key, temperature)
        
        elif decision_type == "greedy":
            # 贪婪决策：选择价值最高的选项
            decision, confidence = self._greedy_decision(options, context_key)
        
        elif decision_type == "threshold":
            # 阈值决策：如果最高价值超过阈值则选择，否则随机
            decision, confidence = self._threshold_decision(options, context_key, threshold)
        
        else:
            # 默认随机决策
            decision = random.choice(options)
            confidence = 1.0 / len(options)
        
        # 记录决策
        self.decision_history.append((decision, context_key))
        
        return {
            "decision": decision,
            "confidence": confidence,
            "action_values": {opt: self.action_values.get((context_key, opt), 0.0) for opt in options}
        }
    
    def _process_reward(self, reward: float, context_key: str) -> None:
        """
        处理奖励
        
        Args:
            reward: 奖励值
            context_key: 上下文键
        """
        # 记录奖励
        self.reward_history.append(reward)
        
        # 如果有决策历史，更新最近决策的价值
        if self.decision_history:
            last_decision, last_context = self.decision_history[-1]
            
            # 只有在相同上下文下才更新
            if last_context == context_key:
                # 获取当前价值估计
                key = (last_context, last_decision)
                current_value = self.action_values.get(key, 0.0)
                
                # 学习率
                alpha = self.params.get("learning_rate", 0.1)
                
                # 更新价值估计：Q(s,a) = Q(s,a) + α[r - Q(s,a)]
                new_value = current_value + alpha * (reward - current_value)
                self.action_values[key] = new_value
    
    def _softmax_decision(
        self, 
        options: List[Any], 
        context_key: str, 
        temperature: float
    ) -> Tuple[Any, float]:
        """
        Softmax决策
        
        Args:
            options: 决策选项
            context_key: 上下文键
            temperature: 温度参数
            
        Returns:
            选择的选项和置信度
        """
        # 获取每个选项的价值
        values = [self.action_values.get((context_key, opt), 0.0) for opt in options]
        
        # 应用softmax
        if temperature <= 0:
            temperature = 0.01  # 避免除以零
        
        # 减去最大值以提高数值稳定性
        max_value = max(values)
        exp_values = [np.exp((v - max_value) / temperature) for v in values]
        sum_exp = sum(exp_values)
        
        if sum_exp == 0:
            # 如果所有指数值都是0，使用均匀分布
            probabilities = [1.0 / len(options) for _ in options]
        else:
            probabilities = [ev / sum_exp for ev in exp_values]
        
        # 根据概率选择
        choice_idx = random.choices(range(len(options)), weights=probabilities)[0]
        choice = options[choice_idx]
        confidence = probabilities[choice_idx]
        
        return choice, confidence
    
    def _greedy_decision(
        self, 
        options: List[Any], 
        context_key: str
    ) -> Tuple[Any, float]:
        """
        贪婪决策
        
        Args:
            options: 决策选项
            context_key: 上下文键
            
        Returns:
            选择的选项和置信度
        """
        # 获取每个选项的价值
        values = [self.action_values.get((context_key, opt), 0.0) for opt in options]
        
        # 找到最大价值的索引
        max_value = max(values)
        max_indices = [i for i, v in enumerate(values) if v == max_value]
        
        # 如果有多个最大值，随机选择一个
        choice_idx = random.choice(max_indices)
        choice = options[choice_idx]
        
        # 计算置信度：最大值与平均值的差距
        avg_value = sum(values) / len(values)
        if avg_value == max_value:
            confidence = 1.0 / len(options)
        else:
            # 归一化置信度到[0,1]
            confidence = min(1.0, max(0.0, (max_value - avg_value) / max_value))
        
        return choice, confidence
    
    def _threshold_decision(
        self, 
        options: List[Any], 
        context_key: str, 
        threshold: float
    ) -> Tuple[Any, float]:
        """
        阈值决策
        
        Args:
            options: 决策选项
            context_key: 上下文键
            threshold: 决策阈值
            
        Returns:
            选择的选项和置信度
        """
        # 获取每个选项的价值
        values = [self.action_values.get((context_key, opt), 0.0) for opt in options]
        
        # 找到最大价值的索引
        max_value = max(values)
        max_indices = [i for i, v in enumerate(values) if v == max_value]
        
        # 如果最大值超过阈值，选择它
        if max_value >= threshold:
            choice_idx = random.choice(max_indices)
            choice = options[choice_idx]
            confidence = max_value
        else:
            # 否则随机选择
            choice_idx = random.randrange(len(options))
            choice = options[choice_idx]
            confidence = max_value / threshold  # 归一化置信度
        
        return choice, confidence