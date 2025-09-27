"""
感知过程模块

实现将外部刺激转换为神经表示的感知功能。
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import random

from BrainSimulationSystem.core.network import NeuralNetwork
from BrainSimulationSystem.models.cognitive_base import CognitiveProcess


class PerceptionProcess(CognitiveProcess):
    """
    感知过程
    
    将外部刺激转换为神经表示
    """
    
    def __init__(self, network: NeuralNetwork, params: Dict[str, Any]):
        """
        初始化感知过程
        
        Args:
            network: 神经网络实例
            params: 参数字典，包含以下键：
                - input_mapping: 输入到神经元的映射方式
                - normalization: 输入归一化方式
                - noise_level: 噪声水平
        """
        super().__init__(network, params)
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理感知输入
        
        Args:
            inputs: 输入数据字典，包含以下键：
                - sensory_data: 感官数据（如视觉、听觉等）
                
        Returns:
            包含处理结果的字典
        """
        # 获取参数
        input_mapping = self.params.get("input_mapping", "direct")
        normalization = self.params.get("normalization", "minmax")
        noise_level = self.params.get("noise_level", 0.0)
        
        # 获取感官数据
        sensory_data = inputs.get("sensory_data", [])
        if not sensory_data:
            return {"perception_output": [], "neural_activity": {}}
        
        # 归一化输入
        normalized_data = self._normalize_input(sensory_data, normalization)
        
        # 添加噪声
        if noise_level > 0:
            normalized_data = self._add_noise(normalized_data, noise_level)
        
        # 映射到神经元
        neural_input = self._map_to_neurons(normalized_data, input_mapping)
        
        # 设置网络输入
        if self.network.input_layer_name:
            input_layer = self.network.layers[self.network.input_layer_name]
            input_size = min(len(neural_input), input_layer.size)
            
            # 截断或填充输入以匹配输入层大小
            if len(neural_input) > input_size:
                neural_input = neural_input[:input_size]
            elif len(neural_input) < input_size:
                neural_input.extend([0.0] * (input_size - len(neural_input)))
            
            # 设置网络输入
            self.network.set_input(neural_input)
        
        return {
            "perception_output": neural_input,
            "neural_activity": {
                neuron_id: self.network.neurons[neuron_id].voltage
                for neuron_id in self.network.neurons
                if self.network.input_layer_name and 
                neuron_id in self.network.layers[self.network.input_layer_name].neuron_ids
            }
        }
    
    def _normalize_input(self, data: List[float], method: str) -> List[float]:
        """
        归一化输入数据
        
        Args:
            data: 输入数据
            method: 归一化方法
            
        Returns:
            归一化后的数据
        """
        if not data:
            return []
        
        if method == "minmax":
            # Min-Max归一化
            min_val = min(data)
            max_val = max(data)
            if max_val == min_val:
                return [0.5] * len(data)
            return [(x - min_val) / (max_val - min_val) for x in data]
        
        elif method == "zscore":
            # Z-score归一化
            mean = sum(data) / len(data)
            std = np.sqrt(sum((x - mean) ** 2 for x in data) / len(data))
            if std == 0:
                return [0.0] * len(data)
            return [(x - mean) / std for x in data]
        
        elif method == "sigmoid":
            # Sigmoid归一化
            return [1.0 / (1.0 + np.exp(-x)) for x in data]
        
        else:
            # 默认不做归一化
            return data
    
    def _add_noise(self, data: List[float], noise_level: float) -> List[float]:
        """
        添加噪声
        
        Args:
            data: 输入数据
            noise_level: 噪声水平
            
        Returns:
            添加噪声后的数据
        """
        return [x + random.uniform(-noise_level, noise_level) for x in data]
    
    def _map_to_neurons(self, data: List[float], mapping: str) -> List[float]:
        """
        将数据映射到神经元输入
        
        Args:
            data: 输入数据
            mapping: 映射方法
            
        Returns:
            神经元输入
        """
        if mapping == "direct":
            # 直接映射
            return data
        
        elif mapping == "population":
            # 群体编码
            result = []
            for x in data:
                # 为每个值创建一个小型群体编码
                population_size = self.params.get("population_size", 5)
                mean = x
                std = self.params.get("population_std", 0.1)
                population = [random.normalvariate(mean, std) for _ in range(population_size)]
                result.extend(population)
            return result
        
        elif mapping == "sparse":
            # 稀疏编码
            sparsity = self.params.get("sparsity", 0.1)
            size = len(data) * 10  # 扩大输出大小
            result = [0.0] * size
            
            for i, x in enumerate(data):
                # 确定激活的神经元数量
                active_count = max(1, int(size * sparsity / len(data)))
                
                # 随机选择神经元激活
                indices = random.sample(range(i * 10, (i + 1) * 10), active_count)
                for idx in indices:
                    if idx < size:
                        result[idx] = x
            
            return result
        
        else:
            # 默认直接映射
            return data