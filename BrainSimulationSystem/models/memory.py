"""
记忆过程模块

实现存储和检索信息的记忆功能。
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np

from BrainSimulationSystem.core.network import NeuralNetwork
from BrainSimulationSystem.models.cognitive_base import CognitiveProcess


class MemoryProcess(CognitiveProcess):
    """
    记忆过程
    
    存储和检索信息
    """

class WorkingMemory(MemoryProcess):
    """工作记忆模块
    
    特性：
    1. 多巴胺门控机制
    2. 奖励关联增强
    3. 容量限制
    """
    
    def __init__(self, network: NeuralNetwork, params: Dict[str, Any]):
        super().__init__(network, params)
        self.dopa_sensitivity = params.get("dopa_sensitivity", 1.0)
        self.reward_weights = {}  # 记忆项-奖励权重映射
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # 获取多巴胺水平
        dopa_level = inputs.get("neuromod.dopamine", 0.0)
        
        # 计算门控强度 (sigmoid函数)
        gate = 1 / (1 + np.exp(-self.dopa_sensitivity * dopa_level))
        
        # 处理记忆存储
        if "store" in inputs:
            mem_key = str(inputs["store"])
            reward = inputs.get("reward", 0.0)
            
            # 更新奖励权重
            self.reward_weights[mem_key] = (
                0.9 * self.reward_weights.get(mem_key, 0.0) + 
                0.1 * reward
            )
            
            # 应用多巴胺门控存储
            if gate > np.random.rand():
                self._store_memory(inputs["store"], gate * (1 + self.reward_weights[mem_key]))
        
        # 处理记忆检索
        result = super().process(inputs)
        result["gate_strength"] = gate
        return result
>>>>>>> 在文件末尾添加新类
=======
class MemoryProcess(CognitiveProcess):
    """
    记忆过程
    
    存储和检索信息
    """

class WorkingMemory(MemoryProcess):
    """工作记忆模块
    
    特性：
    1. 多巴胺门控机制
    2. 奖励关联增强
    3. 容量限制
    """
    
    def __init__(self, network: NeuralNetwork, params: Dict[str, Any]):
        super().__init__(network, params)
        self.dopa_sensitivity = params.get("dopa_sensitivity", 1.0)
        self.reward_weights = {}  # 记忆项-奖励权重映射
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # 获取多巴胺水平
        dopa_level = inputs.get("neuromod.dopamine", 0.0)
        
        # 计算门控强度 (sigmoid函数)
        gate = 1 / (1 + np.exp(-self.dopa_sensitivity * dopa_level))
        
        # 处理记忆存储
        if "store" in inputs:
            mem_key = str(inputs["store"])
            reward = inputs.get("reward", 0.0)
            
            # 更新奖励权重
            self.reward_weights[mem_key] = (
                0.9 * self.reward_weights.get(mem_key, 0.0) + 
                0.1 * reward
            )
            
            # 应用多巴胺门控存储
            if gate > np.random.rand():
                self._store_memory(inputs["store"], gate * (1 + self.reward_weights[mem_key]))
        
        # 处理记忆检索
        result = super().process(inputs)
        result["gate_strength"] = gate
        return result
    
    def __init__(self, network: NeuralNetwork, params: Dict[str, Any]):
        """
        初始化记忆过程
        
        Args:
            network: 神经网络实例
            params: 参数字典，包含以下键：
                - memory_type: 记忆类型
                - capacity: 记忆容量
                - decay_rate: 记忆衰减率
        """
        super().__init__(network, params)
        
        # 初始化记忆存储
        self.memory_type = self.params.get("memory_type", "working")
        self.capacity = self.params.get("capacity", 10)
        
        # 工作记忆
        self.working_memory = []
        
        # 长期记忆
        self.long_term_memory = []
        
        # 记忆强度（用于衰减）
        self.memory_strengths = {}
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理记忆
        
        Args:
            inputs: 输入数据字典，包含以下键：
                - store: 要存储的数据（可选）
                - retrieve: 检索查询（可选）
                - clear: 是否清除记忆（可选）
                
        Returns:
            包含处理结果的字典
        """
        # 获取参数
        decay_rate = self.params.get("decay_rate", 0.01)
        
        # 清除记忆
        if inputs.get("clear", False):
            if self.memory_type == "working":
                self.working_memory = []
            else:
                # 长期记忆不会完全清除，只会减弱
                for key in self.memory_strengths:
                    self.memory_strengths[key] *= 0.5
        
        # 存储数据
        if "store" in inputs:
            self._store_memory(inputs["store"])
        
        # 应用记忆衰减
        self._apply_decay(decay_rate)
        
        # 检索数据
        retrieved_data = None
        if "retrieve" in inputs:
            retrieved_data = self._retrieve_memory(inputs["retrieve"])
        
        return {
            "retrieved_data": retrieved_data,
            "memory_state": {
                "working_memory": self.working_memory,
                "long_term_memory": self.long_term_memory,
                "memory_strengths": self.memory_strengths
            }
        }
    
    def _store_memory(self, data: Any) -> None:
        """
        存储记忆
        
        Args:
            data: 要存储的数据
        """
        if self.memory_type == "working":
            # 工作记忆：有限容量，新数据替换旧数据
            self.working_memory.append(data)
            if len(self.working_memory) > self.capacity:
                self.working_memory.pop(0)
        else:
            # 长期记忆：无限容量，但有强度衰减
            memory_key = str(hash(str(data)))
            
            # 如果是新记忆，添加到长期记忆
            if memory_key not in self.memory_strengths:
                self.long_term_memory.append(data)
                self.memory_strengths[memory_key] = 1.0
            else:
                # 如果是已有记忆，增强强度
                self.memory_strengths[memory_key] = min(1.0, self.memory_strengths[memory_key] + 0.2)
    
    def _retrieve_memory(self, query: Any) -> Any:
        """
        检索记忆
        
        Args:
            query: 检索查询
            
        Returns:
            检索到的数据，如果没有找到则返回None
        """
        if self.memory_type == "working":
            # 工作记忆：直接返回最近的记忆
            if self.working_memory:
                return self.working_memory[-1]
        else:
            # 长期记忆：基于相似度检索
            if not self.long_term_memory:
                return None
            
            # 计算查询与每个记忆的相似度
            similarities = []
            for memory in self.long_term_memory:
                # 这里使用简单的字符串相似度作为示例
                # 实际应用中应该使用更复杂的相似度度量
                memory_key = str(hash(str(memory)))
                similarity = self._calculate_similarity(query, memory)
                
                # 考虑记忆强度
                strength = self.memory_strengths.get(memory_key, 0.0)
                weighted_similarity = similarity * strength
                
                similarities.append((weighted_similarity, memory))
            
            # 返回最相似的记忆
            if similarities:
                similarities.sort(reverse=True)
                return similarities[0][1]
        
        return None
    
    def _calculate_similarity(self, a: Any, b: Any) -> float:
        """
        计算两个数据的相似度
        
        Args:
            a: 第一个数据
            b: 第二个数据
            
        Returns:
            相似度 (0-1)
        """
        # 这里使用简单的字符串相似度作为示例
        # 实际应用中应该使用更复杂的相似度度量
        str_a = str(a)
        str_b = str(b)
        
        # Jaccard相似度
        set_a = set(str_a)
        set_b = set(str_b)
        
        if not set_a and not set_b:
            return 1.0
        
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        
        return intersection / union
    
    def _apply_decay(self, decay_rate: float) -> None:
        """
        应用记忆衰减
        
        Args:
            decay_rate: 衰减率
        """
        if self.memory_type == "long_term":
            # 长期记忆衰减
            keys_to_remove = []
            
            for key, strength in self.memory_strengths.items():
                # 应用衰减
                new_strength = strength * (1.0 - decay_rate)
                self.memory_strengths[key] = new_strength
                
                # 如果强度太低，标记为删除
                if new_strength < 0.1:
                    keys_to_remove.append(key)
            
            # 删除强度太低的记忆
            for key in keys_to_remove:
                del self.memory_strengths[key]
                
                # 从长期记忆中删除
                for i, memory in enumerate(self.long_term_memory):
                    memory_key = str(hash(str(memory)))
                    if memory_key == key:
                        self.long_term_memory.pop(i)
                        break