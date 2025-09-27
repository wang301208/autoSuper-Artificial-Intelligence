"""
工作记忆模块

实现短期信息存储和处理的工作记忆系统，支持乙酰胆碱调节和注意力交互。
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from collections import OrderedDict
import time

from BrainSimulationSystem.core.network import NeuralNetwork
from BrainSimulationSystem.models.cognitive_base import CognitiveProcess


class WorkingMemory(CognitiveProcess):
    """
    工作记忆系统
    
    特性：
    1. 容量限制
    2. 时间衰减
    3. 乙酰胆碱调节的记忆保持
    4. 注意力增强的记忆项
    """
    
    def __init__(self, network: Optional[NeuralNetwork] = None, params: Dict[str, Any] = None):
        super().__init__(network, params or {})
        
        # 工作记忆参数
        self.capacity = params.get("capacity", 7)  # 默认容量限制
        self.decay_rate = params.get("decay_rate", 0.05)  # 记忆衰减率
        self.ach_sensitivity = params.get("ach_sensitivity", 1.0)  # 乙酰胆碱敏感度
        self.attention_boost = params.get("attention_boost", 0.3)  # 注意力增强系数
        
        # 工作记忆状态
        self.memory_items = OrderedDict()  # 记忆项
        self.item_strengths = {}  # 记忆强度
        self.last_access_time = {}  # 最后访问时间
        self.current_time = 0  # 内部时钟
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理工作记忆更新
        
        Args:
            inputs: 输入数据字典，包含：
                - sensory_input: 感觉输入
                - attention_focus: 注意力焦点
                - attention_gain: 注意力增益
                - neuromod.acetylcholine: ACh水平
                - command: 记忆操作命令
                
        Returns:
            工作记忆处理结果
        """
        # 更新内部时钟
        self.current_time += 1
        
        # 获取乙酰胆碱水平
        ach_level = inputs.get("neuromod.acetylcholine", 0.5)
        
        # 计算记忆保持因子
        retention_factor = self._calculate_retention(ach_level)
        
        # 处理记忆操作命令
        command = inputs.get("command", None)
        if command:
            self._process_command(command, inputs)
        
        # 处理新输入
        sensory_input = inputs.get("sensory_input", {})
        if sensory_input:
            self._update_memory(sensory_input, inputs)
        
        # 应用注意力增强
        attention_focus = inputs.get("attention_focus", [])
        attention_gain = inputs.get("attention_gain", 1.0)
        if attention_focus:
            self._apply_attention_boost(attention_focus, attention_gain)
        
        # 应用记忆衰减
        self._apply_decay(retention_factor)
        
        # 清理超出容量的记忆项
        self._cleanup_memory()
        
        # 返回当前工作记忆状态
        return {
            "memory_state": dict(self.memory_items),
            "memory_strengths": self.item_strengths.copy(),
            "capacity_used": len(self.memory_items),
            "capacity_total": self.capacity
        }
    
    def _calculate_retention(self, ach_level: float) -> float:
        """计算乙酰胆碱调节的记忆保持因子"""
        # 非线性保持函数
        base_retention = 0.9  # 基础保持率
        ach_effect = np.tanh(self.ach_sensitivity * ach_level)
        return base_retention + (1.0 - base_retention) * ach_effect
    
    def _process_command(self, command: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        """处理记忆操作命令"""
        cmd_type = command.get("type", "")
        
        if cmd_type == "store":
            # 存储新项目
            key = command.get("key", f"item_{self.current_time}")
            value = command.get("value", None)
            priority = command.get("priority", 0.5)
            
            if value is not None:
                self._store_item(key, value, priority)
                
        elif cmd_type == "retrieve":
            # 检索项目
            key = command.get("key", None)
            if key and key in self.memory_items:
                self._access_item(key)
                
        elif cmd_type == "update":
            # 更新项目
            key = command.get("key", None)
            value = command.get("value", None)
            
            if key and key in self.memory_items and value is not None:
                self._update_item(key, value)
                
        elif cmd_type == "delete":
            # 删除项目
            key = command.get("key", None)
            if key and key in self.memory_items:
                self._delete_item(key)
                
        elif cmd_type == "clear":
            # 清空工作记忆
            self.memory_items.clear()
            self.item_strengths.clear()
            self.last_access_time.clear()
    
    def _store_item(self, key: str, value: Any, priority: float = 0.5) -> None:
        """存储新记忆项"""
        self.memory_items[key] = value
        self.item_strengths[key] = priority
        self.last_access_time[key] = self.current_time
        
        # 将新项目移到队列末尾(最近使用)
        self.memory_items.move_to_end(key)
    
    def _access_item(self, key: str) -> None:
        """访问记忆项，更新强度和访问时间"""
        if key in self.memory_items:
            # 增强记忆强度
            self.item_strengths[key] = min(1.0, self.item_strengths[key] + 0.1)
            self.last_access_time[key] = self.current_time
            
            # 将访问的项目移到队列末尾(最近使用)
            self.memory_items.move_to_end(key)
    
    def _update_item(self, key: str, value: Any) -> None:
        """更新记忆项"""
        if key in self.memory_items:
            self.memory_items[key] = value
            self.last_access_time[key] = self.current_time
            
            # 将更新的项目移到队列末尾(最近使用)
            self.memory_items.move_to_end(key)
    
    def _delete_item(self, key: str) -> None:
        """删除记忆项"""
        if key in self.memory_items:
            del self.memory_items[key]
            
        if key in self.item_strengths:
            del self.item_strengths[key]
            
        if key in self.last_access_time:
            del self.last_access_time[key]
    
    def _update_memory(self, sensory_input: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        """更新工作记忆，处理新输入"""
        # 简单策略：将感觉输入直接添加到工作记忆
        for key, value in sensory_input.items():
            # 检查是否已存在
            if key in self.memory_items:
                # 更新现有项目
                self._update_item(key, value)
            else:
                # 存储新项目
                priority = 0.5  # 默认优先级
                self._store_item(key, value, priority)
    
    def _apply_attention_boost(self, attention_focus: List[str], attention_gain: float) -> None:
        """应用注意力增强到记忆项"""
        for key in attention_focus:
            if key in self.memory_items:
                # 增强注意力焦点项的强度
                boost = self.attention_boost * attention_gain
                self.item_strengths[key] = min(1.0, self.item_strengths[key] + boost)
                self.last_access_time[key] = self.current_time
                
                # 将注意的项目移到队列末尾(最近使用)
                self.memory_items.move_to_end(key)
    
    def _apply_decay(self, retention_factor: float) -> None:
        """应用记忆衰减"""
        for key in list(self.item_strengths.keys()):
            if key in self.memory_items:
                # 计算时间衰减
                time_elapsed = self.current_time - self.last_access_time[key]
                decay = self.decay_rate * time_elapsed
                
                # 应用保持因子减少衰减
                effective_decay = decay * (1.0 - retention_factor)
                
                # 更新强度
                self.item_strengths[key] = max(0.0, self.item_strengths[key] - effective_decay)
    
    def _cleanup_memory(self) -> None:
        """清理超出容量的记忆项"""
        # 如果超出容量，移除最弱的项目
        while len(self.memory_items) > self.capacity:
            # 找出强度最低的项目
            min_key = min(self.item_strengths.items(), key=lambda x: x[1])[0]
            self._delete_item(min_key)
    
    def get_item(self, key: str) -> Optional[Any]:
        """获取记忆项"""
        if key in self.memory_items:
            self._access_item(key)
            return self.memory_items[key]
        return None
    
    def get_all_items(self) -> Dict[str, Any]:
        """获取所有记忆项"""
        return dict(self.memory_items)


class AChModulatedWorkingMemory(WorkingMemory):
    """
    乙酰胆碱调节的工作记忆
    
    增强特性：
    1. 编码增强：高ACh促进新信息编码
    2. 检索调节：低ACh促进记忆检索
    3. 干扰抑制：高ACh减少干扰
    """
    
    def __init__(self, network: Optional[NeuralNetwork] = None, params: Dict[str, Any] = None):
        super().__init__(network, params)
        
        # 额外参数
        self.encoding_factor = params.get("encoding_factor", 1.2)  # 编码增强因子
        self.retrieval_threshold = params.get("retrieval_threshold", 0.4)  # 检索阈值
        self.interference_factor = params.get("interference_factor", 0.8)  # 干扰抑制因子
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """增强的工作记忆处理"""
        # 获取乙酰胆碱水平
        ach_level = inputs.get("neuromod.acetylcholine", 0.5)
        
        # 应用ACh特定效应
        self._apply_ach_effects(ach_level, inputs)
        
        # 调用基类处理
        result = super().process(inputs)
        
        # 添加ACh调节信息
        result["ach_level"] = ach_level
        result["encoding_strength"] = self._calculate_encoding_strength(ach_level)
        result["retrieval_efficiency"] = self._calculate_retrieval_efficiency(ach_level)
        
        return result
    
    def _apply_ach_effects(self, ach_level: float, inputs: Dict[str, Any]) -> None:
        """应用乙酰胆碱特定效应"""
        # 1. 编码增强
        if "sensory_input" in inputs:
            encoding_strength = self._calculate_encoding_strength(ach_level)
            inputs["encoding_boost"] = encoding_strength
            
        # 2. 检索调节
        if "command" in inputs and inputs["command"].get("type") == "retrieve":
            retrieval_efficiency = self._calculate_retrieval_efficiency(ach_level)
            inputs["retrieval_boost"] = retrieval_efficiency
            
        # 3. 干扰抑制
        interference_suppression = self._calculate_interference_suppression(ach_level)
        inputs["interference_suppression"] = interference_suppression
    
    def _calculate_encoding_strength(self, ach_level: float) -> float:
        """计算乙酰胆碱对编码强度的影响"""
        # 高ACh促进编码
        return self.encoding_factor * ach_level
    
    def _calculate_retrieval_efficiency(self, ach_level: float) -> float:
        """计算乙酰胆碱对检索效率的影响"""
        # 低ACh促进检索
        if ach_level < self.retrieval_threshold:
            # 低ACh时检索效率高
            return 1.0 + (self.retrieval_threshold - ach_level)
        else:
            # 高ACh时检索效率降低
            return max(0.5, 1.0 - (ach_level - self.retrieval_threshold))
    
    def _calculate_interference_suppression(self, ach_level: float) -> float:
        """计算乙酰胆碱对干扰抑制的影响"""
        # 高ACh减少干扰
        return self.interference_factor * ach_level
    
    def _update_memory(self, sensory_input: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        """增强的记忆更新，考虑编码增强"""
        encoding_boost = inputs.get("encoding_boost", 1.0)
        
        # 应用编码增强
        for key, value in sensory_input.items():
            if key in self.memory_items:
                # 更新现有项目
                self._update_item(key, value)
                # 增强记忆强度
                self.item_strengths[key] = min(1.0, self.item_strengths[key] * encoding_boost)
            else:
                # 存储新项目，应用编码增强
                priority = 0.5 * encoding_boost  # 增强的默认优先级
                self._store_item(key, value, priority)
    
    def get_item(self, key: str) -> Optional[Any]:
        """增强的记忆检索，考虑检索效率"""
        retrieval_boost = 1.0  # 默认检索增强
        
        if key in self.memory_items:
            # 应用检索增强
            strength_boost = self.item_strengths[key] * retrieval_boost
            self.item_strengths[key] = min(1.0, strength_boost)
            self._access_item(key)
            return self.memory_items[key]
        
        return None