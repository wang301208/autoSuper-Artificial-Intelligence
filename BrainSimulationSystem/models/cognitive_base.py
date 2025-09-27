"""
认知过程基础模块

定义认知过程的基类和通用接口。
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from abc import ABC, abstractmethod

from BrainSimulationSystem.core.network import NeuralNetwork


class CognitiveProcess(ABC):
    """认知过程基类，定义所有认知过程的通用接口
    
    新增功能：
    1. 神经调质受体支持
    2. 调质敏感性参数
    """
    
    def __init__(self, network: NeuralNetwork, params: Dict[str, Any]):
        """
        初始化认知过程
        
        Args:
            network: 神经网络实例
            params: 参数字典，可包含：
                - mod_sensitivity: 调质敏感度配置
        """
        self.network = network
        self.params = params
        
        # 调质敏感性配置
        self.mod_sensitivity = params.get("mod_sensitivity", {
            "dopamine": 1.0,
            "serotonin": 1.0,
            "acetylcholine": 1.0
        })
        
    def apply_neuromodulation(self, mod_type: str, level: float) -> float:
        """
        应用神经调质调节
        
        Args:
            mod_type: 调质类型('dopamine','serotonin','acetylcholine')
            level: 当前调质水平(0-1)
            
        Returns:
            调节后的效果值
        """
        sensitivity = self.mod_sensitivity.get(mod_type.lower(), 0.0)
        return level * sensitivity
>>>>>>> 在文件末尾添加新类
=======
class CognitiveProcess(ABC):
    """认知过程基类，定义所有认知过程的通用接口
    
    新增功能：
    1. 神经调质受体支持
    2. 调质敏感性参数
    """
    
    def __init__(self, network: NeuralNetwork, params: Dict[str, Any]):
        """
        初始化认知过程
        
        Args:
            network: 神经网络实例
            params: 参数字典，可包含：
                - mod_sensitivity: 调质敏感度配置
        """
        self.network = network
        self.params = params
        
        # 调质敏感性配置
        self.mod_sensitivity = params.get("mod_sensitivity", {
            "dopamine": 1.0,
            "serotonin": 1.0,
            "acetylcholine": 1.0
        })
        
    def apply_neuromodulation(self, mod_type: str, level: float) -> float:
        """
        应用神经调质调节
        
        Args:
            mod_type: 调质类型('dopamine','serotonin','acetylcholine')
            level: 当前调质水平(0-1)
            
        Returns:
            调节后的效果值
        """
        sensitivity = self.mod_sensitivity.get(mod_type.lower(), 0.0)
        return level * sensitivity
    
    @abstractmethod
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理输入并产生输出
        
        Args:
            inputs: 输入数据字典
            
        Returns:
            输出数据字典
        """
        pass