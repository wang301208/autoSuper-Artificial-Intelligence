"""
认知控制器模块

实现高级认知功能的协调和控制，整合注意力、工作记忆和其他认知组件。
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from enum import Enum

from BrainSimulationSystem.core.network import NeuralNetwork
from BrainSimulationSystem.models.cognitive_base import CognitiveProcess
from BrainSimulationSystem.models.attention import AttentionSystem
from BrainSimulationSystem.models.working_memory import AChModulatedWorkingMemory


class CognitiveState(Enum):
    """认知状态枚举"""
    IDLE = 0
    PERCEIVING = 1
    ATTENDING = 2
    PROCESSING = 3
    DECIDING = 4
    RESPONDING = 5


class CognitiveController(CognitiveProcess):
    """
    认知控制器
    
    协调和控制多个认知组件的交互，实现高级认知功能。
    特性：
    1. 认知组件集成
    2. 认知状态管理
    3. 神经调质调节
    4. 认知控制信号
    """
    
    def __init__(self, network: Optional[NeuralNetwork] = None, params: Dict[str, Any] = None):
        super().__init__(network, params or {})
        
        # 认知组件
        self.components = {}
        
        # 认知状态
        self.state = CognitiveState.IDLE
        self.state_history = []
        
        # 神经调质水平
        self.neuromodulators = {
            "dopamine": 0.5,
            "serotonin": 0.5,
            "acetylcholine": 0.5,
            "norepinephrine": 0.5
        }
        
        # 控制参数
        self.control_signals = {}
        self.task_goal = None
        
        # 创建核心认知组件
        self._create_core_components()
        
    def _create_core_components(self) -> None:
        """创建核心认知组件"""
        # 创建注意力系统
        attention_params = self.params.get("attention", {})
        self.components["attention"] = AttentionSystem(self.network, attention_params)
        
        # 创建工作记忆系统
        memory_params = self.params.get("working_memory", {})
        self.components["working_memory"] = AChModulatedWorkingMemory(self.network, memory_params)
        
        # 创建注意力-工作记忆接口
        from BrainSimulationSystem.models.attention import AttentionWorkingMemoryInterface
        self.components["attention_memory_interface"] = AttentionWorkingMemoryInterface(
            self.components["attention"],
            self.components["working_memory"]
        )
    
    def add_component(self, name: str, component: CognitiveProcess) -> None:
        """添加认知组件"""
        self.components[name] = component
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理认知控制
        
        Args:
            inputs: 输入数据字典，包含：
                - sensory_input: 感觉输入
                - task_goal: 任务目标
                - control_signals: 控制信号
                - neuromodulators: 神经调质水平
                
        Returns:
            认知处理结果
        """
        # 更新任务目标
        if "task_goal" in inputs:
            self.task_goal = inputs["task_goal"]
            
        # 更新控制信号
        if "control_signals" in inputs:
            self.control_signals.update(inputs["control_signals"])
            
        # 更新神经调质水平
        if "neuromodulators" in inputs:
            self.neuromodulators.update(inputs["neuromodulators"])
            
        # 确定认知状态
        self._update_cognitive_state(inputs)
        
        # 准备组件输入
        component_inputs = self._prepare_component_inputs(inputs)
        
        # 处理认知周期
        cycle_results = self._process_cognitive_cycle(component_inputs)
        
        # 整合结果
        result = {
            "cognitive_state": self.state.name,
            "state_history": [s.name for s in self.state_history[-5:]],
            "neuromodulators": self.neuromodulators.copy(),
            "components": cycle_results,
            "integrated_output": self._integrate_outputs(cycle_results)
        }
        
        return result
    
    def _update_cognitive_state(self, inputs: Dict[str, Any]) -> None:
        """更新认知状态"""
        # 保存历史状态
        self.state_history.append(self.state)
        
        # 简单状态转换逻辑
        if "sensory_input" in inputs and inputs["sensory_input"]:
            if self.state == CognitiveState.IDLE:
                self.state = CognitiveState.PERCEIVING
            elif self.state == CognitiveState.PERCEIVING:
                self.state = CognitiveState.ATTENDING
                
        if self.state == CognitiveState.ATTENDING:
            self.state = CognitiveState.PROCESSING
            
        if "decision_required" in inputs and inputs["decision_required"]:
            self.state = CognitiveState.DECIDING
            
        if "response_required" in inputs and inputs["response_required"]:
            self.state = CognitiveState.RESPONDING
            
        # 完成响应后回到空闲状态
        if self.state == CognitiveState.RESPONDING and "response_complete" in inputs:
            self.state = CognitiveState.IDLE
            
        # 限制历史长度
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
    
    def _prepare_component_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """准备各组件的输入"""
        component_inputs = {}
        
        # 基础输入，所有组件共享
        base_input = {
            "task_goal": self.task_goal,
            "cognitive_state": self.state,
            "neuromod.dopamine": self.neuromodulators["dopamine"],
            "neuromod.serotonin": self.neuromodulators["serotonin"],
            "neuromod.acetylcholine": self.neuromodulators["acetylcholine"],
            "neuromod.norepinephrine": self.neuromodulators["norepinephrine"]
        }
        
        # 添加感觉输入
        if "sensory_input" in inputs:
            base_input["sensory_input"] = inputs["sensory_input"]
            
        # 为每个组件准备输入
        for name in self.components:
            component_inputs[name] = base_input.copy()
            
            # 添加组件特定的控制信号
            if name in self.control_signals:
                component_inputs[name]["control"] = self.control_signals[name]
                
        return component_inputs
    
    def _process_cognitive_cycle(self, component_inputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """处理认知周期，按顺序激活组件"""
        results = {}
        
        # 处理顺序取决于当前认知状态
        if self.state in [CognitiveState.PERCEIVING, CognitiveState.ATTENDING]:
            # 感知和注意阶段：先处理注意力，再处理工作记忆
            processing_order = ["attention", "working_memory", "attention_memory_interface"]
        else:
            # 其他阶段：使用注意力-工作记忆接口
            processing_order = ["attention_memory_interface"]
            
            # 添加其他组件
            for name in self.components:
                if name not in processing_order:
                    processing_order.append(name)
        
        # 按顺序处理组件
        for name in processing_order:
            if name in self.components:
                # 获取组件
                component = self.components[name]
                
                # 准备输入，包括其他组件的结果
                component_input = component_inputs[name].copy()
                for result_name, result in results.items():
                    component_input[f"{result_name}_output"] = result
                
                # 处理组件
                try:
                    result = component.process(component_input)
                    results[name] = result
                except Exception as e:
                    results[name] = {"error": str(e)}
        
        return results
    
    def _integrate_outputs(self, component_results: Dict[str, Any]) -> Dict[str, Any]:
        """整合各组件的输出"""
        # 优先使用接口的集成输出
        if "attention_memory_interface" in component_results:
            interface_result = component_results["attention_memory_interface"]
            if "integrated_state" in interface_result:
                return interface_result["integrated_state"]
        
        # 否则手动整合
        integrated = {}
        
        # 添加注意力焦点
        if "attention" in component_results:
            attention_result = component_results["attention"]
            if "attention_focus" in attention_result:
                integrated["focus"] = attention_result["attention_focus"]
        
        # 添加工作记忆内容
        if "working_memory" in component_results:
            memory_result = component_results["working_memory"]
            if "memory_state" in memory_result:
                integrated["memory_content"] = memory_result["memory_state"]
        
        # 添加其他组件的关键输出
        for name, result in component_results.items():
            if name not in ["attention", "working_memory", "attention_memory_interface"]:
                integrated[f"{name}_output"] = result
        
        return integrated
    
    def set_neuromodulator(self, name: str, level: float) -> None:
        """设置神经调质水平"""
        if name in self.neuromodulators:
            self.neuromodulators[name] = max(0.0, min(1.0, level))
    
    def set_task_goal(self, goal: Any) -> None:
        """设置任务目标"""
        self.task_goal = goal
    
    def set_control_signal(self, component: str, signal_name: str, value: Any) -> None:
        """设置控制信号"""
        if component not in self.control_signals:
            self.control_signals[component] = {}
            
        self.control_signals[component][signal_name] = value


class CognitiveControllerBuilder:
    """
    认知控制器构建器
    
    用于配置和构建认知控制器及其组件。
    """
    
    def __init__(self, network: Optional[NeuralNetwork] = None):
        self.network = network
        self.params = {
            "attention": {},
            "working_memory": {},
            "decision": {},
            "learning": {}
        }
        self.components = {}
        
    def with_attention_params(self, params: Dict[str, Any]) -> 'CognitiveControllerBuilder':
        """配置注意力系统参数"""
        self.params["attention"].update(params)
        return self
        
    def with_working_memory_params(self, params: Dict[str, Any]) -> 'CognitiveControllerBuilder':
        """配置工作记忆系统参数"""
        self.params["working_memory"].update(params)
        return self
        
    def with_component(self, name: str, component: CognitiveProcess) -> 'CognitiveControllerBuilder':
        """添加自定义组件"""
        self.components[name] = component
        return self
        
    def build(self) -> CognitiveController:
        """构建认知控制器"""
        # 创建控制器
        controller = CognitiveController(self.network, self.params)
        
        # 添加自定义组件
        for name, component in self.components.items():
            controller.add_component(name, component)
            
        return controller