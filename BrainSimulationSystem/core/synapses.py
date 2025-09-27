"""
突触连接模型实现模块

包含各种类型的突触模型实现，如静态突触、动态突触、STDP可塑性突触等。
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from abc import ABC, abstractmethod


class Synapse(ABC):
    """突触基类，定义所有突触模型的通用接口"""
    
class STPMechanism:
    """短时程可塑性机制基类
    
    实现两种主要STP类型：
    1. 易化(Facilitation): 高频刺激增强突触响应
    2. 抑制(Depression): 高频刺激减弱突触响应
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        初始化STP机制
        
        Args:
            params: 参数字典，包含：
                - U: 初始释放概率
                - tau_f: 易化时间常数(ms)
                - tau_d: 抑制时间常数(ms)
        """
        self.U = params.get("U", 0.1)  # 初始释放概率
        self.tau_f = params.get("tau_f", 1000.0)  # 易化时间常数
        self.tau_d = params.get("tau_d", 200.0)  # 抑制时间常数
        self.u = self.U  # 当前释放概率
        self.x = 1.0  # 可用资源比例
        self.last_spike_time = -float('inf')
        
    def update(self, dt: float, spike: bool) -> float:
        """
        更新STP状态
        
        Args:
            dt: 时间步长(ms)
            spike: 是否有突触前脉冲
            
        Returns:
            当前突触效能(0-1)
        """
        # 资源恢复
        self.x += dt * ((1.0 - self.x) / self.tau_d)
        
        # 释放概率恢复
        self.u += dt * ((self.U - self.u) / self.tau_f)
        
        if spike:
            # 计算突触效能
            efficacy = self.u * self.x
            
            # 更新状态
            self.x -= self.u * self.x  # 资源消耗
            self.u += self.U * (1 - self.u)  # 释放概率增加
            
            return efficacy
        return 0.0
>>>>>>> 在文件末尾添加新类
=======
class Synapse(ABC):
    """突触基类，定义所有突触模型的通用接口"""
    
class STPMechanism:
    """短时程可塑性机制基类
    
    实现两种主要STP类型：
    1. 易化(Facilitation): 高频刺激增强突触响应
    2. 抑制(Depression): 高频刺激减弱突触响应
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        初始化STP机制
        
        Args:
            params: 参数字典，包含：
                - U: 初始释放概率
                - tau_f: 易化时间常数(ms)
                - tau_d: 抑制时间常数(ms)
        """
        self.U = params.get("U", 0.1)  # 初始释放概率
        self.tau_f = params.get("tau_f", 1000.0)  # 易化时间常数
        self.tau_d = params.get("tau_d", 200.0)  # 抑制时间常数
        self.u = self.U  # 当前释放概率
        self.x = 1.0  # 可用资源比例
        self.last_spike_time = -float('inf')
        
    def update(self, dt: float, spike: bool) -> float:
        """
        更新STP状态
        
        Args:
            dt: 时间步长(ms)
            spike: 是否有突触前脉冲
            
        Returns:
            当前突触效能(0-1)
        """
        # 资源恢复
        self.x += dt * ((1.0 - self.x) / self.tau_d)
        
        # 释放概率恢复
        self.u += dt * ((self.U - self.u) / self.tau_f)
        
        if spike:
            # 计算突触效能
            efficacy = self.u * self.x
            
            # 更新状态
            self.x -= self.u * self.x  # 资源消耗
            self.u += self.U * (1 - self.u)  # 释放概率增加
            
            return efficacy
        return 0.0

class STPSynapse(Synapse):
    """具有短时程可塑性的突触"""
    
    def __init__(self, pre_id: int, post_id: int, params: Dict[str, Any]):
        super().__init__(pre_id, post_id, params)
        self.stp = STPMechanism(params.get("stp_params", {}))
        self.base_weight = params.get("weight", 1.0)
        
    def transmit(self, pre_spike: bool, dt: float) -> float:
        """传递信号并更新STP状态"""
        efficacy = self.stp.update(dt, pre_spike)
        return self.base_weight * efficacy if pre_spike else 0.0
    
    @property
    def weight(self) -> float:
        """获取当前有效权重"""
        return self.base_weight * self.stp.u * self.stp.x
    
    def __init__(self, 
                 pre_neuron_id: int, 
                 post_neuron_id: int, 
                 params: Dict[str, Any]):
        """
        初始化突触
        
        Args:
            pre_neuron_id: 前神经元ID
            post_neuron_id: 后神经元ID
            params: 突触参数字典
        """
        self.pre_id = pre_neuron_id
        self.post_id = post_neuron_id
        self.params = params
        self.reset()
    
    @abstractmethod
    def reset(self) -> None:
        """重置突触状态"""
        pass
    
    @abstractmethod
    def transmit(self, pre_spike: bool, dt: float) -> float:
        """
        传递突触信号
        
        Args:
            pre_spike: 前神经元是否产生脉冲
            dt: 时间步长
            
        Returns:
            传递给后神经元的电流
        """
        pass
    
    @property
    @abstractmethod
    def weight(self) -> float:
        """获取当前突触权重"""
        pass


class StaticSynapse(Synapse):
    """
    静态突触模型
    
    具有固定权重和延迟的简单突触模型
    """
    
    def __init__(self, 
                 pre_neuron_id: int, 
                 post_neuron_id: int, 
                 params: Dict[str, Any]):
        """
        初始化静态突触
        
        Args:
            pre_neuron_id: 前神经元ID
            post_neuron_id: 后神经元ID
            params: 突触参数字典，包含以下键：
                - weight: 突触权重
                - delay: 突触延迟 (ms)
        """
        super().__init__(pre_neuron_id, post_neuron_id, params)
    
    def reset(self) -> None:
        """重置突触状态"""
        self._weight = self.params.get("weight", 1.0)
        self._delay = self.params.get("delay", 1.0)  # ms
        self._delay_queue = []  # 延迟队列
    
    def transmit(self, pre_spike: bool, dt: float) -> float:
        """
        传递突触信号
        
        Args:
            pre_spike: 前神经元是否产生脉冲
            dt: 时间步长
            
        Returns:
            传递给后神经元的电流
        """
        # 将新的脉冲加入延迟队列
        self._delay_queue.append(pre_spike)
        
        # 计算延迟步数
        delay_steps = max(1, int(self._delay / dt))
        
        # 如果队列长度超过延迟步数，取出最早的脉冲
        if len(self._delay_queue) > delay_steps:
            delayed_spike = self._delay_queue.pop(0)
            if delayed_spike:
                return self._weight
        
        return 0.0
    
    @property
    def weight(self) -> float:
        """获取当前突触权重"""
        return self._weight


class DynamicSynapse(Synapse):
    """
    动态突触模型
    
    具有短期可塑性特性的突触模型，包括易化和抑制
    """
    
    def __init__(self, 
                 pre_neuron_id: int, 
                 post_neuron_id: int, 
                 params: Dict[str, Any]):
        """
        初始化动态突触
        
        Args:
            pre_neuron_id: 前神经元ID
            post_neuron_id: 后神经元ID
            params: 突触参数字典，包含以下键：
                - weight: 基础突触权重
                - delay: 突触延迟 (ms)
                - u: 资源利用率
                - tau_d: 抑制时间常数 (ms)
                - tau_f: 易化时间常数 (ms)
        """
        super().__init__(pre_neuron_id, post_neuron_id, params)
    
    def reset(self) -> None:
        """重置突触状态"""
        self._base_weight = self.params.get("weight", 1.0)
        self._delay = self.params.get("delay", 1.0)  # ms
        self._delay_queue = []  # 延迟队列
        
        # 短期可塑性参数
        self._u = self.params.get("u", 0.1)  # 资源利用率
        self._x = 1.0  # 可用资源比例
        self._u_state = self._u  # 当前利用率状态
        
        # 时间常数
        self._tau_d = self.params.get("tau_d", 200.0)  # 抑制时间常数 (ms)
        self._tau_f = self.params.get("tau_f", 600.0)  # 易化时间常数 (ms)
        
        self._last_update_time = 0.0  # 上次更新时间
    
    def transmit(self, pre_spike: bool, dt: float) -> float:
        """
        传递突触信号
        
        Args:
            pre_spike: 前神经元是否产生脉冲
            dt: 时间步长
            
        Returns:
            传递给后神经元的电流
        """
        # 更新短期可塑性状态
        self._update_dynamics(dt)
        
        # 将新的脉冲加入延迟队列
        self._delay_queue.append(pre_spike)
        
        # 计算延迟步数
        delay_steps = max(1, int(self._delay / dt))
        
        # 如果队列长度超过延迟步数，取出最早的脉冲
        if len(self._delay_queue) > delay_steps:
            delayed_spike = self._delay_queue.pop(0)
            if delayed_spike:
                # 当脉冲到达时，计算突触传递
                psr = self._base_weight * self._u_state * self._x
                
                # 更新资源状态
                self._x -= self._u_state * self._x
                self._u_state += self.params.get("U", 0.2) * (1.0 - self._u_state)
                
                return psr
        
        return 0.0
    
    def _update_dynamics(self, dt: float) -> None:
        """
        更新短期可塑性动态
        
        Args:
            dt: 时间步长
        """
        # 资源恢复
        self._x += dt * ((1.0 - self._x) / self._tau_d)
        
        # 利用率恢复
        self._u_state += dt * ((self._u - self._u_state) / self._tau_f)
        
        # 更新时间
        self._last_update_time += dt
    
    @property
    def weight(self) -> float:
        """获取当前有效突触权重"""
        return self._base_weight * self._u_state * self._x


class STDPSynapse(Synapse):
    """
    增强型STDP可塑性突触模型
    
    特性：
    1. 支持双相STDP (LTP/LTD)
    2. 添加NMDA/AMPA受体动力学
    3. 支持权重依赖的STDP
    """
    
    class ReceptorType(Enum):
        AMPA = 1
        NMDA = 2
        
    @dataclass
    class Receptor:
        """突触后受体模型"""
        type: 'STDPSynapse.ReceptorType'
        g_max: float       # 最大电导
        E_rev: float       # 反转电位
        tau_rise: float    # 上升时间常数
        tau_decay: float   # 衰减时间常数
        g: float = 0       # 当前电导
        
        def update(self, dt: float):
            """更新受体状态"""
            self.g *= np.exp(-dt / self.tau_decay)
            
        def activate(self, weight: float):
            """激活受体"""
            self.g += weight * self.g_max
            
        def current(self, V_post: float) -> float:
            """计算受体电流"""
            if self.type == STDPSynapse.ReceptorType.NMDA:
                # NMDA受体的电压依赖性
                mg_block = 1.0 / (1.0 + 0.28 * np.exp(-0.062 * V_post))
                return self.g * mg_block * (V_post - self.E_rev)
            else:
                return self.g * (V_post - self.E_rev)
>>>>>>> 在文件末尾添加新类
=======
class STDPSynapse(Synapse):
    """
    增强型STDP可塑性突触模型
    
    特性：
    1. 支持双相STDP (LTP/LTD)
    2. 添加NMDA/AMPA受体动力学
    3. 支持权重依赖的STDP
    """
    
    class ReceptorType(Enum):
        AMPA = 1
        NMDA = 2
        
    @dataclass
    class Receptor:
        """突触后受体模型"""
        type: 'STDPSynapse.ReceptorType'
        g_max: float       # 最大电导
        E_rev: float       # 反转电位
        tau_rise: float    # 上升时间常数
        tau_decay: float   # 衰减时间常数
        g: float = 0       # 当前电导
        
        def update(self, dt: float):
            """更新受体状态"""
            self.g *= np.exp(-dt / self.tau_decay)
            
        def activate(self, weight: float):
            """激活受体"""
            self.g += weight * self.g_max
            
        def current(self, V_post: float) -> float:
            """计算受体电流"""
            if self.type == STDPSynapse.ReceptorType.NMDA:
                # NMDA受体的电压依赖性
                mg_block = 1.0 / (1.0 + 0.28 * np.exp(-0.062 * V_post))
                return self.g * mg_block * (V_post - self.E_rev)
            else:
                return self.g * (V_post - self.E_rev)

    def __init__(self, 
                 pre_neuron_id: int, 
                 post_neuron_id: int, 
                 params: Dict[str, Any]):
        """
        初始化增强型STDP突触
        
        新增参数：
        - w0: 目标权重
        - lambda: 学习率缩放因子
        - receptor_ratio: NMDA/AMPA受体比例
        """
        super().__init__(pre_neuron_id, post_neuron_id, params)
        
        # 受体配置
        self.receptors = [
            self.Receptor(self.ReceptorType.AMPA, 1.0, 0.0, 0.2, 2.0),
            self.Receptor(self.ReceptorType.NMDA, 
                        params.get("receptor_ratio", 0.3), 
                        0.0, 10.0, 100.0)
        ]
        
        # STDP增强参数
        self.w0 = params.get("w0", 0.5)  # 目标权重
        self.lambda_ = params.get("lambda", 1.0)  # 学习率缩放
        
    def transmit(self, pre_spike: bool, dt: float) -> float:
        """
        增强的突触信号传递，包含受体动力学
        """
        # 更新受体状态
        for receptor in self.receptors:
            receptor.update(dt)
            if pre_spike:
                receptor.activate(self._weight)
        
        # 计算总电流
        total_current = 0
        for receptor in self.receptors:
            total_current += receptor.current(self._post.V)
            
        return total_current
    
    def _update_weight(self, dw: float) -> None:
        """
        增强的权重更新，包含权重依赖
        """
        # 权重依赖的STDP
        scaling = 1.0 - np.abs((self._weight - self.w0) / self.w0)
        dw *= self.lambda_ * scaling
        
        self._weight += dw
        self._weight = max(self._weight_min, min(self._weight, self._weight_max))
    
    def __init__(self, 
                 pre_neuron_id: int, 
                 post_neuron_id: int, 
                 params: Dict[str, Any]):
        """
        初始化STDP突触
        
        Args:
            pre_neuron_id: 前神经元ID
            post_neuron_id: 后神经元ID
            params: 突触参数字典，包含以下键：
                - weight: 初始突触权重
                - delay: 突触延迟 (ms)
                - learning_rate: 学习率
                - a_plus: 正时间窗口幅度
                - a_minus: 负时间窗口幅度
                - tau_plus: 正时间窗口时间常数 (ms)
                - tau_minus: 负时间窗口时间常数 (ms)
                - weight_min: 最小权重
                - weight_max: 最大权重
        """
        super().__init__(pre_neuron_id, post_neuron_id, params)
    
    def reset(self) -> None:
        """重置突触状态"""
        self._weight = self.params.get("weight", 1.0)
        self._delay = self.params.get("delay", 1.0)  # ms
        self._delay_queue = []  # 延迟队列
        
        # STDP参数
        self._learning_rate = self.params.get("learning_rate", 0.01)
        self._a_plus = self.params.get("a_plus", 0.1)
        self._a_minus = self.params.get("a_minus", -0.1)
        self._tau_plus = self.params.get("tau_plus", 20.0)  # ms
        self._tau_minus = self.params.get("tau_minus", 20.0)  # ms
        self._weight_min = self.params.get("weight_min", 0.0)
        self._weight_max = self.params.get("weight_max", 1.0)
        
        # 跟踪变量
        self._pre_trace = 0.0  # 前神经元痕迹
        self._post_trace = 0.0  # 后神经元痕迹
        self._last_pre_spike_time = -float('inf')  # 上次前神经元脉冲时间
        self._last_post_spike_time = -float('inf')  # 上次后神经元脉冲时间
        self._current_time = 0.0  # 当前时间
    
    def transmit(self, pre_spike: bool, dt: float) -> float:
        """
        传递突触信号
        
        Args:
            pre_spike: 前神经元是否产生脉冲
            dt: 时间步长
            
        Returns:
            传递给后神经元的电流
        """
        # 更新时间
        self._current_time += dt
        
        # 更新痕迹
        self._pre_trace *= np.exp(-dt / self._tau_plus)
        self._post_trace *= np.exp(-dt / self._tau_minus)
        
        # 将新的脉冲加入延迟队列
        self._delay_queue.append(pre_spike)
        
        # 计算延迟步数
        delay_steps = max(1, int(self._delay / dt))
        
        # 如果队列长度超过延迟步数，取出最早的脉冲
        if len(self._delay_queue) > delay_steps:
            delayed_spike = self._delay_queue.pop(0)
            if delayed_spike:
                # 记录前神经元脉冲时间
                self._last_pre_spike_time = self._current_time
                
                # 更新前神经元痕迹
                self._pre_trace += 1.0
                
                # 根据后神经元痕迹调整权重 (pre after post)
                dw = self._learning_rate * self._a_minus * self._post_trace
                self._update_weight(dw)
                
                return self._weight
        
        return 0.0
    
    def update_post_spike(self, post_spike: bool) -> None:
        """
        更新后神经元脉冲信息
        
        Args:
            post_spike: 后神经元是否产生脉冲
        """
        if post_spike:
            # 记录后神经元脉冲时间
            self._last_post_spike_time = self._current_time
            
            # 更新后神经元痕迹
            self._post_trace += 1.0
            
            # 根据前神经元痕迹调整权重 (post after pre)
            dw = self._learning_rate * self._a_plus * self._pre_trace
            self._update_weight(dw)
    
    def _update_weight(self, dw: float) -> None:
        """
        更新突触权重
        
        Args:
            dw: 权重变化量
        """
        self._weight += dw
        self._weight = max(self._weight_min, min(self._weight, self._weight_max))
    
    @property
    def weight(self) -> float:
        """获取当前突触权重"""
        return self._weight


class AxonalDelaySynapse(Synapse):
    """
    轴突传导延迟突触模型
    
    特性：
    1. 基于轴突长度和髓鞘化程度的精确延迟计算
    2. 支持可变传导速度
    3. 模拟轴突分支点的延迟累积
    4. 支持轴突传导可靠性
    """
    
    def __init__(self, 
                 pre_neuron_id: int, 
                 post_neuron_id: int, 
                 params: Dict[str, Any]):
        """
        初始化轴突传导延迟突触
        
        Args:
            pre_neuron_id: 前神经元ID
            post_neuron_id: 后神经元ID
            params: 突触参数字典，包含以下键：
                - weight: 突触权重
                - axon_length: 轴突长度 (μm)
                - myelination: 髓鞘化程度 (0-1)
                - branch_points: 轴突分支点数量
                - reliability: 传导可靠性 (0-1)
        """
        super().__init__(pre_neuron_id, post_neuron_id, params)
    
    def reset(self) -> None:
        """重置突触状态"""
        self._weight = self.params.get("weight", 1.0)
        
        # 轴突参数
        self._axon_length = self.params.get("axon_length", 1000.0)  # μm
        self._myelination = self.params.get("myelination", 0.7)  # 髓鞘化程度 (0-1)
        self._branch_points = self.params.get("branch_points", 0)  # 分支点数量
        self._reliability = self.params.get("reliability", 0.98)  # 传导可靠性
        
        # 计算传导速度和延迟
        self._conduction_velocity = self._calculate_conduction_velocity()  # μm/ms
        self._base_delay = self._calculate_delay()  # ms
        
        # 延迟队列 - 使用更精确的时间戳而不是简单的队列
        self._spike_times = []  # 存储脉冲时间戳
        self._current_time = 0.0  # 当前时间
    
    def _calculate_conduction_velocity(self) -> float:
        """
        计算轴突传导速度
        
        Returns:
            传导速度 (μm/ms)
        """
        # 基础传导速度 (无髓鞘): 0.5-2 μm/ms
        # 有髓鞘传导速度: 10-120 μm/ms
        base_velocity = 1.0  # μm/ms (无髓鞘)
        max_myelinated_velocity = 100.0  # μm/ms (完全髓鞘化)
        
        # 髓鞘化对速度的影响 (非线性)
        myelination_factor = self._myelination ** 2  # 平方关系使得髓鞘化效果更显著
        
        # 计算最终速度
        velocity = base_velocity + (max_myelinated_velocity - base_velocity) * myelination_factor
        
        # 轴突直径对速度的影响 (可选参数)
        axon_diameter = self.params.get("axon_diameter", 1.0)  # μm
        diameter_factor = np.sqrt(axon_diameter)  # 直径的平方根关系
        
        return velocity * diameter_factor
    
    def _calculate_delay(self) -> float:
        """
        计算轴突传导延迟
        
        Returns:
            传导延迟 (ms)
        """
        # 基础延迟 = 长度 / 速度
        base_delay = self._axon_length / self._conduction_velocity
        
        # 分支点引起的额外延迟 (每个分支点增加约0.1-0.5ms)
        branch_delay = self._branch_points * 0.3  # ms
        
        # 温度因素 (可选参数)
        temperature = self.params.get("temperature", 37.0)  # 摄氏度
        temp_factor = 2.0 ** ((temperature - 37.0) / 10.0)  # Q10 = 2
        
        # 总延迟
        total_delay = (base_delay + branch_delay) / temp_factor
        
        return max(0.1, total_delay)  # 最小延迟为0.1ms
    
    def transmit(self, pre_spike: bool, dt: float) -> float:
        """
        传递突触信号，考虑轴突传导延迟
        
        Args:
            pre_spike: 前神经元是否产生脉冲
            dt: 时间步长
            
        Returns:
            传递给后神经元的电流
        """
        # 更新当前时间
        self._current_time += dt
        
        # 如果有新脉冲，根据可靠性决定是否传导，并添加到队列
        if pre_spike and random.random() < self._reliability:
            # 计算脉冲到达时间 = 当前时间 + 延迟
            arrival_time = self._current_time + self._base_delay
            
            # 添加抖动 (0.9-1.1倍延迟)
            jitter = random.uniform(0.9, 1.1)
            arrival_time *= jitter
            
            # 将脉冲添加到队列
            self._spike_times.append(arrival_time)
        
        # 检查是否有脉冲到达
        current_output = 0.0
        remaining_spikes = []
        
        for spike_time in self._spike_times:
            if spike_time <= self._current_time:
                # 脉冲已到达
                current_output += self._weight
            else:
                # 脉冲尚未到达
                remaining_spikes.append(spike_time)
        
        # 更新队列
        self._spike_times = remaining_spikes
        
        return current_output
    
    @property
    def weight(self) -> float:
        """获取当前突触权重"""
        return self._weight
    
    @property
    def delay(self) -> float:
        """获取当前突触延迟"""
        return self._base_delay
    
    @property
    def conduction_velocity(self) -> float:
        """获取当前传导速度"""
        return self._conduction_velocity


def create_synapse(synapse_type: str, 
                   pre_neuron_id: int, 
                   post_neuron_id: int, 
                   params: Dict[str, Any]) -> Synapse:
    """
    创建指定类型的突触
    
    Args:
        synapse_type: 突触类型，可选值：'static', 'dynamic', 'stdp', 'gaba', 'axonal'
        pre_neuron_id: 前神经元ID
        post_neuron_id: 后神经元ID
        params: 突触参数字典
        
    Returns:
        创建的突触实例
        
    Raises:
        ValueError: 如果指定的突触类型不支持
    """
    synapse_classes = {
        'static': StaticSynapse,
        'dynamic': DynamicSynapse,
        'stdp': STDPSynapse,
        'gaba': GABAergicSynapse,
        'axonal': AxonalDelaySynapse,
    }
>>>>>>> 在文件末尾添加新类
=======
def create_synapse(synapse_type: str, 
                   pre_neuron_id: int, 
                   post_neuron_id: int, 
                   params: Dict[str, Any]) -> Synapse:
    """
    创建指定类型的突触
    
    Args:
        synapse_type: 突触类型，可选值：'static', 'dynamic', 'stdp', 'gaba'
        pre_neuron_id: 前神经元ID
        post_neuron_id: 后神经元ID
        params: 突触参数字典
        
    Returns:
        创建的突触实例
        
    Raises:
        ValueError: 如果指定的突触类型不支持
    """
    synapse_classes = {
        'static': StaticSynapse,
        'dynamic': DynamicSynapse,
        'stdp': STDPSynapse,
        'gaba': GABAergicSynapse,
    }

class GABAergicSynapse(Synapse):
    """GABA能抑制性突触
    
    特性：
    1. 实现GABA_A和GABA_B受体
    2. 精确的抑制性突触后电位(IPSP)
    3. 可配置的抑制强度和时间常数
    """
    
    def __init__(self, pre_id: int, post_id: int, params: Dict[str, Any]):
        super().__init__(pre_id, post_id, params)
        
        # 默认GABA受体配置
        self.receptors = [
            Receptor(ReceptorType.GABA_A, 1.0, -80.0, 0.5, 5.0),  # 快速抑制
            Receptor(ReceptorType.GABA_B, 0.5, -90.0, 10.0, 150.0)  # 慢速抑制 
        ]
        
        # 抑制强度调节
        self.inhibition_scale = params.get("inhibition_scale", 1.0)
        
    def transmit(self, pre_spike: bool, dt: float) -> float:
        """传递抑制性信号"""
        total_current = 0
        
        for receptor in self.receptors:
            receptor.update(dt)
            
            if pre_spike:
                receptor.activate(self._weight)
                
            # GABA电流为负值(抑制性)
            total_current += receptor.current(self._post.V) * self.inhibition_scale
            
        return total_current
    
    @property
    def weight(self) -> float:
        """获取当前抑制强度(取绝对值)"""
        return abs(self._weight)
    
    if synapse_type not in synapse_classes:
        raise ValueError(f"不支持的突触类型: {synapse_type}")
    
    return synapse_classes[synapse_type](pre_neuron_id, post_neuron_id, params)