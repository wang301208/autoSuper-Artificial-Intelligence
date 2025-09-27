"""
神经元模型实现模块

包含各种类型的神经元模型实现，如LIF（Leaky Integrate-and-Fire）、
Izhikevich模型等。
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from abc import ABC, abstractmethod


class Neuron(ABC):
    """神经元基类，定义所有神经元模型的通用接口"""
    
    def __init__(self, neuron_id: int, params: Dict[str, Any]):
        """
        初始化神经元
        
        Args:
            neuron_id: 神经元唯一标识符
            params: 神经元参数字典
        """
        self.id = neuron_id
        self.params = params
        self.reset()
        
    @abstractmethod
    def reset(self) -> None:
        """重置神经元状态"""
        pass
    
    @abstractmethod
    def update(self, input_current: float, dt: float) -> bool:
        """
        更新神经元状态
        
        Args:
            input_current: 输入电流
            dt: 时间步长
            
        Returns:
            是否产生脉冲
        """
        pass
    
    @property
    @abstractmethod
    def voltage(self) -> float:
        """获取当前膜电位"""
        pass


class LIFNeuron(Neuron):
    """
    Leaky Integrate-and-Fire神经元模型
    
    基于电容-电阻电路模型，具有漏电流特性
    """
    
    def __init__(self, neuron_id: int, params: Dict[str, Any]):
        """
        初始化LIF神经元
        
        Args:
            neuron_id: 神经元唯一标识符
            params: 神经元参数字典，包含以下键：
                - threshold: 激活阈值
                - reset_potential: 重置电位
                - resting_potential: 静息电位
                - time_constant: 时间常数 (ms)
                - refractory_period: 不应期 (ms)
        """
        super().__init__(neuron_id, params)
    
    def reset(self) -> None:
        """重置神经元状态"""
        self._voltage = self.params.get("resting_potential", 0.0)
        self._last_spike_time = -float('inf')  # 上次脉冲时间
        self._spike_history = []  # 脉冲历史记录
        
    def update(self, input_current: float, dt: float) -> bool:
        """
        更新神经元状态
        
        Args:
            input_current: 输入电流
            dt: 时间步长 (ms)
            
        Returns:
            是否产生脉冲
        """
        # 获取参数
        threshold = self.params.get("threshold", 1.0)
        reset_potential = self.params.get("reset_potential", 0.0)
        resting_potential = self.params.get("resting_potential", 0.0)
        tau = self.params.get("time_constant", 10.0)  # 膜时间常数
        refractory_period = self.params.get("refractory_period", 2.0)  # 不应期
        
        # 当前时间
        current_time = len(self._spike_history) * dt if self._spike_history else 0.0
        
        # 检查是否处于不应期
        if current_time - self._last_spike_time < refractory_period:
            self._voltage = reset_potential
            self._spike_history.append(0)
            return False
        
        # 更新膜电位 (Leaky Integrate-and-Fire方程)
        dv = (-(self._voltage - resting_potential) + input_current) * (dt / tau)
        self._voltage += dv
        
        # 检查是否产生脉冲
        if self._voltage >= threshold:
            self._voltage = reset_potential
            self._last_spike_time = current_time
            self._spike_history.append(1)
            return True
        
        self._spike_history.append(0)
        return False
    
    @property
    def voltage(self) -> float:
        """获取当前膜电位"""
        return self._voltage
    
    @property
    def spike_history(self) -> List[int]:
        """获取脉冲历史记录"""
        return self._spike_history


class IzhikevichNeuron(Neuron):
    """
    Izhikevich神经元模型
    
    能够模拟多种生物神经元的行为模式，如规则发放、爆发发放等
    """
    
    def __init__(self, neuron_id: int, params: Dict[str, Any]):
        """
        初始化Izhikevich神经元
        
        Args:
            neuron_id: 神经元唯一标识符
            params: 神经元参数字典，包含以下键：
                - a: 恢复变量u的时间尺度
                - b: 恢复变量u对膜电位v的敏感度
                - c: 脉冲后的膜电位重置值
                - d: 脉冲后的恢复变量u的重置增量
                - threshold: 脉冲阈值
        """
        super().__init__(neuron_id, params)
    
    def reset(self) -> None:
        """重置神经元状态"""
        self._v = self.params.get("c", -65.0)  # 膜电位
        self._u = self.params.get("b", 0.2) * self._v  # 恢复变量
        self._spike_history = []  # 脉冲历史记录
    
    def update(self, input_current: float, dt: float) -> bool:
        """
        更新神经元状态
        
        Args:
            input_current: 输入电流
            dt: 时间步长 (ms)
            
        Returns:
            是否产生脉冲
        """
        # 获取参数
        a = self.params.get("a", 0.02)
        b = self.params.get("b", 0.2)
        c = self.params.get("c", -65.0)
        d = self.params.get("d", 8.0)
        threshold = self.params.get("threshold", 30.0)
        
        # 更新膜电位和恢复变量 (Izhikevich方程)
        dv = (0.04 * self._v**2 + 5 * self._v + 140 - self._u + input_current) * dt
        du = (a * (b * self._v - self._u)) * dt
        
        self._v += dv
        self._u += du
        
        # 检查是否产生脉冲
        if self._v >= threshold:
            self._v = c
            self._u += d
            self._spike_history.append(1)
            return True
        
        self._spike_history.append(0)
        return False
    
    @property
    def voltage(self) -> float:
        """获取当前膜电位"""
        return self._v
    
    @property
    def spike_history(self) -> List[int]:
        """获取脉冲历史记录"""
        return self._spike_history


class AdExNeuron(Neuron):
    """
    Adaptive Exponential Integrate-and-Fire神经元模型
    
    具有自适应特性和指数项，能更好地拟合生物神经元的行为
    """
    
    def __init__(self, neuron_id: int, params: Dict[str, Any]):
        """
        初始化AdEx神经元
        
        Args:
            neuron_id: 神经元唯一标识符
            params: 神经元参数字典
        """
        super().__init__(neuron_id, params)
    
    def reset(self) -> None:
        """重置神经元状态"""
        self._v = self.params.get("resting_potential", -70.0)  # 膜电位
        self._w = 0.0  # 适应变量
        self._last_spike_time = -float('inf')  # 上次脉冲时间
        self._spike_history = []  # 脉冲历史记录
    
    def update(self, input_current: float, dt: float) -> bool:
        """
        更新神经元状态
        
        Args:
            input_current: 输入电流
            dt: 时间步长 (ms)
            
        Returns:
            是否产生脉冲
        """
        # 获取参数
        C = self.params.get("capacitance", 281.0)  # 膜电容
        g_L = self.params.get("leak_conductance", 30.0)  # 漏电导
        E_L = self.params.get("resting_potential", -70.6)  # 静息电位
        V_T = self.params.get("threshold_potential", -50.4)  # 阈值电位
        Delta_T = self.params.get("slope_factor", 2.0)  # 斜率因子
        a = self.params.get("adaptation_coupling", 4.0)  # 适应耦合
        tau_w = self.params.get("adaptation_time_constant", 144.0)  # 适应时间常数
        b = self.params.get("spike_triggered_adaptation", 80.5)  # 脉冲触发适应
        V_reset = self.params.get("reset_potential", -70.6)  # 重置电位
        threshold = self.params.get("threshold", 0.0)  # 脉冲阈值
        refractory_period = self.params.get("refractory_period", 2.0)  # 不应期
        
        # 当前时间
        current_time = len(self._spike_history) * dt if self._spike_history else 0.0
        
        # 检查是否处于不应期
        if current_time - self._last_spike_time < refractory_period:
            self._v = V_reset
            self._spike_history.append(0)
            return False
        
        # 计算指数项
        if self._v < threshold:
            exp_term = Delta_T * np.exp((self._v - V_T) / Delta_T)
        else:
            exp_term = 0.0
        
        # 更新膜电位和适应变量 (AdEx方程)
        dv = (-(self._v - E_L) + exp_term - self._w / C + input_current / C) * dt
        dw = (a * (self._v - E_L) - self._w) * dt / tau_w
        
        self._v += dv
        self._w += dw
        
        # 检查是否产生脉冲
        if self._v >= threshold:
            self._v = V_reset
            self._w += b
            self._last_spike_time = current_time
            self._spike_history.append(1)
            return True
        
        self._spike_history.append(0)
        return False
    
    @property
    def voltage(self) -> float:
        """获取当前膜电位"""
        return self._v
    
    @property
    def spike_history(self) -> List[int]:
        """获取脉冲历史记录"""
        return self._spike_history


def create_neuron(neuron_type: str, neuron_id: int, params: Dict[str, Any]) -> Neuron:
    """
    创建指定类型的神经元
    
    Args:
        neuron_type: 神经元类型，可选值：
            - 'lif': Leaky Integrate-and-Fire
            - 'izhikevich': Izhikevich模型
            - 'adex': Adaptive Exponential IF
            - 'hh': Hodgkin-Huxley模型
        neuron_id: 神经元唯一标识符
        params: 神经元参数字典，不同模型需要不同参数：
            HH模型必须包含：
            - C_m: 膜电容(μF/cm^2)
            - g_Na: 钠最大电导(mS/cm^2)
            - g_K: 钾最大电导(mS/cm^2)
            - g_L: 漏电导(mS/cm^2)
            - E_Na: 钠平衡电位(mV)
            - E_K: 钾平衡电位(mV)
            - E_L: 漏平衡电位(mV)
        
    Returns:
        创建的神经元实例
        
    Raises:
        ValueError: 如果指定的神经元类型不支持或参数缺失
    """
    neuron_classes = {
        'lif': LIFNeuron,
        'izhikevich': IzhikevichNeuron,
        'adex': AdExNeuron,
        'hh': HodgkinHuxleyNeuron,
    }
    
    if neuron_type not in neuron_classes:
        raise ValueError(f"不支持的神经元类型: {neuron_type}")
    
    # HH模型参数验证
    if neuron_type == 'hh':
        required = ['C_m', 'g_Na', 'g_K', 'g_L', 'E_Na', 'E_K', 'E_L']
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"HH模型缺少必要参数: {missing}")
    
    return neuron_classes[neuron_type](neuron_id, params)


class PositionalNeuron:
    """带位置信息的神经元包装类"""
    
    def __init__(self, neuron: Neuron, position: Optional[np.ndarray] = None):
        """
        初始化带位置信息的神经元
        
        Args:
            neuron: 基础神经元实例
            position: 3D位置坐标(μm)
        """
        self.neuron = neuron
        self.position = position if position is not None else np.zeros(3)
        
    def __getattr__(self, name):
        """转发所有未定义属性到基础神经元"""
        return getattr(self.neuron, name)


class HodgkinHuxleyNeuron(Neuron):
    """Hodgkin-Huxley神经元模型"""
    
    def __init__(self, neuron_id: int, params: Dict[str, Any]):
        """
        初始化HH神经元
        
        Args:
            neuron_id: 神经元唯一标识符
            params: 神经元参数字典，包含：
                - C_m: 膜电容(μF/cm^2)
                - g_Na: 钠最大电导(mS/cm^2)
                - g_K: 钾最大电导(mS/cm^2) 
                - g_L: 漏电导(mS/cm^2)
                - E_Na: 钠平衡电位(mV)
                - E_K: 钾平衡电位(mV)
                - E_L: 漏平衡电位(mV)
                - dendrite_params: 树突参数字典 (可选)
        """
        super().__init__(neuron_id, params)
        
        # 初始化树突参数
        self.dendrite_params = params.get("dendrite_params", {
            'apical': {
                'length': 100.0,  # μm
                'taper': 0.5,     # 锥度系数
                'R_m': 30000.0,   # Ω·cm²
                'C_m': 1.0,       # μF/cm²
                'segments': 10    # 分段数
            },
            'basal': {
                'length': 50.0,
                'taper': 0.7,
                'R_m': 40000.0,
                'C_m': 1.0,
                'segments': 8
            }
        })
        
        # 初始化树突段
        self.dendritic_segments = {
            'apical': self._init_dendrite_segments('apical'),
            'basal': self._init_dendrite_segments('basal')
        }
        
        self.reset()
        
    def _init_dendrite_segments(self, dendrite_type: str) -> List[Dict]:
        """初始化树突段"""
        params = self.dendrite_params[dendrite_type]
        segments = []
        
        for i in range(params['segments']):
            segments.append({
                'length': params['length'] / params['segments'],
                'diameter': 2.0 * (1 - i * params['taper'] / params['segments']),  # μm
                'V_m': self.params.get("resting_potential", -65.0),
                'I_syn': 0.0,
                'Ca2_plus': 0.0,
                'active_channels': {
                    'Na': 0.0,
                    'K': 0.0,
                    'Ca': 0.0
                }
            })
        
        return segments
    
    def reset(self) -> None:
        """重置神经元状态"""
        self.V = self.params.get("resting_potential", -65.0)
        self.m = 0.05  # 钠激活门控
        self.h = 0.6   # 钠失活门控 
        self.n = 0.32   # 钾激活门控
        self._spike_history = []
        
        # 重置树突状态
        for dendrite_type in self.dendritic_segments:
            for segment in self.dendritic_segments[dendrite_type]:
                segment['V_m'] = self.V
                segment['I_syn'] = 0.0
                segment['Ca2_plus'] = 0.0
                segment['active_channels'] = {'Na': 0.0, 'K': 0.0, 'Ca': 0.0}
        
        # 树突整合参数
        self.dendritic_integration_window = self.params.get('dendritic_window', 5.0)  # ms
        self.dendritic_spike_threshold = self.params.get('dendritic_threshold', -45.0)  # mV
        self.dendritic_spikes = []
        self.dendritic_spike_times = []
    
    def update(self, I_ext: float, dt: float) -> bool:
        """
        更新神经元状态
        
        Args:
            I_ext: 输入电流
            dt: 时间步长(ms)
            
        Returns:
            是否产生脉冲
        """
        # 获取参数
        C_m = self.params.get("C_m", 1.0)
        g_Na = self.params.get("g_Na", 120.0)
        g_K = self.params.get("g_K", 36.0)
        g_L = self.params.get("g_L", 0.3)
        E_Na = self.params.get("E_Na", 50.0)
        E_K = self.params.get("E_K", -77.0)
        E_L = self.params.get("E_L", -54.387)
        
        # 计算门控变量变化率
        alpha_m = 0.1*(self.V+40)/(1-np.exp(-(self.V+40)/10))
        beta_m = 4.0*np.exp(-(self.V+65)/18)
        
        alpha_h = 0.07*np.exp(-(self.V+65)/20)
        beta_h = 1/(1+np.exp(-(self.V+35)/10))
        
        alpha_n = 0.01*(self.V+55)/(1-np.exp(-(self.V+55)/10))
        beta_n = 0.125*np.exp(-(self.V+65)/80)
        
        # 更新门控变量
        self.m += dt * (alpha_m*(1-self.m) - beta_m*self.m)
        self.h += dt * (alpha_h*(1-self.h) - beta_h*self.h) 
        self.n += dt * (alpha_n*(1-self.n) - beta_n*self.n)
        
        # 计算离子电流
        I_Na = g_Na * self.m**3 * self.h * (self.V - E_Na)
        I_K = g_K * self.n**4 * (self.V - E_K)
        I_L = g_L * (self.V - E_L)
        
        # 更新膜电位
        dVdt = (I_ext - I_Na - I_K - I_L) / C_m
        self.V += dVdt * dt
        
        # 检测动作电位
        spike = self.V >= 0
        self._spike_history.append(1 if spike else 0)
        
        if spike:
            self.V = self.params.get("reset_potential", -65.0)
            
        return spike
    
    @property
    def voltage(self) -> float:
        """获取当前膜电位"""
        return self.V
    
    @property 
    def spike_history(self) -> List[int]:
        """获取脉冲历史记录"""
        return self._spike_history