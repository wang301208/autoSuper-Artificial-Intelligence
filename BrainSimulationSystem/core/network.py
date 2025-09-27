"""
神经网络模型实现模块

定义神经网络的结构、连接模式和信息传递机制。
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np
from collections import defaultdict
import random

from BrainSimulationSystem.core.neurons import Neuron, create_neuron
from BrainSimulationSystem.core.synapses import Synapse, create_synapse, STDPSynapse


class Layer:
    """神经网络层，包含一组神经元"""
    
    def __init__(self, name: str, size: int, neuron_type: str, params: Dict[str, Any]):
        """
        初始化神经网络层
        
        Args:
            name: 层名称
            size: 神经元数量
            neuron_type: 神经元类型
            params: 神经元参数字典
        """
        self.name = name
        self.size = size
        self.neuron_type = neuron_type
        self.params = params
        self.neurons: List[Neuron] = []
        self.neuron_ids: List[int] = []
    
    def create_neurons(self, start_id: int) -> int:
        """
        创建层中的神经元
        
        Args:
            start_id: 起始神经元ID
            
        Returns:
            下一个可用的神经元ID
        """
        next_id = start_id
        self.neurons = []
        self.neuron_ids = []
        
        for i in range(self.size):
            neuron_id = next_id
            neuron = create_neuron(self.neuron_type, neuron_id, self.params)
            self.neurons.append(neuron)
            self.neuron_ids.append(neuron_id)
            next_id += 1
        
        return next_id
    
    def reset(self) -> None:
        """重置层中所有神经元的状态"""
        for neuron in self.neurons:
            neuron.reset()


class NeuralNetwork:
    """神经网络模型，包含多个神经元层和它们之间的连接"""
    
    def __init__(self, config: Dict[str, Any]):
>>>>>>> 在文件末尾添加新类
=======
class NeuralNetwork:
    """神经网络模型，包含多个神经元层和它们之间的连接"""
    
    def __init__(self, config: Dict[str, Any]):
        
class CorticalColumn(NeuralNetwork):
    """生物学合理的皮层柱网络结构
    
    实现6层皮层柱结构，包含：
    - L1: 分子层 (输入整合)
    - L2/3: 表层锥体细胞 (局部处理)
    - L4: 颗粒细胞 (主要输入层) 
    - L5: 深层锥体细胞 (主要输出层)
    - L6: 多形层 (反馈调节)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化皮层柱
        
        Args:
            config: 配置字典，包含：
                - total_neurons: 总神经元数量
                - layer_sizes: 各层神经元数量比例 [L1, L2/3, L4, L5, L6]
                - neuron_params: 各层神经元参数
                - connection_rules: 连接规则
        """
        # 设置默认配置
        self.default_config = {
            "total_neurons": 1000,
            "layer_sizes": [0.1, 0.3, 0.2, 0.3, 0.1],  # 各层比例
            "neuron_types": {
                "L1": "hh",
                "L2/3": "hh",
                "L4": "adex",
                "L5": "hh", 
                "L6": "izhikevich"
            },
            "connection_rules": {
                "intra_layer": 0.3,
                "feedforward": {
                    "L4->L2/3": 0.4,
                    "L2/3->L5": 0.5,
                    "L4->L5": 0.2
                },
                "feedback": {
                    "L5->L2/3": 0.3,
                    "L6->L4": 0.4
                }
            }
        }
        super().__init__(config={**self.default_config, **config})
        self._create_cortical_layers()
        self._create_cortical_connections()
    
    def _create_cortical_layers(self):
        """创建皮层各层神经元"""
        layer_sizes = [int(self.config["total_neurons"] * p) 
                      for p in self.config["layer_sizes"]]
        
        # 创建各层神经元
        neuron_id = 0
        for layer_name, size in zip(["L1", "L2/3", "L4", "L5", "L6"], layer_sizes):
            neuron_type = self.config["neuron_types"][layer_name]
            params = {"layer": layer_name}
            
            # 创建层
            layer = Layer(layer_name, size, neuron_type, params)
            self.layers[layer_name] = layer
            
            # 创建神经元
            neuron_id = layer.create_neurons(neuron_id)
            for neuron in layer.neurons:
                self.neurons[neuron.id] = neuron
    
    def _create_cortical_connections(self):
        """创建皮层特异性连接"""
        # 1. 层内连接
        for layer_name, layer in self.layers.items():
            self._connect_intra_layer(layer, 
                                    self.config["connection_rules"]["intra_layer"])
        
        # 2. 前馈连接
        for conn, prob in self.config["connection_rules"]["feedforward"].items():
            src, tgt = conn.split("->")
            self._connect_layers(self.layers[src], self.layers[tgt], prob)
        
        # 3. 反馈连接 
        for conn, prob in self.config["connection_rules"]["feedback"].items():
            src, tgt = conn.split("->")
            self._connect_layers(self.layers[src], self.layers[tgt], prob)
    
    def _connect_intra_layer(self, layer: Layer, prob: float):
        """创建层内局部连接"""
        for i, pre in enumerate(layer.neurons):
            for j, post in enumerate(layer.neurons):
                if i != j and np.random.rand() < prob:
                    # 20%抑制性神经元
                    is_inhibitory = np.random.rand() < 0.2
                    weight = np.random.uniform(0.1, 0.5) * (-1 if is_inhibitory else 1)
                    self.add_synapse(pre.id, post.id, weight)
    
    def add_thalamic_input(self, thalamic_neurons: List[Neuron], target_layer: str = "L4"):
        """添加丘脑输入到指定皮层"""
        target_layer = self.layers[target_layer]
        for thalamic_neuron in thalamic_neurons:
            for cortical_neuron in target_layer.neurons:
                if np.random.rand() < 0.3:  # 30%连接概率
                    self.add_synapse(thalamic_neuron.id, cortical_neuron.id, 0.5)
        """
        初始化神经网络
        
        Args:
            config: 网络配置字典，包含层定义、连接模式等
        """
        self.config = config
        self.layers: Dict[str, Layer] = {}
        self.neurons: Dict[int, Neuron] = {}
        self.synapses: Dict[Tuple[int, int], Synapse] = {}
        self.input_layer_name: Optional[str] = None
        self.output_layer_name: Optional[str] = None
        
        # 连接信息
        self.pre_synapses: Dict[int, List[Synapse]] = defaultdict(list)  # 前向突触
        self.post_synapses: Dict[int, List[Synapse]] = defaultdict(list)  # 后向突触
        
        # 初始化网络
        self._initialize_network()
    
    def _initialize_network(self) -> None:
        """初始化网络结构，创建层和神经元"""
        # 创建层
        next_id = 0
        for layer_config in self.config["layers"]:
            layer_name = layer_config["name"]
            layer_size = layer_config["size"]
            layer_type = layer_config.get("type", "hidden")
            
            # 确定神经元类型和参数
            neuron_type = layer_config.get("neuron_type", "lif")
            neuron_params = self.config["neuron_params"].get(
                layer_config.get("neuron_params", "default")
            )
            
            # 创建层
            layer = Layer(layer_name, layer_size, neuron_type, neuron_params)
            self.layers[layer_name] = layer
            
            # 创建神经元
            next_id = layer.create_neurons(next_id)
            
            # 添加神经元到全局字典
            for neuron in layer.neurons:
                self.neurons[neuron.id] = neuron
            
            # 记录输入和输出层
            if layer_type == "input" and self.input_layer_name is None:
                self.input_layer_name = layer_name
            elif layer_type == "output" and self.output_layer_name is None:
                self.output_layer_name = layer_name
        
        # 创建连接
        self._create_connections()
    
    def _create_connections(self) -> None:
        """创建神经元之间的连接"""
        # 获取连接模式
        connection_patterns = self.config.get("connection_patterns", {})
        
        # 默认连接模式
        default_pattern = {
            "probability": 0.1,
            "weight_init": "random_uniform",
        }
        
        # 遍历所有层对
        layer_names = list(self.layers.keys())
        for i, pre_layer_name in enumerate(layer_names):
            pre_layer = self.layers[pre_layer_name]
            
            for j, post_layer_name in enumerate(layer_names):
                post_layer = self.layers[post_layer_name]
                
                # 确定连接模式
                connection_type = None
                if i < j:  # 前馈连接
                    connection_type = "feedforward"
                elif i == j:  # 循环连接
                    connection_type = "recurrent"
                else:  # 反馈连接
                    continue  # 默认不创建反馈连接
                
                # 获取连接参数
                connection_params = connection_patterns.get(connection_type, default_pattern)
                probability = connection_params.get("probability", 0.1)
                weight_init = connection_params.get("weight_init", "random_uniform")
                
                # 创建连接
                self._connect_layers(
                    pre_layer, post_layer, probability, weight_init, connection_params
                )
    
    def _connect_layers(
        self,
        pre_layer: Layer,
        post_layer: Layer,
        probability: float,
        weight_init: str,
        connection_params: Dict[str, Any],
    ) -> None:
        """
        连接两个层
        
        Args:
            pre_layer: 前层
            post_layer: 后层
            probability: 连接概率
            weight_init: 权重初始化方法
            connection_params: 连接参数
        """
        # 获取突触参数
        synapse_type = connection_params.get("synapse_type", "static")
        synapse_params_key = connection_params.get("synapse_params", "default")
        synapse_params = self.config["synapse_params"].get(synapse_params_key, {})
        
        # 遍历所有可能的连接
        for pre_neuron in pre_layer.neurons:
            for post_neuron in post_layer.neurons:
                # 如果是同一个神经元，跳过
                if pre_neuron.id == post_neuron.id:
                    continue
                
                # 根据概率决定是否创建连接
                if random.random() < probability:
                    # 初始化权重
                    weight = self._initialize_weight(weight_init, connection_params)
                    
                    # 创建突触参数
                    params = synapse_params.copy()
                    params["weight"] = weight
                    
                    # 随机延迟
                    delay_range = params.get("delay_range", [1.0, 5.0])
                    params["delay"] = random.uniform(delay_range[0], delay_range[1])
                    
                    # 创建突触
                    synapse = create_synapse(
                        synapse_type, pre_neuron.id, post_neuron.id, params
                    )
                    
                    # 添加到突触字典
                    self.synapses[(pre_neuron.id, post_neuron.id)] = synapse
                    
                    # 添加到连接信息
                    self.pre_synapses[post_neuron.id].append(synapse)
                    self.post_synapses[pre_neuron.id].append(synapse)
    
    def _initialize_weight(self, weight_init: str, params: Dict[str, Any]) -> float:
        """
        初始化突触权重
        
        Args:
            weight_init: 权重初始化方法
            params: 初始化参数
            
        Returns:
            初始化的权重值
        """
        if weight_init == "constant":
            return params.get("weight_value", 1.0)
        
        elif weight_init == "random_uniform":
            weight_range = params.get("weight_range", [-0.1, 0.1])
            return random.uniform(weight_range[0], weight_range[1])
        
        elif weight_init == "random_normal":
            mean = params.get("weight_mean", 0.0)
            std = params.get("weight_std", 0.1)
            return random.normalvariate(mean, std)
        
        elif weight_init == "xavier":
            # Xavier/Glorot初始化
            n_in = params.get("n_in", 1)
            n_out = params.get("n_out", 1)
            limit = np.sqrt(6 / (n_in + n_out))
            return random.uniform(-limit, limit)
        
        else:
            return 1.0
    
    def reset(self) -> None:
        """重置网络状态"""
        # 重置所有神经元
        for neuron in self.neurons.values():
            neuron.reset()
        
        # 重置所有突触
        for synapse in self.synapses.values():
            synapse.reset()
    
    def set_input(self, input_values: List[float]) -> None:
        """
        设置输入层的值
        
        Args:
            input_values: 输入值列表
        """
        if self.input_layer_name is None:
            raise ValueError("网络没有定义输入层")
        
        input_layer = self.layers[self.input_layer_name]
        if len(input_values) != input_layer.size:
            raise ValueError(f"输入值数量 ({len(input_values)}) 与输入层大小 ({input_layer.size}) 不匹配")
        
        # 将输入值转换为电流
        for i, value in enumerate(input_values):
            neuron_id = input_layer.neuron_ids[i]
            self.neurons[neuron_id]._input_current = value
    
    def get_output(self) -> List[float]:
        """
        获取输出层的值
        
        Returns:
            输出值列表
        """
        if self.output_layer_name is None:
            raise ValueError("网络没有定义输出层")
        
        output_layer = self.layers[self.output_layer_name]
        output_values = []
        
        for neuron_id in output_layer.neuron_ids:
            neuron = self.neurons[neuron_id]
            # 使用膜电位作为输出值
            output_values.append(neuron.voltage)
        
        return output_values
    
    def step(self, dt: float) -> Dict[str, Any]:
        """
        执行一步网络更新
        
        Args:
            dt: 时间步长
            
        Returns:
            包含网络状态信息的字典
        """
        # 记录脉冲信息
        spikes = set()
        
        # 第一阶段：计算突触传递的电流
        neuron_currents = defaultdict(float)
        
        for pre_id, neuron in self.neurons.items():
            # 检查神经元是否产生脉冲
            pre_spike = hasattr(neuron, '_spike_history') and neuron._spike_history and neuron._spike_history[-1] == 1
            
            if pre_spike:
                spikes.add(pre_id)
            
            # 传递突触信号
            for synapse in self.post_synapses[pre_id]:
                post_id = synapse.post_id
                current = synapse.transmit(pre_spike, dt)
                neuron_currents[post_id] += current
        
        # 第二阶段：更新神经元状态
        for neuron_id, neuron in self.neurons.items():
            # 获取输入电流
            input_current = neuron_currents[neuron_id]
            
            # 添加外部输入电流（如果有）
            if hasattr(neuron, '_input_current'):
                input_current += neuron._input_current
                neuron._input_current = 0.0  # 重置外部输入
            
            # 更新神经元状态
            spike = neuron.update(input_current, dt)
            
            if spike:
                spikes.add(neuron_id)
        
        # 第三阶段：更新STDP突触（如果有）
        for neuron_id in spikes:
            # 更新后向突触的STDP
            for synapse in self.pre_synapses[neuron_id]:
                if isinstance(synapse, STDPSynapse):
                    synapse.update_post_spike(True)
        
        # 返回网络状态
        return {
            "spikes": list(spikes),
            "voltages": {nid: neuron.voltage for nid, neuron in self.neurons.items()},
            "weights": {(s.pre_id, s.post_id): s.weight for s in self.synapses.values()},
        }
    
    def simulate(self, inputs: List[List[float]], duration: float, dt: float) -> Dict[str, Any]:
        """
        模拟网络运行
        
        Args:
            inputs: 输入序列，每个时间步的输入值列表
            duration: 模拟持续时间
            dt: 时间步长
            
        Returns:
            包含模拟结果的字典
        """
        # 重置网络状态
        self.reset()
        
        # 计算时间步数
        steps = int(duration / dt)
        
        # 准备结果容器
        results = {
            "spikes": [],
            "voltages": [],
            "weights": [],
            "times": [],
        }
        
        # 执行模拟
        for step in range(steps):
            # 设置输入（如果有）
            if step < len(inputs):
                self.set_input(inputs[step])
            
            # 执行一步更新
            state = self.step(dt)
            
            # 记录结果
            results["spikes"].append(state["spikes"])
            results["voltages"].append(state["voltages"])
            results["weights"].append(state["weights"])
            results["times"].append(step * dt)
        
        return results