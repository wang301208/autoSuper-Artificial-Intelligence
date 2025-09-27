"""
病理模型模拟器

模拟各种神经系统疾病的情况。
"""

import numpy as np


class PathologicalModel:
    """病理模型基类"""
    
    def __init__(self, params=None):
        """
        初始化病理模型
        
        参数:
            params (dict): 配置参数
        """
        self.params = params or {}
        self.severity = self.params.get("severity", 0.5)  # 严重程度，0-1之间
    
    def apply_to_network(self, network):
        """
        将病理模型应用到神经网络
        
        参数:
            network: 神经网络实例
            
        返回:
            修改后的神经网络
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def apply_to_cognitive_system(self, cognitive_system):
        """
        将病理模型应用到认知系统
        
        参数:
            cognitive_system: 认知系统实例
            
        返回:
            修改后的认知系统
        """
        raise NotImplementedError("子类必须实现此方法")


class ADHDModel(PathologicalModel):
    """注意力缺陷多动障碍(ADHD)模型"""
    
    def __init__(self, params=None):
        """
        初始化ADHD模型
        
        参数:
            params (dict): 配置参数，可包含：
                - severity: 严重程度，0-1之间
                - dopamine_deficit: 多巴胺缺乏程度，0-1之间
                - norepinephrine_deficit: 去甲肾上腺素缺乏程度，0-1之间
        """
        super().__init__(params)
        self.dopamine_deficit = self.params.get("dopamine_deficit", self.severity)
        self.norepinephrine_deficit = self.params.get("norepinephrine_deficit", self.severity)
    
    def apply_to_network(self, network):
        """
        将ADHD模型应用到神经网络
        
        参数:
            network: 神经网络实例
            
        返回:
            修改后的神经网络
        """
        # 修改前额叶皮层和纹状体之间的连接
        if hasattr(network, "regions") and "prefrontal_cortex" in network.regions and "striatum" in network.regions:
            pfc = network.regions["prefrontal_cortex"]
            striatum = network.regions["striatum"]
            
            # 减弱前额叶皮层到纹状体的连接
            for synapse in network.get_synapses_between(pfc, striatum):
                synapse.weight *= (1 - self.dopamine_deficit * 0.7)
        
        # 修改神经元的多巴胺敏感性
        for neuron in network.neurons:
            if hasattr(neuron, "dopamine_sensitivity"):
                neuron.dopamine_sensitivity *= (1 - self.dopamine_deficit)
        
        return network
    
    def apply_to_cognitive_system(self, cognitive_system):
        """
        将ADHD模型应用到认知系统
        
        参数:
            cognitive_system: 认知系统实例
            
        返回:
            修改后的认知系统
        """
        # 修改注意力系统
        if hasattr(cognitive_system, "attention_system"):
            # 降低注意力持续时间
            if hasattr(cognitive_system.attention_system, "focus_duration"):
                cognitive_system.attention_system.focus_duration *= (1 - self.severity * 0.6)
            
            # 增加注意力转移概率
            if hasattr(cognitive_system.attention_system, "shift_probability"):
                cognitive_system.attention_system.shift_probability *= (1 + self.severity * 0.8)
        
        # 修改工作记忆系统
        if hasattr(cognitive_system, "working_memory"):
            # 降低工作记忆容量
            if hasattr(cognitive_system.working_memory, "capacity"):
                cognitive_system.working_memory.capacity = max(1, int(cognitive_system.working_memory.capacity * (1 - self.severity * 0.4)))
            
            # 增加记忆衰减率
            if hasattr(cognitive_system.working_memory, "decay_rate"):
                cognitive_system.working_memory.decay_rate *= (1 + self.severity * 0.5)
        
        # 修改神经调质系统
        if hasattr(cognitive_system, "neuromodulatory_system"):
            # 降低多巴胺水平
            if hasattr(cognitive_system.neuromodulatory_system, "dopamine_level"):
                cognitive_system.neuromodulatory_system.dopamine_level *= (1 - self.dopamine_deficit)
            
            # 降低去甲肾上腺素水平
            if hasattr(cognitive_system.neuromodulatory_system, "norepinephrine_level"):
                cognitive_system.neuromodulatory_system.norepinephrine_level *= (1 - self.norepinephrine_deficit)
        
        return cognitive_system


class SchizophreniaModel(PathologicalModel):
    """精神分裂症模型"""
    
    def __init__(self, params=None):
        """
        初始化精神分裂症模型
        
        参数:
            params (dict): 配置参数，可包含：
                - severity: 严重程度，0-1之间
                - dopamine_excess: 多巴胺过量程度，0-1之间
                - glutamate_deficit: 谷氨酸缺乏程度，0-1之间
                - gaba_deficit: GABA缺乏程度，0-1之间
        """
        super().__init__(params)
        self.dopamine_excess = self.params.get("dopamine_excess", self.severity)
        self.glutamate_deficit = self.params.get("glutamate_deficit", self.severity * 0.8)
        self.gaba_deficit = self.params.get("gaba_deficit", self.severity * 0.7)
    
    def apply_to_network(self, network):
        """
        将精神分裂症模型应用到神经网络
        
        参数:
            network: 神经网络实例
            
        返回:
            修改后的神经网络
        """
        # 修改中脑边缘多巴胺通路
        if hasattr(network, "regions") and "mesolimbic_pathway" in network.regions:
            mesolimbic = network.regions["mesolimbic_pathway"]
            
            # 增强多巴胺神经元的活动
            for neuron in mesolimbic.neurons:
                if hasattr(neuron, "baseline_activity"):
                    neuron.baseline_activity *= (1 + self.dopamine_excess)
        
        # 减弱NMDA受体功能
        for synapse in network.synapses:
            if hasattr(synapse, "receptor_types") and "NMDA" in synapse.receptor_types:
                synapse.receptor_types["NMDA"] *= (1 - self.glutamate_deficit)
        
        # 减弱GABA抑制性突触
        for neuron in network.neurons:
            if hasattr(neuron, "neurotransmitter") and neuron.neurotransmitter == "GABA":
                for synapse in neuron.output_synapses:
                    synapse.weight *= (1 - self.gaba_deficit)
        
        return network
    
    def apply_to_cognitive_system(self, cognitive_system):
        """
        将精神分裂症模型应用到认知系统
        
        参数:
            cognitive_system: 认知系统实例
            
        返回:
            修改后的认知系统
        """
        # 修改感知过滤系统
        if hasattr(cognitive_system, "perception_filter"):
            # 降低感知过滤能力
            if hasattr(cognitive_system.perception_filter, "threshold"):
                cognitive_system.perception_filter.threshold *= (1 - self.severity * 0.8)
            
            # 增加噪声
            if hasattr(cognitive_system.perception_filter, "noise_level"):
                cognitive_system.perception_filter.noise_level *= (1 + self.severity)
        
        # 修改工作记忆系统
        if hasattr(cognitive_system, "working_memory"):
            # 增加记忆干扰
            if hasattr(cognitive_system.working_memory, "interference_level"):
                cognitive_system.working_memory.interference_level *= (1 + self.severity * 0.7)
        
        # 修改神经调质系统
        if hasattr(cognitive_system, "neuromodulatory_system"):
            # 增加多巴胺水平
            if hasattr(cognitive_system.neuromodulatory_system, "dopamine_level"):
                cognitive_system.neuromodulatory_system.dopamine_level *= (1 + self.dopamine_excess)
            
            # 降低谷氨酸水平
            if hasattr(cognitive_system.neuromodulatory_system, "glutamate_level"):
                cognitive_system.neuromodulatory_system.glutamate_level *= (1 - self.glutamate_deficit)
            
            # 降低GABA水平
            if hasattr(cognitive_system.neuromodulatory_system, "gaba_level"):
                cognitive_system.neuromodulatory_system.gaba_level *= (1 - self.gaba_deficit)
        
        return cognitive_system


class AlzheimersModel(PathologicalModel):
    """阿尔茨海默病模型"""
    
    def __init__(self, params=None):
        """
        初始化阿尔茨海默病模型
        
        参数:
            params (dict): 配置参数，可包含：
                - severity: 严重程度，0-1之间
                - amyloid_level: 淀粉样蛋白水平，0-1之间
                - tau_level: Tau蛋白水平，0-1之间
                - acetylcholine_deficit: 乙酰胆碱缺乏程度，0-1之间
        """
        super().__init__(params)
        self.amyloid_level = self.params.get("amyloid_level", self.severity)
        self.tau_level = self.params.get("tau_level", self.severity * 0.9)
        self.acetylcholine_deficit = self.params.get("acetylcholine_deficit", self.severity * 0.8)
    
    def apply_to_network(self, network):
        """
        将阿尔茨海默病模型应用到神经网络
        
        参数:
            network: 神经网络实例
            
        返回:
            修改后的神经网络
        """
        # 随机移除突触，模拟神经元连接丢失
        synapses_to_remove = []
        for i, synapse in enumerate(network.synapses):
            if np.random.random() < self.severity * 0.3:
                synapses_to_remove.append(i)
        
        # 从后向前移除，避免索引问题
        for i in sorted(synapses_to_remove, reverse=True):
            if i < len(network.synapses):
                network.synapses.pop(i)
        
        # 降低海马体和前额叶皮层的连接强度
        if hasattr(network, "regions") and "hippocampus" in network.regions and "prefrontal_cortex" in network.regions:
            hippocampus = network.regions["hippocampus"]
            pfc = network.regions["prefrontal_cortex"]
            
            for synapse in network.get_synapses_between(hippocampus, pfc):
                synapse.weight *= (1 - self.severity * 0.6)
            
            for synapse in network.get_synapses_between(pfc, hippocampus):
                synapse.weight *= (1 - self.severity * 0.6)
        
        # 降低乙酰胆碱能神经元的活动
        for neuron in network.neurons:
            if hasattr(neuron, "neurotransmitter") and neuron.neurotransmitter == "acetylcholine":
                if hasattr(neuron, "baseline_activity"):
                    neuron.baseline_activity *= (1 - self.acetylcholine_deficit)
        
        return network
    
    def apply_to_cognitive_system(self, cognitive_system):
        """
        将阿尔茨海默病模型应用到认知系统
        
        参数:
            cognitive_system: 认知系统实例
            
        返回:
            修改后的认知系统
        """
        # 修改记忆系统
        if hasattr(cognitive_system, "memory_system"):
            # 降低长期记忆形成能力
            if hasattr(cognitive_system.memory_system, "encoding_strength"):
                cognitive_system.memory_system.encoding_strength *= (1 - self.severity * 0.7)
            
            # 增加长期记忆检索难度
            if hasattr(cognitive_system.memory_system, "retrieval_threshold"):
                cognitive_system.memory_system.retrieval_threshold *= (1 + self.severity * 0.8)
        
        # 修改工作记忆系统
        if hasattr(cognitive_system, "working_memory"):
            # 降低工作记忆容量
            if hasattr(cognitive_system.working_memory, "capacity"):
                cognitive_system.working_memory.capacity = max(1, int(cognitive_system.working_memory.capacity * (1 - self.severity * 0.5)))
            
            # 增加记忆衰减率
            if hasattr(cognitive_system.working_memory, "decay_rate"):
                cognitive_system.working_memory.decay_rate *= (1 + self.severity * 0.6)
        
        # 修改神经调质系统
        if hasattr(cognitive_system, "neuromodulatory_system"):
            # 降低乙酰胆碱水平
            if hasattr(cognitive_system.neuromodulatory_system, "acetylcholine_level"):
                cognitive_system.neuromodulatory_system.acetylcholine_level *= (1 - self.acetylcholine_deficit)
        
        return cognitive_system


class ParkinsonModel(PathologicalModel):
    """帕金森病模型"""
    
    def __init__(self, params=None):
        """
        初始化帕金森病模型
        
        参数:
            params (dict): 配置参数，可包含：
                - severity: 严重程度，0-1之间
                - dopamine_deficit: 多巴胺缺乏程度，0-1之间
                - substantia_nigra_degeneration: 黑质变性程度，0-1之间
        """
        super().__init__(params)
        self.dopamine_deficit = self.params.get("dopamine_deficit", self.severity)
        self.substantia_nigra_degeneration = self.params.get("substantia_nigra_degeneration", self.severity * 0.9)
    
    def apply_to_network(self, network):
        """
        将帕金森病模型应用到神经网络
        
        参数:
            network: 神经网络实例
            
        返回:
            修改后的神经网络
        """
        # 降低黑质多巴胺神经元的活动
        if hasattr(network, "regions") and "substantia_nigra" in network.regions:
            substantia_nigra = network.regions["substantia_nigra"]
            
            # 移除部分黑质神经元，模拟神经元死亡
            neurons_to_remove = []
            for i, neuron in enumerate(substantia_nigra.neurons):
                if np.random.random() < self.substantia_nigra_degeneration:
                    neurons_to_remove.append(i)
            
            # 从后向前移除，避免索引问题
            for i in sorted(neurons_to_remove, reverse=True):
                if i < len(substantia_nigra.neurons):
                    substantia_nigra.neurons.pop(i)
            
            # 降低剩余神经元的活动
            for neuron in substantia_nigra.neurons:
                if hasattr(neuron, "baseline_activity"):
                    neuron.baseline_activity *= (1 - self.dopamine_deficit)
        
        # 修改基底神经节回路
        if hasattr(network, "regions") and "basal_ganglia" in network.regions:
            basal_ganglia = network.regions["basal_ganglia"]
            
            # 增强间接通路
            for neuron in basal_ganglia.neurons:
                if hasattr(neuron, "pathway") and neuron.pathway == "indirect":
                    for synapse in neuron.output_synapses:
                        synapse.weight *= (1 + self.severity * 0.5)
            
            # 减弱直接通路
            for neuron in basal_ganglia.neurons:
                if hasattr(neuron, "pathway") and neuron.pathway == "direct":
                    for synapse in neuron.output_synapses:
                        synapse.weight *= (1 - self.severity * 0.5)
        
        return network
    
    def apply_to_cognitive_system(self, cognitive_system):
        """
        将帕金森病模型应用到认知系统
        
        参数:
            cognitive_system: 认知系统实例
            
        返回:
            修改后的认知系统
        """
        # 修改运动系统
        if hasattr(cognitive_system, "motor_system"):
            # 增加运动启动延迟
            if hasattr(cognitive_system.motor_system, "initiation_threshold"):
                cognitive_system.motor_system.initiation_threshold *= (1 + self.severity * 0.8)
            
            # 增加运动抖动
            if hasattr(cognitive_system.motor_system, "tremor_level"):
                cognitive_system.motor_system.tremor_level = max(
                    cognitive_system.motor_system.tremor_level,
                    self.severity * 0.7
                )
            else:
                cognitive_system.motor_system.tremor_level = self.severity * 0.7
        
        # 修改认知灵活性
        if hasattr(cognitive_system, "cognitive_flexibility"):
            # 降低认知灵活性
            if hasattr(cognitive_system.cognitive_flexibility, "switching_speed"):
                cognitive_system.cognitive_flexibility.switching_speed *= (1 - self.severity * 0.4)
        
        # 修改神经调质系统
        if hasattr(cognitive_system, "neuromodulatory_system"):
            # 降低多巴胺水平
            if hasattr(cognitive_system.neuromodulatory_system, "dopamine_level"):
                cognitive_system.neuromodulatory_system.dopamine_level *= (1 - self.dopamine_deficit)
        
        return cognitive_system


def create_pathological_model(model_type, params=None):
    """
    创建病理模型
    
    参数:
        model_type (str): 模型类型
        params (dict): 配置参数
        
    返回:
        病理模型实例
    """
    models = {
        "adhd": ADHDModel,
        "schizophrenia": SchizophreniaModel,
        "alzheimers": AlzheimersModel,
        "parkinson": ParkinsonModel
    }
    
    if model_type not in models:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    return models[model_type](params)