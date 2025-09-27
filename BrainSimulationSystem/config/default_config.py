"""
大脑模拟系统的默认配置参数
"""

from typing import Dict, Any, List, Optional, Union

# 神经网络配置
NETWORK_CONFIG = {
    # 网络结构
    "layers": [
        {"name": "input", "size": 100, "type": "input"},
        {"name": "hidden1", "size": 200, "type": "hidden"},
        {"name": "hidden2", "size": 200, "type": "hidden"},
        {"name": "output", "size": 50, "type": "output"}
    ],
    
    # 神经元参数
    "neuron_params": {
        "default": {
            "threshold": 1.0,           # 激活阈值
            "reset_potential": 0.0,     # 重置电位
            "resting_potential": 0.0,   # 静息电位
            "time_constant": 10.0,      # 时间常数 (ms)
            "refractory_period": 2.0,   # 不应期 (ms)
        },
        "inhibitory": {
            "threshold": 0.8,
            "reset_potential": -0.1,
            "resting_potential": -0.1,
            "time_constant": 8.0,
            "refractory_period": 1.5,
        }
    },
    
    # 突触参数
    "synapse_params": {
        "default": {
            "weight_range": [-1.0, 1.0],  # 权重范围
            "delay_range": [1.0, 5.0],    # 延迟范围 (ms)
            "plasticity": True,           # 是否启用可塑性
        },
        "excitatory": {
            "weight_range": [0.0, 1.0],
            "delay_range": [1.0, 5.0],
            "plasticity": True,
        },
        "inhibitory": {
            "weight_range": [-1.0, 0.0],
            "delay_range": [1.0, 3.0],
            "plasticity": True,
        }
    },
    
    # 连接模式
    "connection_patterns": {
        "feedforward": {
            "probability": 0.1,
            "weight_init": "random_uniform",
        },
        "recurrent": {
            "probability": 0.05,
            "weight_init": "random_normal",
        },
        "lateral_inhibition": {
            "probability": 0.2,
            "weight_init": "constant",
            "weight_value": -0.5,
        }
    },
    
    # 学习规则
    "learning_rules": {
        "stdp": {
            "enabled": True,
            "learning_rate": 0.01,
            "time_window": 20.0,  # ms
            "a_plus": 0.1,
            "a_minus": -0.1,
        },
        "homeostatic": {
            "enabled": True,
            "target_rate": 10.0,  # Hz
            "learning_rate": 0.001,
        }
    }
}

# 模拟参数
SIMULATION_CONFIG = {
    "dt": 0.1,                # 时间步长 (ms)
    "duration": 1000.0,       # 模拟持续时间 (ms)
    "random_seed": 42,        # 随机种子
    "backend": "numpy",       # 计算后端 ("numpy", "tensorflow", "torch")
    "device": "cpu",          # 计算设备 ("cpu", "gpu")
    "precision": "float32",   # 计算精度
    "batch_size": 1,          # 批处理大小
    "save_results": True,     # 是否保存结果
    "save_interval": 100,     # 保存间隔 (ms)
}

# 可视化参数
VISUALIZATION_CONFIG = {
    "enabled": True,
    "update_interval": 10.0,  # 更新间隔 (ms)
    "plot_types": ["raster", "voltage", "weight_matrix", "network_graph"],
    "max_neurons_to_display": 100,
    "colormap": "viridis",
    "3d_rendering": False,
}

# API配置
API_CONFIG = {
    "host": "localhost",
    "port": 8080,
    "debug": False,
    "enable_cors": True,
    "rate_limit": 100,  # 每分钟请求数
}

def get_config() -> Dict[str, Any]:
    """返回完整的配置字典"""
    return {
        "network": NETWORK_CONFIG,
        "simulation": SIMULATION_CONFIG,
        "visualization": VISUALIZATION_CONFIG,
        "api": API_CONFIG,
    }

def update_config(config_updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    更新配置参数
    
    Args:
        config_updates: 要更新的配置参数字典
        
    Returns:
        更新后的完整配置字典
    """
    config = get_config()
    
    # 递归更新嵌套字典
    def update_nested_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    return update_nested_dict(config, config_updates)