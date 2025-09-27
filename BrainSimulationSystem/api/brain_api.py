"""
大脑模拟系统API接口

提供RESTful API接口，允许外部系统访问和控制大脑模拟系统。
"""

import sys
import os
import json
import threading
import time
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify, Response
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from BrainSimulationSystem.models.cognitive_controller import CognitiveControllerBuilder
from BrainSimulationSystem.core.network import NeuralNetwork


class BrainSimulationAPI:
    """大脑模拟系统API接口类"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        """初始化API接口"""
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        
        # 创建认知控制器
        self.controller = self._create_controller()
        
        # 模拟状态
        self.simulation_running = False
        self.simulation_thread = None
        self.simulation_results = []
        self.current_step = 0
        self.max_steps = 100
        self.step_interval = 0.5  # 秒
        
        # 注册路由
        self._register_routes()
    
    def _create_controller(self):
        """创建认知控制器"""
        # 配置注意力系统参数
        attention_params = {
            "ach_sensitivity": 1.2,
            "bottom_up_weight": 0.6,
            "top_down_weight": 0.4,
            "attention_span": 3
        }
        
        # 配置工作记忆参数
        memory_params = {
            "capacity": 7,
            "decay_rate": 0.05,
            "ach_sensitivity": 1.5,
            "attention_boost": 0.4,
            "encoding_factor": 1.3,
            "retrieval_threshold": 0.35,
            "interference_factor": 0.75
        }
        
        # 使用构建器创建控制器
        builder = CognitiveControllerBuilder()
        builder.with_attention_params(attention_params)
        builder.with_working_memory_params(memory_params)
        
        return builder.build()
    
    def _register_routes(self):
        """注册API路由"""
        # 系统信息
        self.app.route('/api/info', methods=['GET'])(self.get_system_info)
        
        # 认知控制
        self.app.route('/api/cognitive/state', methods=['GET'])(self.get_cognitive_state)
        self.app.route('/api/cognitive/process', methods=['POST'])(self.process_cognitive_input)
        
        # 注意力系统
        self.app.route('/api/attention/focus', methods=['GET'])(self.get_attention_focus)
        self.app.route('/api/attention/params', methods=['GET', 'PUT'])(self.attention_params)
        
        # 工作记忆
        self.app.route('/api/memory/content', methods=['GET'])(self.get_memory_content)
        self.app.route('/api/memory/item/<key>', methods=['GET', 'PUT', 'DELETE'])(self.memory_item)
        
        # 神经调质
        self.app.route('/api/neuromodulators', methods=['GET', 'PUT'])(self.neuromodulators)
        
        # 模拟控制
        self.app.route('/api/simulation/start', methods=['POST'])(self.start_simulation)
        self.app.route('/api/simulation/stop', methods=['POST'])(self.stop_simulation)
        self.app.route('/api/simulation/status', methods=['GET'])(self.simulation_status)
        self.app.route('/api/simulation/results', methods=['GET'])(self.simulation_results_endpoint)
    
    def run(self):
        """运行API服务器"""
        self.app.run(host=self.host, port=self.port, debug=False)
    
    # API端点实现
    
    def get_system_info(self):
        """获取系统信息"""
        info = {
            "name": "大脑模拟系统",
            "version": "1.0.0",
            "components": list(self.controller.components.keys()),
            "status": "running" if self.simulation_running else "idle"
        }
        return jsonify(info)
    
    def get_cognitive_state(self):
        """获取认知状态"""
        state = {
            "cognitive_state": self.controller.state.name,
            "state_history": [s.name for s in self.controller.state_history[-10:]],
            "neuromodulators": self.controller.neuromodulators
        }
        return jsonify(state)
    
    def process_cognitive_input(self):
        """处理认知输入"""
        try:
            data = request.json
            if not data:
                return jsonify({"error": "无效的输入数据"}), 400
                
            # 处理认知周期
            result = self.controller.process(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    def get_attention_focus(self):
        """获取注意力焦点"""
        if "attention" not in self.controller.components:
            return jsonify({"error": "注意力系统不可用"}), 404
            
        attention = self.controller.components["attention"]
        focus = attention.focus if hasattr(attention, "focus") else []
        
        return jsonify({
            "focus": focus,
            "attention_map": attention.salience_map if hasattr(attention, "salience_map") else {}
        })
    
    def attention_params(self):
        """获取或设置注意力参数"""
        if "attention" not in self.controller.components:
            return jsonify({"error": "注意力系统不可用"}), 404
            
        attention = self.controller.components["attention"]
        
        if request.method == 'GET':
            params = {
                "ach_sensitivity": attention.ach_sensitivity,
                "bottom_up_weight": attention.bottom_up_weight,
                "top_down_weight": attention.top_down_weight,
                "attention_span": attention.attention_span
            }
            return jsonify(params)
        else:  # PUT
            try:
                data = request.json
                if not data:
                    return jsonify({"error": "无效的参数数据"}), 400
                    
                # 更新参数
                if "ach_sensitivity" in data:
                    attention.ach_sensitivity = float(data["ach_sensitivity"])
                if "bottom_up_weight" in data:
                    attention.bottom_up_weight = float(data["bottom_up_weight"])
                if "top_down_weight" in data:
                    attention.top_down_weight = float(data["top_down_weight"])
                if "attention_span" in data:
                    attention.attention_span = int(data["attention_span"])
                    
                return jsonify({"status": "success"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    def get_memory_content(self):
        """获取工作记忆内容"""
        if "working_memory" not in self.controller.components:
            return jsonify({"error": "工作记忆系统不可用"}), 404
            
        memory = self.controller.components["working_memory"]
        
        content = {
            "items": memory.get_all_items(),
            "strengths": memory.item_strengths if hasattr(memory, "item_strengths") else {},
            "capacity": {
                "used": len(memory.memory_items) if hasattr(memory, "memory_items") else 0,
                "total": memory.capacity if hasattr(memory, "capacity") else 0
            }
        }
        
        return jsonify(content)
    
    def memory_item(self, key):
        """获取、设置或删除工作记忆项"""
        if "working_memory" not in self.controller.components:
            return jsonify({"error": "工作记忆系统不可用"}), 404
            
        memory = self.controller.components["working_memory"]
        
        if request.method == 'GET':
            # 获取记忆项
            item = memory.get_item(key)
            if item is None:
                return jsonify({"error": f"记忆项 '{key}' 不存在"}), 404
                
            return jsonify({
                "key": key,
                "value": item,
                "strength": memory.item_strengths.get(key, 0) if hasattr(memory, "item_strengths") else 0
            })
        
        elif request.method == 'PUT':
            # 设置记忆项
            try:
                data = request.json
                if not data or "value" not in data:
                    return jsonify({"error": "无效的记忆项数据"}), 400
                    
                value = data["value"]
                priority = data.get("priority", 0.5)
                
                # 存储记忆项
                memory._store_item(key, value, priority)
                
                return jsonify({"status": "success"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        else:  # DELETE
            # 删除记忆项
            if not hasattr(memory, "_delete_item"):
                return jsonify({"error": "不支持删除记忆项"}), 501
                
            memory._delete_item(key)
            return jsonify({"status": "success"})
    
    def neuromodulators(self):
        """获取或设置神经调质水平"""
        if request.method == 'GET':
            return jsonify(self.controller.neuromodulators)
        else:  # PUT
            try:
                data = request.json
                if not data:
                    return jsonify({"error": "无效的神经调质数据"}), 400
                    
                # 更新神经调质水平
                for name, level in data.items():
                    if name in self.controller.neuromodulators:
                        self.controller.set_neuromodulator(name, float(level))
                        
                return jsonify({"status": "success"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    def start_simulation(self):
        """启动模拟"""
        if self.simulation_running:
            return jsonify({"error": "模拟已在运行中"}), 400
            
        try:
            data = request.json or {}
            self.max_steps = int(data.get("steps", 100))
            self.step_interval = float(data.get("interval", 0.5))
            
            # 重置模拟状态
            self.current_step = 0
            self.simulation_results = []
            self.simulation_running = True
            
            # 启动模拟线程
            self.simulation_thread = threading.Thread(target=self._run_simulation)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
            return jsonify({
                "status": "started",
                "max_steps": self.max_steps,
                "step_interval": self.step_interval
            })
        except Exception as e:
            self.simulation_running = False
            return jsonify({"error": str(e)}), 500
    
    def stop_simulation(self):
        """停止模拟"""
        if not self.simulation_running:
            return jsonify({"error": "模拟未在运行"}), 400
            
        self.simulation_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)
            
        return jsonify({
            "status": "stopped",
            "completed_steps": self.current_step,
            "results_count": len(self.simulation_results)
        })
    
    def simulation_status(self):
        """获取模拟状态"""
        status = {
            "running": self.simulation_running,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "progress": (self.current_step / self.max_steps) * 100 if self.max_steps > 0 else 0,
            "results_count": len(self.simulation_results)
        }
        return jsonify(status)
    
    def simulation_results_endpoint(self):
        """获取模拟结果"""
        # 获取分页参数
        page = int(request.args.get("page", 0))
        page_size = int(request.args.get("page_size", 10))
        
        # 计算分页
        start = page * page_size
        end = start + page_size
        
        # 获取结果子集
        results_subset = self.simulation_results[start:end] if start < len(self.simulation_results) else []
        
        return jsonify({
            "total": len(self.simulation_results),
            "page": page,
            "page_size": page_size,
            "results": results_subset
        })
    
    def _run_simulation(self):
        """运行模拟线程"""
        try:
            for step in range(self.max_steps):
                if not self.simulation_running:
                    break
                    
                self.current_step = step
                
                # 生成模拟输入
                sensory_input = self._generate_sensory_input(step)
                neuromodulators = self._modulate_neuromodulators(step)
                
                # 准备控制器输入
                controller_input = {
                    "sensory_input": sensory_input,
                    "neuromodulators": neuromodulators,
                    "task_goal": "simulation_task",
                    "decision_required": step % 10 == 0,
                    "response_required": step % 10 == 5,
                    "response_complete": step % 10 == 6
                }
                
                # 处理认知周期
                result = self.controller.process(controller_input)
                self.simulation_results.append(result)
                
                # 等待下一步
                time.sleep(self.step_interval)
                
        except Exception as e:
            print(f"模拟线程错误: {e}")
        finally:
            self.simulation_running = False
    
    def _generate_sensory_input(self, step: int) -> Dict[str, Any]:
        """生成模拟的感觉输入"""
        # 模拟不同的感觉输入项
        sensory_input = {
            "visual_object_1": {"shape": "circle", "color": "red", "size": 0.8},
            "visual_object_2": {"shape": "square", "color": "blue", "size": 0.5},
            "auditory_input": {"frequency": 440, "volume": 0.7}
        }
        
        # 每3个时间步改变一个对象的属性
        if step % 3 == 0:
            colors = ["red", "blue", "green", "yellow", "purple"]
            sensory_input["visual_object_1"]["color"] = colors[step % len(colors)]
        
        # 每5个时间步添加一个新对象
        if step % 5 == 0:
            sensory_input[f"new_object_{step}"] = {
                "shape": "star", 
                "color": "yellow", 
                "size": 0.9,
                "novelty": 1.0
            }
            
        return sensory_input
    
    def _modulate_neuromodulators(self, step: int) -> Dict[str, float]:
        """模拟神经调质水平变化"""
        # 基础水平
        base_level = 0.5
        
        # 模拟乙酰胆碱水平变化 (注意力相关)
        ach_level = base_level + 0.3 * np.sin(step / 10)
        
        # 模拟多巴胺水平变化 (奖励相关)
        dopa_level = base_level
        if step % 15 == 0:
            dopa_level += 0.4  # 奖励峰值
        
        # 模拟去甲肾上腺素水平变化 (警觉相关)
        ne_level = base_level + 0.2 * np.random.random()
        
        # 模拟5-羟色胺水平变化 (情绪相关)
        serotonin_level = base_level + 0.1 * np.sin(step / 20)
        
        return {
            "acetylcholine": max(0.0, min(1.0, ach_level)),
            "dopamine": max(0.0, min(1.0, dopa_level)),
            "norepinephrine": max(0.0, min(1.0, ne_level)),
            "serotonin": max(0.0, min(1.0, serotonin_level))
        }


def main():
    """主函数"""
    api = BrainSimulationAPI(host='0.0.0.0', port=5000)
    print("启动大脑模拟系统API服务器...")
    api.run()


if __name__ == "__main__":
    main()