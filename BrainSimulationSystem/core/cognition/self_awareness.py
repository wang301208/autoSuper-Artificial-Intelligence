"""
自我意识框架

实现自我表征、自传体记忆和元认知整合
"""

from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class BodyRepresentation:
    """身体状态表征"""
    limb_positions: Dict[str, float]
    sensory_inputs: Dict[str, float]
    
    def update_from_sensors(self, sensor_data: Dict):
        """根据感觉输入更新身体模型"""
        for k, v in sensor_data.items():
            if k in self.limb_positions:
                self.limb_positions[k] = 0.9 * self.limb_positions[k] + 0.1 * v

@dataclass
class AutobiographicalMemory:
    """自传体记忆系统"""
    episodes: List[Dict]
    self_concept: Dict[str, float]
    
    def retrieve_episode(self, query: Dict) -> Dict:
        """检索相关记忆片段"""
        # 计算查询与记忆的相似度
        similarities = [
            (ep, self._calculate_similarity(ep, query))
            for ep in self.episodes
        ]
        return max(similarities, key=lambda x: x[1])[0]
        
    def _calculate_similarity(self, episode: Dict, query: Dict) -> float:
        """计算记忆相似度"""
        score = 0.0
        for k, v in query.items():
            if k in episode:
                score += 0.5 if episode[k] == v else 0.1
        return score / len(query) if query else 0.0

class SelfAwarenessFramework:
    def __init__(self):
        self.body_model = BodyRepresentation(
            limb_positions={'arm': 0.5, 'head': 0.0},
            sensory_inputs={}
        )
        self.memory = AutobiographicalMemory(
            episodes=[],
            self_concept={'agency': 0.7, 'identity': 0.8}
        )
        self.meta_cognition = None  # 需接入元认知模块
        
    def update_self_state(self, sensor_data: Dict = None):
        """更新自我状态"""
        if sensor_data:
            self.body_model.update_from_sensors(sensor_data)
            
        # 生成当前自我表征
        current_self = {
            'body': self.body_model.limb_positions,
            'time': self._get_internal_clock()
        }
        
        # 记忆当前状态
        self.memory.episodes.append(current_self)
        
        # 更新自我概念
        if len(self.memory.episodes) > 10:
            self._update_self_concept()
            
    def _get_internal_clock(self) -> float:
        """内部时间感知"""
        return np.random.uniform(0, 1)  # 简化实现
        
    def _update_self_concept(self):
        """基于近期经历更新自我概念"""
        recent_agency = sum(
            ep.get('action_intensity', 0.5) 
            for ep in self.memory.episodes[-10:]
        ) / 10
        self.memory.self_concept['agency'] = 0.9 * self.memory.self_concept['agency'] + 0.1 * recent_agency
        
    def recognize_self(self, mirror_input: Dict) -> bool:
        """镜像自我识别测试"""
        # 比较观察到的动作与自身运动模式
        motion_similarity = sum(
            abs(mirror_input[k] - self.body_model.limb_positions.get(k, 0))
            for k in mirror_input
        ) / len(mirror_input)
        return motion_similarity < 0.2