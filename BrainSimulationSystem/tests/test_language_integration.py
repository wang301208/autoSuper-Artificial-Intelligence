"""
语言处理模块集成测试

测试语言处理模块与其他系统的集成功能。
"""

import unittest
import numpy as np
from ..models.language_processing import LanguageProcessor
from ..models.working_memory import AchModulatedWorkingMemory
from ..models.attention import AcetylcholineModulatedAttention


class TestLanguageIntegration(unittest.TestCase):
    """语言处理集成测试类"""
    
    def setUp(self):
        """初始化测试环境"""
        # 创建语言处理器
        self.language_processor = LanguageProcessor()
        
        # 创建工作记忆系统
        self.working_memory = AchModulatedWorkingMemory()
        
        # 创建注意力系统
        self.attention_system = AcetylcholineModulatedAttention()
        
        # 连接系统
        self._connect_systems()
    
    def _connect_systems(self):
        """连接各系统"""
        # 语言处理器使用工作记忆
        self.language_processor.working_memory = self.working_memory
        
        # 语言处理器使用注意力系统
        self.language_processor.attention_system = self.attention_system
    
    def test_speech_processing_with_memory(self):
        """测试带工作记忆的语音处理"""
        # 在工作记忆中存储一些信息
        self.working_memory.add_item("context", "animal")
        
        # 处理语音输入
        phonemes = ['d', 'o', 'g']
        result = self.language_processor.process_speech(phonemes)
        
        # 验证结果
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['words'][0], 'dog')
        self.assertIn('animal', result['meaning'])
        
        # 验证工作记忆更新
        self.assertIn('dog', [item['content'] for item in self.working_memory.items])
    
    def test_attention_modulation(self):
        """测试注意力对语言处理的影响"""
        # 设置注意力焦点
        self.attention_system.set_focus("animals")
        
        # 处理两个可能的词汇输入
        phonemes1 = ['d', 'o', 'g']  # dog
        phonemes2 = ['k', 'a', 't']  # cat
        
        # 处理第一个输入
        result1 = self.language_processor.process_speech(phonemes1)
        
        # 改变注意力焦点
        self.attention_system.set_focus("objects")
        
        # 处理第二个输入
        result2 = self.language_processor.process_speech(phonemes2)
        
        # 验证注意力影响
        self.assertGreater(
            result1['visualization']['words']['dog']['activation'],
            result2['visualization']['words']['cat']['activation']
        )
    
    def test_language_generation(self):
        """测试语言生成功能"""
        # 激活语义概念
        self.language_processor.semantic_network.activate_concept("dog", 0.8)
        self.language_processor.semantic_network.activate_concept("chased", 0.7)
        self.language_processor.semantic_network.activate_concept("cat", 0.6)
        
        # 生成语言
        phonemes = self.language_processor.generate_speech({
            'intention': 'describe_action'
        })
        
        # 验证生成的音素序列
        self.assertTrue(len(phonemes) > 0)
        self.assertTrue('d' in phonemes or 'k' in phonemes)
    
    def test_visualization_data(self):
        """测试可视化数据生成"""
        # 处理语音输入
        phonemes = ['d', 'o', 'g']
        result = self.language_processor.process_speech(phonemes)
        
        # 验证可视化数据
        self.assertIn('phonemes', result['visualization'])
        self.assertIn('dog', result['visualization']['words'])
        self.assertIn('dog', result['visualization']['semantic'])
        self.assertIsNotNone(result['visualization']['syntax'])
        
        # 获取独立可视化数据
        viz_data = self.language_processor.get_visualization_data()
        self.assertEqual(viz_data['last_update'], result['timestamp'])


if __name__ == '__main__':
    unittest.main()