"""
语言处理模块

实现大脑的语言理解功能，包括语音识别、语义理解和语言生成。
"""

import numpy as np
from collections import defaultdict


class PhonemeProcessor:
    """音素处理器"""
    
    def __init__(self, params=None):
        """
        初始化音素处理器
        
        参数:
            params (dict): 配置参数
        """
        self.params = params or {}
        
        # 音素特征映射
        self.phoneme_features = {
            # 元音
            'i': {'type': 'vowel', 'front': 1.0, 'high': 1.0, 'rounded': 0.0},
            'e': {'type': 'vowel', 'front': 1.0, 'high': 0.5, 'rounded': 0.0},
            'a': {'type': 'vowel', 'front': 0.5, 'high': 0.0, 'rounded': 0.0},
            'o': {'type': 'vowel', 'front': 0.0, 'high': 0.5, 'rounded': 1.0},
            'u': {'type': 'vowel', 'front': 0.0, 'high': 1.0, 'rounded': 1.0},
            
            # 辅音
            'p': {'type': 'consonant', 'place': 'bilabial', 'manner': 'stop', 'voiced': 0.0},
            'b': {'type': 'consonant', 'place': 'bilabial', 'manner': 'stop', 'voiced': 1.0},
            't': {'type': 'consonant', 'place': 'alveolar', 'manner': 'stop', 'voiced': 0.0},
            'd': {'type': 'consonant', 'place': 'alveolar', 'manner': 'stop', 'voiced': 1.0},
            'k': {'type': 'consonant', 'place': 'velar', 'manner': 'stop', 'voiced': 0.0},
            'g': {'type': 'consonant', 'place': 'velar', 'manner': 'stop', 'voiced': 1.0},
            's': {'type': 'consonant', 'place': 'alveolar', 'manner': 'fricative', 'voiced': 0.0},
            'z': {'type': 'consonant', 'place': 'alveolar', 'manner': 'fricative', 'voiced': 1.0},
            'm': {'type': 'consonant', 'place': 'bilabial', 'manner': 'nasal', 'voiced': 1.0},
            'n': {'type': 'consonant', 'place': 'alveolar', 'manner': 'nasal', 'voiced': 1.0},
            'l': {'type': 'consonant', 'place': 'alveolar', 'manner': 'lateral', 'voiced': 1.0},
            'r': {'type': 'consonant', 'place': 'alveolar', 'manner': 'approximant', 'voiced': 1.0}
        }
        
        # 音位变体规则
        self.allophonic_rules = [
            {'context': ('_', 'i'), 'change': ('t', 'tʃ')},  # /t/ -> /tʃ/ before /i/
            {'context': ('s', '_'), 'change': ('p', 'pʰ')},  # /p/ -> /pʰ/ after /s/
            {'context': ('n', '_'), 'change': ('k', 'ŋ')}    # /k/ -> /ŋ/ after /n/
        ]
        
        # 语音工作记忆
        self.phoneme_buffer = []
        self.buffer_size = self.params.get("buffer_size", 5)
    
    def process_phoneme(self, phoneme, context=None):
        """
        处理单个音素
        
        参数:
            phoneme (str): 输入音素
            context (tuple): 上下文音素 (前一个, 后一个)
            
        返回:
            处理后的音素
        """
        # 应用音位变体规则
        processed_phoneme = phoneme
        if context:
            for rule in self.allophonic_rules:
                if (rule['context'][0] == context[0] or rule['context'][0] == '_') and \
                   (rule['context'][1] == context[1] or rule['context'][1] == '_'):
                    if phoneme == rule['change'][0]:
                        processed_phoneme = rule['change'][1]
                        break
        
        # 更新音素缓冲区
        self.phoneme_buffer.append(processed_phoneme)
        if len(self.phoneme_buffer) > self.buffer_size:
            self.phoneme_buffer.pop(0)
        
        return processed_phoneme
    
    def get_phoneme_features(self, phoneme):
        """
        获取音素特征
        
        参数:
            phoneme (str): 音素
            
        返回:
            音素特征字典
        """
        return self.phoneme_features.get(phoneme, {})


class WordRecognizer:
    """词汇识别器"""
    
    def __init__(self, params=None):
        """
        初始化词汇识别器
        
        参数:
            params (dict): 配置参数
        """
        self.params = params or {}
        
        # 心理词典
        self.mental_lexicon = defaultdict(dict)
        
        # 词汇激活参数
        self.activation_decay = self.params.get("activation_decay", 0.1)
        self.activation_threshold = self.params.get("activation_threshold", 0.5)
        
        # 当前激活词汇
        self.active_words = {}
    
    def add_word(self, word, phonemes, frequency=1.0):
        """
        添加词汇到心理词典
        
        参数:
            word (str): 词汇
            phonemes (list): 音素序列
            frequency (float): 词频
        """
        self.mental_lexicon[word]['phonemes'] = phonemes
        self.mental_lexicon[word]['frequency'] = frequency
        self.mental_lexicon[word]['activation'] = 0.0
    
    def activate_word(self, word, amount=0.1):
        """
        激活词汇
        
        参数:
            word (str): 词汇
            amount (float): 激活量
        """
        if word in self.mental_lexicon:
            self.mental_lexicon[word]['activation'] = min(1.0, 
                self.mental_lexicon[word]['activation'] + amount)
            self.active_words[word] = self.mental_lexicon[word]['activation']
    
    def decay_activations(self):
        """衰减所有词汇激活"""
        for word in list(self.active_words.keys()):
            self.mental_lexicon[word]['activation'] = max(0.0, 
                self.mental_lexicon[word]['activation'] - self.activation_decay)
            
            if self.mental_lexicon[word]['activation'] < self.activation_threshold:
                del self.active_words[word]
    
    def recognize_word(self, phonemes):
        """
        识别词汇
        
        参数:
            phonemes (list): 音素序列
            
        返回:
            识别的词汇
        """
        best_word = None
        best_score = 0.0
        
        for word, info in self.mental_lexicon.items():
            # 计算音素序列匹配度
            match_score = self._match_phonemes(phonemes, info['phonemes'])
            
            # 综合词频和当前激活度
            score = match_score * info['frequency'] * (1 + info['activation'])
            
            if score > best_score:
                best_score = score
                best_word = word
        
        # 激活最佳匹配词汇
        if best_word and best_score > self.activation_threshold:
            self.activate_word(best_word, amount=0.3)
            return best_word
        
        return None
    
    def _match_phonemes(self, input_phonemes, word_phonemes):
        """
        计算音素序列匹配度
        
        参数:
            input_phonemes (list): 输入音素序列
            word_phonemes (list): 词汇音素序列
            
        返回:
            匹配度分数，0-1之间
        """
        if not input_phonemes or not word_phonemes:
            return 0.0
        
        # 简单匹配算法
        match_count = 0
        min_length = min(len(input_phonemes), len(word_phonemes))
        
        for i in range(min_length):
            if input_phonemes[i] == word_phonemes[i]:
                match_count += 1
        
        return match_count / max(len(input_phonemes), len(word_phonemes))


class SemanticNetwork:
    """语义网络"""
    
    def __init__(self, params=None):
        """
        初始化语义网络
        
        参数:
            params (dict): 配置参数
        """
        self.params = params or {}
        
        # 语义节点
        self.nodes = {}
        
        # 语义关系
        self.relations = defaultdict(dict)
        
        # 激活扩散参数
        self.activation_spread = self.params.get("activation_spread", 0.3)
        self.activation_decay = self.params.get("activation_decay", 0.05)
    
    def add_node(self, concept, attributes=None):
        """
        添加语义节点
        
        参数:
            concept (str): 概念
            attributes (dict): 概念属性
        """
        if attributes is None:
            attributes = {}
        
        self.nodes[concept] = {
            'activation': 0.0,
            'attributes': attributes
        }
    
    def add_relation(self, concept1, concept2, relation_type, strength=1.0):
        """
        添加语义关系
        
        参数:
            concept1 (str): 概念1
            concept2 (str): 概念2
            relation_type (str): 关系类型
            strength (float): 关系强度
        """
        if concept1 in self.nodes and concept2 in self.nodes:
            self.relations[concept1][concept2] = {
                'type': relation_type,
                'strength': strength
            }
    
    def activate_concept(self, concept, amount=0.5):
        """
        激活语义概念
        
        参数:
            concept (str): 概念
            amount (float): 激活量
        """
        if concept in self.nodes:
            self.nodes[concept]['activation'] = min(1.0, 
                self.nodes[concept]['activation'] + amount)
            
            # 激活扩散
            for related_concept, relation in self.relations[concept].items():
                spread_amount = amount * relation['strength'] * self.activation_spread
                self.activate_concept(related_concept, spread_amount)
    
    def decay_activations(self):
        """衰减所有激活"""
        for concept in self.nodes:
            self.nodes[concept]['activation'] = max(0.0, 
                self.nodes[concept]['activation'] - self.activation_decay)
    
    def get_related_concepts(self, concept, relation_type=None):
        """
        获取相关概念
        
        参数:
            concept (str): 概念
            relation_type (str): 关系类型 (可选)
            
        返回:
            相关概念列表
        """
        if concept not in self.relations:
            return []
        
        if relation_type:
            return [c for c, rel in self.relations[concept].items() 
                   if rel['type'] == relation_type]
        else:
            return list(self.relations[concept].keys())
    
    def get_most_activated(self, threshold=0.3):
        """
        获取激活度最高的概念
        
        参数:
            threshold (float): 激活阈值
            
        返回:
            激活概念列表
        """
        return [c for c, info in self.nodes.items() if info['activation'] >= threshold]


class SyntaxProcessor:
    """句法处理器"""
    
    def __init__(self, params=None):
        """
        初始化句法处理器
        
        参数:
            params (dict): 配置参数
        """
        self.params = params or {}
        
        # 句法规则
        self.rules = {
            'S': [['NP', 'VP']],
            'NP': [['Det', 'N'], ['NP', 'PP'], ['Pronoun']],
            'VP': [['V', 'NP'], ['VP', 'PP']],
            'PP': [['P', 'NP']]
        }
        
        # 词性标记
        self.pos_tags = {
            'Det': ['the', 'a', 'an'],
            'N': ['dog', 'cat', 'house', 'man', 'woman'],
            'V': ['chased', 'saw', 'ate', 'walked'],
            'P': ['in', 'on', 'at', 'with'],
            'Pronoun': ['he', 'she', 'it']
        }
        
        # 句法工作记忆
        self.syntax_buffer = []
        self.buffer_size = self.params.get("buffer_size", 7)
    
    def parse_sentence(self, words):
        """
        解析句子
        
        参数:
            words (list): 词汇列表
            
        返回:
            句法树
        """
        # 获取词性标记
        pos_tags = []
        for word in words:
            pos_tags.append(self._get_pos_tag(word))
        
        # 初始化句法树
        tree = {'type': 'S', 'children': []}
        
        # 简单的自顶向下解析
        self._parse_phrase(tree, pos_tags)
        
        return tree
    
    def _parse_phrase(self, node, pos_tags):
        """
        解析短语
        
        参数:
            node (dict): 当前节点
            pos_tags (list): 词性标记序列
        """
        node_type = node['type']
        
        # 尝试应用每条规则
        for expansion in self.rules.get(node_type, []):
            if len(expansion) <= len(pos_tags):
                matched = True
                children = []
                
                for i, child_type in enumerate(expansion):
                    if child_type in self.rules:
                        # 非终结符
                        child_node = {'type': child_type, 'children': []}
                        if self._parse_phrase(child_node, pos_tags[i:]):
                            children.append(child_node)
                            pos_tags = pos_tags[i + len(child_node['children']):]
                        else:
                            matched = False
                            break
                    else:
                        # 终结符
                        if pos_tags[i] == child_type:
                            children.append({'type': child_type, 'word': pos_tags[i]})
                        else:
                            matched = False
                            break
                
                if matched:
                    node['children'] = children
                    return True
        
        return False
    
    def _get_pos_tag(self, word):
        """
        获取词性标记
        
        参数:
            word (str): 词汇
            
        返回:
            词性标记
        """
        for pos, words in self.pos_tags.items():
            if word in words:
                return pos
        return 'UNK'


class LanguageGenerator:
    """语言生成器"""
    
    def __init__(self, params=None):
        """
        初始化语言生成器
        
        参数:
            params (dict): 配置参数
        """
        self.params = params or {}
        
        # 生成策略
        self.generation_strategy = self.params.get("strategy", "semantic")
        
        # 词汇选择参数
        self.word_selection_threshold = self.params.get("word_selection_threshold", 0.4)
        
        # 流利度参数
        self.fluency = self.params.get("fluency", 0.7)
    
    def generate_utterance(self, semantic_network, activated_concepts):
        """
        生成话语
        
        参数:
            semantic_network: 语义网络实例
            activated_concepts: 激活概念列表
            
        返回:
            生成的话语
        """
        if self.generation_strategy == "semantic":
            return self._generate_from_semantics(semantic_network, activated_concepts)
        else:
            return self._generate_from_syntax()
    
    def _generate_from_semantics(self, semantic_network, activated_concepts):
        """
        基于语义生成话语
        
        参数:
            semantic_network: 语义网络实例
            activated_concepts: 激活概念列表
            
        返回:
            生成的话语
        """
        if not activated_concepts:
            return ""
        
        # 选择核心概念
        core_concept = activated_concepts[0]
        
        # 获取相关概念
        related_agents = semantic_network.get_related_concepts(core_concept, "agent")
        related_actions = semantic_network.get_related_concepts(core_concept, "action")
        related_objects = semantic_network.get_related_concepts(core_concept, "object")
        
        # 构建简单句子
        if related_agents and related_actions:
            agent = np.random.choice(related_agents)
            action = np.random.choice(related_actions)
            
            if related_objects:
                obj = np.random.choice(related_objects)
                return f"{agent} {action} {obj}"
            else:
                return f"{agent} {action}"
        else:
            return core_concept
    
    def _generate_from_syntax(self):
        """
        基于句法生成话语
        
        返回:
            生成的话语
        """
        # 简单实现 - 实际应用中应使用更复杂的句法生成
        subjects = ["the dog", "the cat", "a man", "a woman"]
        verbs = ["chased", "saw", "ate", "walked"]
        objects = ["the ball", "a mouse", "the house", "a tree"]
        
        return f"{np.random.choice(subjects)} {np.random.choice(verbs)} {np.random.choice(objects)}"


class LanguageProcessor:
    """语言处理器"""
    
    def __init__(self, params=None):
        """
        初始化语言处理器
        
        参数:
            params (dict): 配置参数
        """
        self.params = params or {}
        
        # 语音处理模块
        self.phoneme_processor = PhonemeProcessor(self.params.get("phoneme", {}))
        
        # 词汇识别模块
        self.word_recognizer = WordRecognizer(self.params.get("word", {}))
        
        # 语义处理模块
        self.semantic_network = SemanticNetwork(self.params.get("semantic", {}))
        
        # 句法处理模块
        self.syntax_processor = SyntaxProcessor(self.params.get("syntax", {}))
        
        # 语言生成模块
        self.language_generator = LanguageGenerator(self.params.get("generation", {}))
        
        # 可视化状态
        self.visualization_data = {
            'phoneme_processing': [],
            'word_activation': {},
            'semantic_activation': {},
            'syntax_tree': None
        }
        
        # 初始化词典和语义网络
        self._initialize_lexicon()
        self._initialize_semantic_network()

    def get_visualization_data(self):
        """
        获取可视化数据
        
        返回:
            包含可视化数据的字典
        """
        return {
            'phonemes': self.visualization_data['phoneme_processing'],
            'words': self.visualization_data['word_activation'],
            'semantic': self.visualization_data['semantic_activation'],
            'syntax': self.visualization_data['syntax_tree'],
            'last_update': len(self.visualization_data['phoneme_processing'])
        }
    
    def _initialize_lexicon(self):
        """初始化心理词典"""
        # 添加一些基本词汇
        self.word_recognizer.add_word("dog", ['d', 'o', 'g'], frequency=0.8)
        self.word_recognizer.add_word("cat", ['k', 'a', 't'], frequency=0.7)
        self.word_recognizer.add_word("man", ['m', 'a', 'n'], frequency=0.6)
        self.word_recognizer.add_word("woman", ['w', 'u', 'm', 'a', 'n'], frequency=0.5)
        self.word_recognizer.add_word("chased", ['tʃ', 'e', 'i', 's', 't'], frequency=0.4)
        self.word_recognizer.add_word("saw", ['s', 'o'], frequency=0.4)
    
    def _initialize_semantic_network(self):
        """初始化语义网络"""
        # 添加概念
        self.semantic_network.add_node("dog", {"type": "animal", "size": "medium"})
        self.semantic_network.add_node("cat", {"type": "animal", "size": "small"})
        self.semantic_network.add_node("man", {"type": "human", "gender": "male"})
        self.semantic_network.add_node("woman", {"type": "human", "gender": "female"})
        self.semantic_network.add_node("chased", {"type": "action", "speed": "fast"})
        self.semantic_network.add_node("saw", {"type": "action", "speed": "instant"})
        
        # 添加关系
        self.semantic_network.add_relation("dog", "animal", "is-a", 1.0)
        self.semantic_network.add_relation("cat", "animal", "is-a", 1.0)
        self.semantic_network.add_relation("man", "human", "is-a", 1.0)
        self.semantic_network.add_relation("woman", "human", "is-a", 1.0)
        self.semantic_network.add_relation("dog", "chased", "can-do", 0.7)
        self.semantic_network.add_relation("cat", "chased", "can-do", 0.6)
        self.semantic_network.add_relation("man", "saw", "can-do", 0.8)
        self.semantic_network.add_relation("woman", "saw", "can-do", 0.8)
        self.semantic_network.add_relation("chased", "dog", "agent", 0.7)
        self.semantic_network.add_relation("chased", "cat", "agent", 0.6)
        self.semantic_network.add_relation("saw", "man", "agent", 0.8)
        self.semantic_network.add_relation("saw", "woman", "agent", 0.8)
    
    def process_speech(self, phonemes):
        """
        处理语音输入
        
        参数:
            phonemes (list): 音素序列
            
        返回:
            理解的意义和API状态
        """
        # 重置可视化数据
        self.visualization_data = {
            'phoneme_processing': [],
            'word_activation': {},
            'semantic_activation': {},
            'syntax_tree': None
        }
        
        # 处理音素并记录可视化数据
        processed_phonemes = []
        for i, phoneme in enumerate(phonemes):
            context = (
                phonemes[i-1] if i > 0 else None,
                phonemes[i+1] if i < len(phonemes)-1 else None
            )
            processed = self.phoneme_processor.process_phoneme(phoneme, context)
            processed_phonemes.append(processed)
            self.visualization_data['phoneme_processing'].append({
                'input': phoneme,
                'processed': processed,
                'context': context
            })
        
        # 识别词汇并记录激活状态
        words = []
        for i in range(len(processed_phonemes)):
            for j in range(i+1, min(i+5, len(processed_phonemes)+1)):
                phoneme_segment = processed_phonemes[i:j]
                word = self.word_recognizer.recognize_word(phoneme_segment)
                if word:
                    words.append(word)
                    # 激活语义概念
                    self.semantic_network.activate_concept(word)
                    # 记录词汇激活
                    self.visualization_data['word_activation'][word] = {
                        'phonemes': phoneme_segment,
                        'activation': self.word_recognizer.mental_lexicon[word]['activation']
                    }
                    break
        
        # 衰减激活
        self.word_recognizer.decay_activations()
        self.semantic_network.decay_activations()
        
        # 记录语义激活状态
        for concept, info in self.semantic_network.nodes.items():
            if info['activation'] > 0.1:
                self.visualization_data['semantic_activation'][concept] = {
                    'activation': info['activation'],
                    'attributes': info['attributes']
                }
        
        # 句法分析
        if words:
            syntax_tree = self.syntax_processor.parse_sentence(words)
            self.visualization_data['syntax_tree'] = syntax_tree
            
            return {
                'status': 'success',
                'words': words,
                'syntax_tree': syntax_tree,
                'meaning': self._extract_meaning(words),
                'visualization': self.get_visualization_data(),
                'timestamp': len(self.visualization_data['phoneme_processing'])
            }
        
        return {
            'status': 'no_words_recognized',
            'visualization': self.get_visualization_data(),
            'timestamp': len(self.visualization_data['phoneme_processing'])
        }
    
    def _extract_meaning(self, words):
        """
        提取意义
        
        参数:
            words (list): 词汇列表
            
        返回:
            意义表示
        """
        # 简单实现 - 实际应用中应使用更复杂的语义分析
        if len(words) >= 2:
            return {
                'agent': words[0],
                'action': words[1],
                'object': words[2] if len(words) > 2 else None
            }
        elif words:
            return {
                'concept': words[0]
            }
        else:
            return None
    
    def generate_speech(self, intention):
        """
        生成语音输出
        
        参数:
            intention (dict): 意图
            
        返回:
            生成的音素序列
        """
        # 激活相关概念
        if 'concept' in intention:
            self.semantic_network.activate_concept(intention['concept'], amount=0.7)
        
        # 获取激活概念
        activated_concepts = self.semantic_network.get_most_activated(threshold=0.3)
        
        # 生成语言
        utterance = self.language_generator.generate_utterance(self.semantic_network, activated_concepts)
        
        # 转换为音素序列
        if utterance:
            words = utterance.split()
            phonemes = []
            for word in words:
                if word in self.word_recognizer.mental_lexicon:
                    phonemes.extend(self.word_recognizer.mental_lexicon[word]['phonemes'])
                else:
                    # 简单处理未知词汇
                    phonemes.extend(list(word))
            
            return phonemes
        
        return []