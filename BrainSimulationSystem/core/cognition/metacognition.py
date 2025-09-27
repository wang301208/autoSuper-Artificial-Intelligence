"""
元认知监控模块

实现对自己认知过程的认知和调节
"""

class MetaCognitiveMonitor:
    def __init__(self):
        # 校准参数
        self.confidence_calibration = 0.5  # 初始校准值(0-1)
        self.calibration_learning_rate = 0.1
        self.knowledge_gaps = set()  # 已识别的知识缺口
        
        # 监测指标
        self.monitoring_accuracy = 0.0
        self.control_efficiency = 0.0
        
    def evaluate_decision(self, decision, outcome):
        """评估决策质量并更新元认知参数
        
        Args:
            decision: 包含confidence属性的决策对象
            outcome: 包含accuracy属性的结果对象(0-1)
        """
        # 计算校准误差
        calibration_error = abs(decision.confidence - outcome.accuracy)
        
        # 动态调整学习率(误差大时调整幅度大)
        adaptive_lr = self.calibration_learning_rate * (1 + calibration_error)
        
        # 更新校准值(误差越小权重越低)
        self.confidence_calibration = (
            self.confidence_calibration * (1 - adaptive_lr) + 
            (1 - calibration_error) * adaptive_lr
        )
        
        # 记录知识缺口
        if outcome.accuracy < 0.5:
            self.knowledge_gaps.add(decision.domain)
            
        # 更新监测准确率(滑动平均)
        self.monitoring_accuracy = 0.9 * self.monitoring_accuracy + 0.1 * (1 - calibration_error)
        
    def get_control_suggestion(self):
        """生成认知控制建议"""
        if self.confidence_calibration < 0.3:
            return "需要更多验证性学习"
        elif len(self.knowledge_gaps) > 3:
            return "建议系统性知识补充"
        else:
            return "当前认知策略有效"
            
    def reset(self):
        """重置临时状态(保留学习到的参数)"""
        self.knowledge_gaps = set()