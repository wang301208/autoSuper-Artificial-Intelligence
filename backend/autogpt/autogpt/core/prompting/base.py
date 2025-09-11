import abc

from autogpt.core.configuration import SystemConfiguration
from autogpt.core.resource.model_providers import AssistantChatMessage

from .schema import ChatPrompt, LanguageModelClassification


class PromptStrategy(abc.ABC):
    """提示策略的抽象基类。
    
    该类定义了AutoGPT系统中不同提示策略的接口。每个策略封装了
    构建提示、分类语言模型和解析响应的逻辑。
    
    属性:
        default_configuration: 该策略的默认系统配置
    """
=======
    default_configuration: SystemConfiguration

    @property
    @abc.abstractmethod
    def model_classification(self) -> LanguageModelClassification:
        """获取该策略的语言模型分类。
        
        该属性定义了此提示策略正常运行所需的语言模型类型。
        
        返回:
            LanguageModelClassification: 所需模型的分类
        """
=======
        ...

    @abc.abstractmethod
    def build_prompt(self, *_, **kwargs) -> ChatPrompt:
        """基于给定参数构建聊天提示。
        
        该方法构造一个可以发送给语言模型的结构化提示。
        具体实现取决于策略的目的和要求。
        
        参数:
            *_: 位置参数（策略特定）
            **kwargs: 关键字参数（策略特定）
            
        返回:
            ChatPrompt: 准备发送给语言模型的结构化提示
            
        异常:
            NotImplementedError: 如果子类没有实现此方法
        """
=======
        ...

    @abc.abstractmethod
    def parse_response_content(self, response_content: AssistantChatMessage):
        """解析来自语言模型的响应内容。
        
        该方法处理来自语言模型的原始响应，并根据策略的
        预期格式提取相关信息。
        
        参数:
            response_content: 来自助手的响应消息
            
        返回:
            解析后的内容（类型取决于具体策略）
            
        异常:
            NotImplementedError: 如果子类没有实现此方法
        """
=======
        ...
=======
