"""AutoGPT 代理基类定义。

本模块定义了所有 AutoGPT 代理的抽象基类，规定了代理必须实现的核心接口。
代理是 AutoGPT 系统的核心组件，负责执行任务、做出决策和管理工作流程。
"""

import abc
import logging
from pathlib import Path


class Agent(abc.ABC):
    """AutoGPT 代理的抽象基类。
    
    所有具体的代理实现都必须继承此类并实现其抽象方法。
    代理负责自主执行任务、决策制定和能力调用。
    
    核心职责:
        - 任务规划和执行
        - 能力选择和调用
        - 状态管理和持久化
        - 与外部系统交互
    
    设计模式:
        使用抽象基类确保所有代理实现都遵循统一的接口规范，
        便于系统的扩展和维护。
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """初始化代理实例。
        
        参数:
            *args: 位置参数，具体参数由子类定义
            **kwargs: 关键字参数，具体参数由子类定义
            
        注意:
            子类必须实现此方法来完成代理的初始化工作，
            包括配置加载、状态初始化等。
        """
        ...

    @classmethod
    @abc.abstractmethod
    def from_workspace(
        cls,
        workspace_path: Path,
        logger: logging.Logger,
    ) -> "Agent":
        """从工作空间创建代理实例。
        
        这是一个工厂方法，用于从指定的工作空间路径创建代理实例。
        工作空间包含代理的配置、状态和相关文件。
        
        参数:
            workspace_path: 工作空间目录路径
            logger: 日志记录器实例
            
        返回:
            Agent: 创建的代理实例
            
        异常:
            FileNotFoundError: 工作空间路径不存在
            ValueError: 工作空间配置无效
            
        注意:
            子类必须实现此方法来支持从工作空间恢复代理状态。
        """
        ...

    @abc.abstractmethod
    async def determine_next_ability(self, *args, **kwargs):
        """确定下一个要执行的能力。
        
        这是代理的核心决策方法，负责分析当前状态和目标，
        决定下一步应该执行哪个能力来推进任务完成。
        
        参数:
            *args: 位置参数，具体参数由子类定义
            **kwargs: 关键字参数，具体参数由子类定义
            
        返回:
            具体返回类型由子类定义，通常包含能力信息和参数
            
        注意:
            这是一个异步方法，因为决策过程可能涉及网络请求、
            文件 I/O 或其他耗时操作。
        """
        ...

    @abc.abstractmethod
    def __repr__(self):
        """返回代理的字符串表示。
        
        返回:
            str: 代理的描述性字符串，用于调试和日志记录
            
        注意:
            应该包含代理的关键信息，如类型、状态、配置等。
        """
        ...
