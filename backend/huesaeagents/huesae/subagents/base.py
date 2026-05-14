"""子Agent基类

定义所有子Agent的统一接口，主Agent通过此接口调用子Agent。
"""
from abc import ABC, abstractmethod


class BaseSubAgent(ABC):
    """子Agent基类

    所有子Agent必须实现 process 方法，返回标准化结果。
    子Agent本身是无状态的，每次调用接收完整对话历史做决策。
    """

    @abstractmethod
    def process(self, state: dict, user_input: str) -> dict:
        """处理用户输入，返回标准化结果

        Args:
            state: 当前状态，包含 messages 等
            user_input: 用户最新输入

        Returns:
            dict: 标准化结果，必须包含以下字段：
                - action: str - 动作类型
                - response: str - 给用户的回复（供主Agent包装或直接展示）
                - prompt: str | None - 确认的提示词（generate时）
                - provider: str | None - 选择的生图工具
                - data: dict - 额外数据（扩展用）
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """子Agent名称，用于注册和识别"""
        raise NotImplementedError
