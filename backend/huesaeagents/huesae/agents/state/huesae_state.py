"""Huesae 对话状态类。

仅保存当前运行时需要的主对话历史和活跃子Agent上下文。
"""


class HuesaeState:
    """Huesae 对话状态。

    Attributes:
        messages: 主对话历史。
        active_subagent: 当前活跃的子Agent上下文。
    """

    def __init__(self):
        self.messages: list = []
        self.active_subagent: dict | None = None

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "messages": self.messages,
            "active_subagent": self.active_subagent,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HuesaeState":
        """从字典恢复状态。"""
        state = cls()
        state.messages = data.get("messages", [])
        state.active_subagent = data.get("active_subagent")
        return state

    def clear_subagent(self) -> None:
        """清除当前活跃子Agent。"""
        self.active_subagent = None
