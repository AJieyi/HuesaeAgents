"""Huesae 对话状态类

显式记录当前意图，替代从历史消息推断状态的方式。

Example:
    >>> from state.huesae_state import HuesaeState
    >>> state = HuesaeState()
    >>> state.messages.append(HumanMessage(content="你好"))
    >>> state.intent = "chat"
    >>> state.image_intent = "generate_image"
    >>> data = state.to_dict()
    >>> restored = HuesaeState.from_dict(data)
"""
from typing import Any


class HuesaeState:
    """Huesae 对话状态

    Attributes:
        messages: 主对话历史（只保留核心业务消息）
        image_context: 生图Agent子对话历史（隔离中间追问/扩写等过程）
        intent: 当前主意图（chat/image/voice/memory/search/remind）
        image_intent: image 意图下的子分类（generate_image/expand_prompt/convert_tags）
    """

    def __init__(self):
        self.messages: list = []
        self.image_context: list = []  # 生图Agent子上下文
        self.intent: str | None = None
        self.image_intent: str | None = None
        self.current_image_prompt: str | None = None  # 当前确认的提示词（用于换图）

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "messages": self.messages,
            "image_context": self.image_context,
            "intent": self.intent,
            "image_intent": self.image_intent,
            "current_image_prompt": self.current_image_prompt,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HuesaeState":
        """从字典恢复状态"""
        state = cls()
        state.messages = data.get("messages", [])
        state.image_context = data.get("image_context", [])
        state.intent = data.get("intent")
        state.image_intent = data.get("image_intent")
        state.current_image_prompt = data.get("current_image_prompt")
        return state

    def clear_image(self) -> None:
        """清除 image 相关状态（子Agent完成时调用）"""
        self.image_intent = None
        self.image_context = []
        self.current_image_prompt = None
