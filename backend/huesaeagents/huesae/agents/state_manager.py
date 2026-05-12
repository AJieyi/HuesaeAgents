"""对话状态管理器

管理单会话的状态，包括对话历史、image_goal 等元数据。
支持内存和文件持久化两种模式。

Example:
    >>> from state_manager import StateManager
    >>> sm = StateManager(persist_path="./conversations")
    >>> state = sm.get_state("user_001")
    >>> state.messages.append(HumanMessage(content="你好"))
    >>> sm.save_state("user_001")
"""
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    message_to_dict,
    messages_from_dict,
)


class ConversationState:
    """单会话的状态对象

    Attributes:
        messages: 对话历史（LangChain Message 列表）
        image_goal: 当前图像相关意图（generate_image/expand_prompt/convert_tags）
        metadata: 其他扩展字段
    """

    def __init__(self):
        self.messages: list = []
        self.image_goal: str | None = None
        self.metadata: dict[str, Any] = {}

    def to_dict(self) -> dict:
        """序列化为字典（用于 JSON 持久化）"""
        return {
            "messages": [message_to_dict(m) for m in self.messages],
            "image_goal": self.image_goal,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationState":
        """从字典恢复状态"""
        state = cls()
        state.messages = messages_from_dict(data.get("messages", []))
        state.image_goal = data.get("image_goal")
        state.metadata = data.get("metadata", {})
        return state


class StateManager:
    """状态管理器

    管理多个会话的状态，支持内存和文件持久化。

    Args:
        persist_path: 持久化目录路径，为 None 时仅内存存储

    Example:
        >>> sm = StateManager(persist_path="./conversations")
        >>> state = sm.get_state("terminal_user")
        >>> state.image_goal = "generate_image"
        >>> sm.save_state("terminal_user")
    """

    def __init__(self, persist_path: str | None = None):
        self.persist_path = Path(persist_path) if persist_path else None
        self._states: dict[str, ConversationState] = {}

    def get_state(self, session_id: str = "default") -> ConversationState:
        """获取指定会话的状态

        如果状态不存在，先尝试从文件加载，否则创建新的。
        """
        if session_id not in self._states:
            # 尝试从文件加载
            loaded = self._load_from_file(session_id)
            self._states[session_id] = loaded if loaded else ConversationState()

        return self._states[session_id]

    def save_state(self, session_id: str = "default") -> None:
        """保存指定会话的状态到文件"""
        if not self.persist_path:
            return

        state = self._states.get(session_id)
        if not state:
            return

        self.persist_path.mkdir(parents=True, exist_ok=True)
        file_path = self.persist_path / f"{session_id}.json"
        file_path.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear_state(self, session_id: str = "default") -> None:
        """清除指定会话的状态（内存 + 文件）"""
        self._states.pop(session_id, None)

        if self.persist_path:
            file_path = self.persist_path / f"{session_id}.json"
            if file_path.exists():
                file_path.unlink()

    def _load_from_file(self, session_id: str) -> ConversationState | None:
        """从文件加载会话状态"""
        if not self.persist_path:
            return None

        file_path = self.persist_path / f"{session_id}.json"
        if not file_path.exists():
            return None

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            return ConversationState.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    # ============== 便捷方法 ==============

    def update_image_goal(
        self, session_id: str, image_goal: str | None
    ) -> None:
        """更新指定会话的 image_goal"""
        state = self.get_state(session_id)
        state.image_goal = image_goal

    def get_image_goal(self, session_id: str = "default") -> str | None:
        """获取指定会话的 image_goal"""
        return self.get_state(session_id).image_goal

    def add_message(
        self, session_id: str, message: HumanMessage | AIMessage | SystemMessage
    ) -> None:
        """向指定会话添加一条消息"""
        state = self.get_state(session_id)
        state.messages.append(message)
