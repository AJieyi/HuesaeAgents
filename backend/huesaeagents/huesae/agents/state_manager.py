"""对话状态管理器

管理单会话的状态，包括对话历史、image_intent 等元数据。
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

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    message_to_dict,
    messages_from_dict,
)

from .state.huesae_state import HuesaeState


class StateManager:
    """状态管理器

    管理多个会话的状态，支持内存和文件持久化。

    Args:
        persist_path: 持久化目录路径，为 None 时仅内存存储

    Example:
        >>> sm = StateManager(persist_path="./conversations")
        >>> state = sm.get_state("terminal_user")
        >>> state.image_intent = "generate_image"
        >>> sm.save_state("terminal_user")
    """

    def __init__(self, persist_path: str | None = None):
        self.persist_path = Path(persist_path) if persist_path else None
        self._states: dict[str, HuesaeState] = {}

    def get_state(self, session_id: str = "default") -> HuesaeState:
        """获取指定会话的状态

        如果状态不存在，先尝试从文件加载，否则创建新的。
        """
        if session_id not in self._states:
            loaded = self._load_from_file(session_id)
            self._states[session_id] = loaded if loaded else HuesaeState()

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

        # 序列化消息
        data = {
            "messages": [message_to_dict(m) for m in state.messages],
            "image_context": [message_to_dict(m) for m in state.image_context],
            "intent": state.intent,
            "image_intent": state.image_intent,
            "current_image_prompt": state.current_image_prompt,
        }
        file_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear_state(self, session_id: str = "default") -> None:
        """清除指定会话的状态（内存 + 文件）"""
        self._states.pop(session_id, None)

        if self.persist_path:
            file_path = self.persist_path / f"{session_id}.json"
            if file_path.exists():
                file_path.unlink()

    def _load_from_file(self, session_id: str) -> HuesaeState | None:
        """从文件加载会话状态"""
        if not self.persist_path:
            return None

        file_path = self.persist_path / f"{session_id}.json"
        if not file_path.exists():
            return None

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            state = HuesaeState()
            state.messages = messages_from_dict(data.get("messages", []))
            state.image_context = messages_from_dict(data.get("image_context", []))
            state.intent = data.get("intent")
            state.image_intent = data.get("image_intent")
            state.current_image_prompt = data.get("current_image_prompt")
            return state
        except (json.JSONDecodeError, KeyError):
            return None

    # ============== 便捷方法 ==============

    def update_image_intent(
        self, session_id: str, image_intent: str | None
    ) -> None:
        """更新指定会话的 image_intent"""
        state = self.get_state(session_id)
        state.image_intent = image_intent

    def get_image_intent(self, session_id: str = "default") -> str | None:
        """获取指定会话的 image_intent"""
        return self.get_state(session_id).image_intent

    def add_message(
        self, session_id: str, message: HumanMessage | AIMessage | SystemMessage
    ) -> None:
        """向指定会话添加一条消息"""
        state = self.get_state(session_id)
        state.messages.append(message)
