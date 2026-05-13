"""对话状态管理器（内存版）。

管理单会话的状态，包括主对话历史和当前活跃子Agent。
仅内存存储，不持久化到文件。

Example:
    >>> from state_manager import StateManager
    >>> sm = StateManager()
    >>> state = sm.get_state("user_001")
    >>> state.messages.append(HumanMessage(content="你好"))
"""
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from .state.huesae_state import HuesaeState


class StateManager:
    """状态管理器（仅内存）

    管理多个会话的状态，仅内存存储，退出后状态丢失。

    Example:
        >>> sm = StateManager()
        >>> state = sm.get_state("terminal_user")
        >>> state.messages
        []
    """

    def __init__(self):
        self._states: dict[str, HuesaeState] = {}

    def get_state(self, session_id: str = "default") -> HuesaeState:
        """获取指定会话的状态

        如果状态不存在，创建新的。
        """
        if session_id not in self._states:
            self._states[session_id] = HuesaeState()

        return self._states[session_id]

    def save_state(self, session_id: str = "default") -> None:
        """保存指定会话的状态（当前为内存存储，此方法为空操作）"""
        pass

    def clear_state(self, session_id: str = "default") -> None:
        """清除指定会话的状态（仅内存）"""
        self._states.pop(session_id, None)

    # ============== 便捷方法 ==============

    def add_message(
        self, session_id: str, message: HumanMessage | AIMessage | SystemMessage
    ) -> None:
        """向指定会话添加一条消息"""
        state = self.get_state(session_id)
        state.messages.append(message)
