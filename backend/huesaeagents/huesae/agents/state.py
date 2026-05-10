"""Agent 状态定义"""
from typing import TypedDict, Annotated
from langgraph.graph import add_messages


class ThreadState(TypedDict):
    """
    Agent 线程状态定义

    用于在 LangGraph 工作流中传递状态

    Attributes:
        messages: 对话消息列表，用于存储用户消息和AI回复
        character_id: 当前角色ID，用于标识当前使用的角色
        emotion_state: 情绪状态（开心/难过/害羞/寂寞/愤怒等）
        emotion_score: 情绪强度 0-1
        user_id: 用户ID，用于标识用户身份
        thread_id: 线程ID，用于区分不同的对话会话
    """

    messages: Annotated[list, add_messages]  # 对话消息列表，支持消息合并
    character_id: str | None  # 当前角色ID
    emotion_state: str | None  # 情绪状态
    emotion_score: float | None  # 情绪强度 0-1
    user_id: str | None  # 用户ID
    thread_id: str | None  # 线程ID
