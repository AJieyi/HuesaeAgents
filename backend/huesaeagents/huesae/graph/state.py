"""LangGraph 统一状态定义

所有工作流节点共享此状态，通过 Annotated + reducer 实现状态合并。
"""
from typing import TypedDict, Annotated
from langgraph.graph import add_messages


class HuesaeState(TypedDict):
    """HuesaeAgents 主工作流统一状态

    包含对话、意图路由、角色情绪、生图功能等所有状态字段。
    通过 add_messages reducer 自动合并消息列表。
    """

    # ---- 对话 ----
    messages: Annotated[list, add_messages]

    # ---- 意图路由 ----
    intent: str | None

    # ---- 角色与情绪 ----
    character_id: str
    emotion_state: str
    emotion_score: float

    # ---- 生图功能（LLM驱动对话管理） ----
    image_step: str | None             # 当前步骤：ask_prompt/recommend/ask_confirm/generate/show_image/finish
    image_prompt: str | None           # 当前确认的提示词
    selected_provider: str | None      # 选择的Provider
    generated_image_url: str | None    # 生成的图片URL

    # ---- 元信息 ----
    user_id: str | None
    thread_id: str | None

    # ---- 安全 ----
    safety_flag: bool
    high_risk_flag: bool
