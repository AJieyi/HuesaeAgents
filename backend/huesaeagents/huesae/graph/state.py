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

    # ---- 生图功能 ----
    image_step: str | None             # 当前步骤
    image_mode: str | None             # 当前模式（direct_image/convert_tags/expand_prompt）
    image_prompt: str | None           # 当前提示词
    selected_provider: str | None      # 选择的Provider
    danbooru_tags: list[str] | None    # 生成的Danbooru标签
    expanded_prompt: str | None        # 扩写后的提示词
    generated_image_url: str | None    # 生成的图片URL
    need_more_input: bool              # 是否需要补充输入

    # ---- 元信息 ----
    user_id: str | None
    thread_id: str | None

    # ---- 安全 ----
    safety_flag: bool
    high_risk_flag: bool
