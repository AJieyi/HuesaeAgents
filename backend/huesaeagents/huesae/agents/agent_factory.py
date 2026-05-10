"""Agent 工厂函数"""
from typing import Any
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from .graph import create_workflow


def create_huesae_agent(
    model: BaseChatModel | None = None,
    **kwargs
) -> Any:
    """
    创建 Huesae Agent 实例

    这是 Agent 的入口函数，用于创建可执行的 LangGraph Agent

    Args:
        model: 聊天模型实例，如果为 None 则使用默认的 create_chat_model 创建
        **kwargs: 其他参数，会传递给 create_workflow

    Returns:
        编译后的 LangGraph，可以调用 invoke() 或 stream()

    Example:
        >>> from huesaeagents.huesae.agents.agent_factory import create_huesae_agent
        >>> agent = create_huesae_agent()
        >>> result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
    """
    # 如果没有传入模型，使用默认的
    # （这里可以扩展为支持自定义模型）
    if model is None:
        from ..models.factory import create_chat_model
        model = create_chat_model("deepseek")

    # 创建工作流（已经编译过）
    workflow = create_workflow()

    # 返回编译后的工作流
    return workflow


def create_image_agent(
    model: BaseChatModel | None = None,
) -> Any:
    """
    创建带图片生成工具的 Agent

    使用 create_agent 模式，让 Agent 能够调用图片生成工具

    Args:
        model: 聊天模型实例，如果为 None 则使用默认的 create_chat_model 创建

    Returns:
        编译后的 Agent

    Example:
        >>> from huesaeagents.huesae.agents.agent_factory import create_image_agent
        >>> agent = create_image_agent()
        >>> result = agent.invoke({
        ...     "messages": [{"role": "user", "content": "画一个银发红瞳的少女"}]
        ... })
    """
    if model is None:
        from ..models.factory import create_chat_model
        model = create_chat_model("deepseek")

    from .utils.agent_tools import IMAGE_TOOLS

    # 使用 create_agent 创建 Agent
    agent = create_agent(model, tools=IMAGE_TOOLS)
    return agent