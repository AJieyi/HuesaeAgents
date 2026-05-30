"""Agents 模块 - Agent核心系统

LangChain + 主智能体委派架构。
"""
from .lead_agent import HuesaeMainAgent, run_chat_loop
from ..subagents.base import BaseSubAgent
from ..subagents.general_agent import GeneralSubAgent, create_general_agent
from ..subagents.image_agent import ImageSubAgent, ImageDecision, create_image_agent
from ..subagents.registry import SubAgentInfo, SubAgentRegistry

__all__ = [
    "HuesaeMainAgent",
    "run_chat_loop",
    "BaseSubAgent",
    "GeneralSubAgent",
    "ImageSubAgent",
    "ImageDecision",
    "create_general_agent",
    "create_image_agent",
    "SubAgentInfo",
    "SubAgentRegistry",
]
