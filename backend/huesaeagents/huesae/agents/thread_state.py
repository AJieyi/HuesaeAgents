"""LangGraph state schema for Huesae agents."""

from __future__ import annotations

from typing import Any

from langchain.agents import AgentState
from typing_extensions import NotRequired


class HuesaeThreadState(AgentState):
    """Global thread state used by LangChain's LangGraph-backed agents."""

    user_input: NotRequired[str]
    current_subagent: NotRequired[dict[str, Any] | None]
    vision_context: NotRequired[dict[str, Any]]
    artifacts: NotRequired[list[dict[str, Any]]]
    step: NotRequired[int]
    result: NotRequired[dict[str, Any]]


__all__ = ["HuesaeThreadState"]
