"""Subagent execution boundary.

This mirrors DeerFlow's task delegation shape: the lead agent asks for a
subagent through ``task_tool``, while a dedicated executor owns lookup,
thread-id derivation, state sanitization, invocation, and result normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.messages import AIMessage, HumanMessage

from .registry import SubAgentRegistry


@dataclass(frozen=True)
class SubagentExecution:
    """Normalized subagent execution output."""

    message: AIMessage
    context: dict[str, Any] | None
    artifacts: list[dict[str, Any]]
    completed: bool
    error: bool = False

    def to_result(self) -> dict:
        """Return the public response shape used by the lead agent."""
        result = {
            "messages": [self.message],
            "artifacts": self.artifacts,
        }
        if self.context:
            result["current_subagent"] = self.context
        return result


class SubagentExecutor:
    """Run registered subagents in their own graph/checkpoint context."""

    def __init__(self, registry: SubAgentRegistry):
        self.registry = registry

    def start(
        self,
        subagent_type: str,
        description: str,
        *,
        parent_thread_id: str,
    ) -> SubagentExecution:
        """Start a delegated subagent task."""
        agent = self.registry.get(subagent_type)
        if not agent:
            available = self.registry.names()
            return self._error(f"抱歉，暂时没有处理这种任务的子Agent~ 可用的子Agent：{available}")

        sub_thread_id = self.subagent_thread_id(parent_thread_id, subagent_type)
        sub_state = self.initial_state(subagent_type)
        sub_result = agent.invoke(description, thread_id=sub_thread_id, state=sub_state)
        context = {
            "agent_type": subagent_type,
            "thread_id": sub_thread_id,
            "state": sub_state,
            "history": [
                HumanMessage(content=description),
                AIMessage(content=sub_result.get("response", "")),
            ],
        }
        self.apply_state_update(context, sub_result)
        return self.finalize(sub_result, context)

    def resume(
        self,
        user_input: str,
        *,
        parent_thread_id: str,
        context: dict,
    ) -> SubagentExecution:
        """Continue an interactive delegated subagent task."""
        context = self.sanitize_context(context)
        agent_type = context.get("agent_type")
        agent = self.registry.get(agent_type)
        if not agent:
            return self._error("子Agent状态异常，请重新开始~", completed=True)

        sub_thread_id = context.get("thread_id") or self.subagent_thread_id(parent_thread_id, str(agent_type))
        sub_state = self.sanitize_state(context.get("state", {}))
        history = list(context.get("history", []))
        sub_state["messages"] = history

        sub_result = agent.invoke(user_input, thread_id=sub_thread_id, state=sub_state)
        self.apply_state_update(context, sub_result)
        sub_state = self.sanitize_state(context.get("state", sub_state))

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=sub_result.get("response", "")))
        sub_state["messages"] = history
        context.update(
            {
                "agent_type": agent_type,
                "thread_id": sub_thread_id,
                "state": sub_state,
                "history": history,
            }
        )
        return self.finalize(sub_result, context)

    def finalize(self, sub_result: dict, context: dict) -> SubagentExecution:
        """Normalize a raw subagent result for lead-agent state updates."""
        response = str(sub_result.get("response") or "")
        message = AIMessage(content=response or "任务已经完成。")
        action = sub_result.get("action", "")
        agent_type = context.get("agent_type", "")
        artifacts = list((sub_result.get("data") or {}).get("artifacts") or [])
        completed = agent_type == "general" or action == "finish"
        next_context = None if completed else self.sanitize_context(context)
        return SubagentExecution(
            message=message,
            context=next_context,
            artifacts=artifacts,
            completed=completed,
        )

    @staticmethod
    def initial_state(subagent_type: str) -> dict:
        """Return a serializable initial subagent state."""
        if subagent_type == "general":
            return {"messages": []}
        return {
            "messages": [],
            "image_task_type": "generate_image",
            "image_phase": "collecting_prompt",
        }

    @staticmethod
    def subagent_thread_id(parent_thread_id: str, subagent_type: str) -> str:
        """Derive an isolated subagent checkpoint id."""
        return f"{parent_thread_id}:{subagent_type}"

    @classmethod
    def apply_state_update(cls, subagent_context: dict, sub_result: dict) -> None:
        """Apply a subagent state patch while keeping checkpoint values serializable."""
        state_update = (sub_result.get("data") or {}).get("state_update") or {}
        if not state_update:
            return
        state = subagent_context.setdefault("state", {})
        state.update(state_update)
        subagent_context["state"] = cls.sanitize_state(state)

    @classmethod
    def sanitize_context(cls, context: dict | None) -> dict:
        """Remove runtime-only objects from a persisted subagent context."""
        sanitized = dict(context or {})
        sanitized.pop("agent", None)
        sanitized["state"] = cls.sanitize_state(sanitized.get("state", {}))
        return sanitized

    @staticmethod
    def sanitize_state(state: dict | None) -> dict:
        """Remove runtime-only objects from subagent-local state."""
        sanitized = dict(state or {})
        sanitized.pop("agent", None)
        sanitized.pop("runtime", None)
        sanitized.pop("skill_registry", None)
        return sanitized

    @staticmethod
    def _error(message: str, *, completed: bool = True) -> SubagentExecution:
        return SubagentExecution(
            message=AIMessage(content=message),
            context=None,
            artifacts=[],
            completed=completed,
            error=True,
        )


__all__ = ["SubagentExecution", "SubagentExecutor"]
