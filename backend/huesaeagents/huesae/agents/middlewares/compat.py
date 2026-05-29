"""Compatibility adapters for LangChain's native middleware runner."""

from __future__ import annotations

from typing import Any

from .base import AgentMiddleware


class LifecycleMiddlewareAdapter(AgentMiddleware):
    """Provide the state shape older Huesae middleware hooks expected."""

    def __init__(self, wrapped: AgentMiddleware):
        self.wrapped = wrapped

    @property
    def name(self) -> str:
        return f"HuesaeCompat{self.wrapped.name}"

    def before_agent(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
        hook = getattr(self.wrapped, "before_agent", None)
        if hook is None:
            return None
        return hook(dict(state), runtime)

    def before_model(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
        hook = getattr(self.wrapped, "before_model", None)
        if hook is None:
            return None
        compat_state = dict(state)
        compat_state.setdefault("step", 0)
        return hook(compat_state, runtime)

    def after_model(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
        hook = getattr(self.wrapped, "after_model", None)
        if hook is None:
            return None
        compat_state = dict(state)
        compat_state.setdefault("step", 0)
        return hook(compat_state, runtime)

    def after_agent(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
        hook = getattr(self.wrapped, "after_agent", None)
        if hook is None:
            return None
        compat_state = dict(state)
        if "result" not in compat_state:
            compat_state["result"] = {"messages": self._final_messages(compat_state)}
        return hook(compat_state, runtime)

    @staticmethod
    def _final_messages(state: dict) -> list:
        messages = state.get("messages") or []
        for message in reversed(messages):
            if getattr(message, "type", None) == "ai":
                return [message]
        return messages[-1:] if messages else []


def adapt_middlewares(middlewares: tuple[AgentMiddleware, ...]) -> list[AgentMiddleware]:
    """Wrap existing middleware objects for native LangChain execution."""

    return [LifecycleMiddlewareAdapter(middleware) for middleware in middlewares]


__all__ = ["LifecycleMiddlewareAdapter", "adapt_middlewares"]
