"""中间件执行管道。

HuesaeAgents 当前使用自定义 ReAct 循环，因此需要显式调度 LangChain
AgentMiddleware 的生命周期钩子。
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any

from .base import AgentMiddleware


class MiddlewarePipeline:
    """按顺序执行 AgentMiddleware 钩子的轻量管道。"""

    def __init__(
        self,
        middlewares: Iterable[AgentMiddleware] | None = None,
        runtime: Any = None,
    ):
        self._middlewares: list[AgentMiddleware] = list(middlewares or [])
        self.runtime = runtime

    @property
    def middlewares(self) -> tuple[AgentMiddleware, ...]:
        """返回已注册中间件的只读视图。"""
        return tuple(self._middlewares)

    def add(self, middleware: AgentMiddleware) -> None:
        """追加一个中间件。"""
        self._middlewares.append(middleware)

    def run_before_agent(self, state: dict) -> dict:
        """Agent 循环开始前调用一次。"""
        return self._run_hook("before_agent", state)

    def run_before_model(self, state: dict) -> dict:
        """每次模型调用前调用。"""
        return self._run_hook("before_model", state)

    def run_after_model(self, state: dict) -> dict:
        """每次模型响应后调用。"""
        return self._run_hook("after_model", state)

    def run_after_agent(self, state: dict) -> dict:
        """Agent 循环结束后调用一次。"""
        return self._run_hook("after_agent", state)

    async def arun_before_agent(self, state: dict) -> dict:
        """异步 Agent 循环开始前调用一次。"""
        return await self._arun_hook("abefore_agent", "before_agent", state)

    async def arun_before_model(self, state: dict) -> dict:
        """异步模型调用前调用。"""
        return await self._arun_hook("abefore_model", "before_model", state)

    async def arun_after_model(self, state: dict) -> dict:
        """异步模型响应后调用。"""
        return await self._arun_hook("aafter_model", "after_model", state)

    async def arun_after_agent(self, state: dict) -> dict:
        """异步 Agent 循环结束后调用一次。"""
        return await self._arun_hook("aafter_agent", "after_agent", state)

    def _run_hook(self, hook_name: str, state: dict) -> dict:
        current_state = dict(state)
        for middleware in self._middlewares:
            hook = getattr(middleware, hook_name, None)
            if hook is None:
                continue
            update = hook(current_state, self.runtime)
            current_state = self._merge_state(current_state, update)
        return current_state

    async def _arun_hook(self, async_hook_name: str, sync_hook_name: str, state: dict) -> dict:
        current_state = dict(state)
        for middleware in self._middlewares:
            hook = getattr(middleware, async_hook_name, None) or getattr(middleware, sync_hook_name, None)
            if hook is None:
                continue
            update = hook(current_state, self.runtime)
            if inspect.isawaitable(update):
                update = await update
            current_state = self._merge_state(current_state, update)
        return current_state

    @staticmethod
    def _merge_state(state: dict, update: dict | None) -> dict:
        """合并中间件返回值；messages 追加，其他字段覆盖。"""
        if not update:
            return state
        if not isinstance(update, dict):
            return state

        merged = dict(state)
        for key, value in update.items():
            if key == "messages":
                current_messages = merged.get("messages") or []
                next_messages = value or []
                if next_messages is current_messages:
                    merged["messages"] = current_messages
                    continue
                if not isinstance(next_messages, list):
                    next_messages = [next_messages]
                if (
                    isinstance(current_messages, list)
                    and len(next_messages) >= len(current_messages)
                    and next_messages[: len(current_messages)] == current_messages
                ):
                    merged["messages"] = next_messages
                    continue
                merged["messages"] = list(current_messages) + list(next_messages)
                continue
            merged[key] = value
        return merged


__all__ = ["MiddlewarePipeline"]
