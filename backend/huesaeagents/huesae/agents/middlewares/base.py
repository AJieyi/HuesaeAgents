"""中间件基础类型。

优先复用 LangChain 1.x 的 AgentMiddleware；当上游包版本缺少该类时，
提供一个兼容的本地基类，保证项目内部导入路径稳定。
"""

from __future__ import annotations

from typing import Any


try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError:
    class AgentMiddleware:
        """兼容 LangChain AgentMiddleware 生命周期钩子的本地基类。"""

        def before_agent(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
            return None

        def before_model(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
            return None

        def after_model(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
            return None

        def after_agent(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
            return None

        async def abefore_agent(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
            return self.before_agent(state, runtime)

        async def abefore_model(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
            return self.before_model(state, runtime)

        async def aafter_model(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
            return self.after_model(state, runtime)

        async def aafter_agent(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
            return self.after_agent(state, runtime)


__all__ = ["AgentMiddleware"]
