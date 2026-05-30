"""Runtime tool middleware for dynamic Huesae tool views."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langchain.messages import ToolMessage
from langchain.agents.middleware import ModelRequest
from langgraph.types import Command

from .base import AgentMiddleware


LOAD_MCP_TOOLS_SIGNAL = "__LOAD_MCP_TOOLS__"
MCP_TOOLS_LOADED_MESSAGE = (
    "MCP扩展工具已加载。请结合用户原始需求，根据更新后的工具列表重新选择最合适的具体工具，"
    "并严格使用工具 schema 中的参数名。"
)
MCP_NO_TOOLS_MESSAGE = "当前：MCP扩展工具已加载，但当前没有可用的 MCP 工具。"


class RuntimeToolMiddleware(AgentMiddleware):
    """Expose the current SharedToolRuntime view to a compiled agent."""

    def __init__(
        self,
        *,
        tools: Callable[[], list],
        tool_map: Callable[[], dict],
        system_message: Callable[[str | None], Any],
        ensure_mcp_tools: Callable[[], None],
        is_mcp_loaded: Callable[[], bool],
        has_mcp_tools: Callable[[], bool] | None = None,
        subagent_executor: Any | None = None,
    ):
        self._tools = tools
        self._tool_map = tool_map
        self._system_message = system_message
        self._ensure_mcp_tools = ensure_mcp_tools
        self._is_mcp_loaded = is_mcp_loaded
        self._has_mcp_tools = has_mcp_tools or (lambda: True)
        self._subagent_executor = subagent_executor

    def wrap_model_call(self, request: ModelRequest, handler):
        user_input = request.state.get("user_input") if isinstance(request.state, dict) else None
        return handler(
            request.override(
                tools=self._tools(),
                system_message=self._system_message(user_input),
            )
        )

    def wrap_tool_call(self, request, handler):
        tool_name = request.tool_call.get("name") or ""
        if tool_name == "task_tool" and self._subagent_executor is not None:
            return self._run_subagent_task(request)

        tool = self._tool_map().get(tool_name) or request.tool
        if tool is not None:
            result = handler(request.override(tool=tool))
            if isinstance(result, ToolMessage) and str(result.content) == LOAD_MCP_TOOLS_SIGNAL:
                if not self._is_mcp_loaded():
                    self._ensure_mcp_tools()
                content = MCP_TOOLS_LOADED_MESSAGE if self._has_mcp_tools() else MCP_NO_TOOLS_MESSAGE
                return self._tool_message(request, content)
            return result

        if not self._is_mcp_loaded():
            self._ensure_mcp_tools()
            content = MCP_TOOLS_LOADED_MESSAGE if self._has_mcp_tools() else MCP_NO_TOOLS_MESSAGE
            return self._tool_message(request, content)

        available = list(self._tool_map().keys())
        return self._tool_message(request, f"错误：未知工具 {tool_name}。可用工具：{available}", status="error")

    def _run_subagent_task(self, request) -> Command:
        args = request.tool_call.get("args") or {}
        description = str(args.get("description") or "").strip()
        subagent_type = str(args.get("subagent_type") or "image").strip() or "image"
        task_id = str(uuid4())
        parent_thread_id = self._thread_id(request)

        if not description:
            tool_message = self._tool_message(
                request,
                "错误：task_tool 缺少 description 参数。",
                status="error",
            )
            return Command(update={"messages": [tool_message]})

        self._write_stream_event(
            request,
            {"type": "task_started", "task_id": task_id, "description": description},
        )
        execution = self._subagent_executor.start(
            subagent_type,
            description,
            parent_thread_id=parent_thread_id,
        )
        status = "error" if execution.error else "success"
        content = str(execution.message.content or "")
        event_type = "task_failed" if execution.error else "task_completed"
        self._write_stream_event(
            request,
            {"type": event_type, "task_id": task_id, "result": content},
        )

        tool_message = self._tool_message(request, content, status=status)
        update = {
            "messages": [tool_message],
            "current_subagent": execution.context,
            "artifacts": execution.artifacts,
        }
        return Command(update=update)

    @staticmethod
    def _thread_id(request) -> str:
        config = getattr(getattr(request, "runtime", None), "config", None) or {}
        configurable = config.get("configurable") if isinstance(config, dict) else None
        return str((configurable or {}).get("thread_id") or "default")

    @staticmethod
    def _write_stream_event(request, event: dict[str, Any]) -> None:
        writer = getattr(getattr(request, "runtime", None), "stream_writer", None)
        if not callable(writer):
            return
        try:
            writer(event)
        except Exception:
            return

    @staticmethod
    def _tool_message(request, content: str, status: str = "success") -> ToolMessage:
        return ToolMessage(
            content=content,
            name=request.tool_call.get("name"),
            tool_call_id=request.tool_call.get("id") or request.tool_call.get("name") or "",
            status=status,
        )


__all__ = ["MCP_NO_TOOLS_MESSAGE", "MCP_TOOLS_LOADED_MESSAGE", "RuntimeToolMiddleware"]
