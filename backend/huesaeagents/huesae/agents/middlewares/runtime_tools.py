"""运行时工具中间件。

这个文件负责把 SharedToolRuntime 的“动态工具视图”接入 LangChain ``create_agent``：

- 每次调用模型前，注入最新工具列表和动态系统提示词。
- 模型请求 MCP 扩展能力时，触发 MCP 工具懒加载，并让模型重新规划。
- 模型调用 ``task_tool`` 时，不执行工具函数体，而是直接启动对应子 Agent。
- 将子 Agent 结果通过 LangGraph ``Command(update=...)`` 写回主图状态。

这样主 Agent 不需要自己手写 ReAct 循环，也不需要在启动时加载全部 MCP 工具。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langchain.messages import ToolMessage
from langchain.agents.middleware import ModelRequest
from langgraph.types import Command

from .base import AgentMiddleware


# load_mcp_tools_tool 返回的内部信号。
# 这个文本不是给用户展示的，而是通知中间件执行 MCP discovery。
LOAD_MCP_TOOLS_SIGNAL = "__LOAD_MCP_TOOLS__"

# MCP discovery 完成后，把这个工具消息返回给模型。
# 模型会在下一轮看到更新后的工具 schema，再选择具体 MCP 工具。
MCP_TOOLS_LOADED_MESSAGE = (
    "MCP扩展工具已加载。请结合用户原始需求，根据更新后的工具列表重新选择最合适的具体工具，"
    "并严格使用工具 schema 中的参数名。"
)
MCP_NO_TOOLS_MESSAGE = "当前：MCP扩展工具已加载，但当前没有可用的 MCP 工具。"


class RuntimeToolMiddleware(AgentMiddleware):
    """向已编译 Agent 暴露最新的运行时工具视图。

    ``create_agent`` 编译时会接收一份初始工具列表，但 HuesaeAgents 的工具
    不是完全静态的：注册子 Agent、加载 MCP 工具后，可见工具都会变化。
    因此这里使用回调函数，每次模型调用和工具调用时读取最新状态。
    """

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
        # 返回当前可见工具列表。通常指向主 Agent 的 self.tools。
        self._tools = tools

        # 返回工具名 -> 工具对象映射，执行 tool_call 时用于查找最新工具对象。
        self._tool_map = tool_map

        # 动态构建系统提示词。工具、MCP、Skill、记忆等变化后，提示词也会随之更新。
        self._system_message = system_message

        # MCP 懒加载回调：第一次需要扩展能力时才执行 discovery。
        self._ensure_mcp_tools = ensure_mcp_tools
        self._is_mcp_loaded = is_mcp_loaded
        self._has_mcp_tools = has_mcp_tools or (lambda: True)

        # 可选的子 Agent 执行器。主 Agent 会传入，general 子 Agent 不需要再次委派。
        self._subagent_executor = subagent_executor

    def wrap_model_call(self, request: ModelRequest, handler):
        """模型调用前注入最新工具列表和系统提示词。

        ``request.override`` 不修改原 request，而是为本次模型调用创建一个带最新
        tools/system_message 的请求。这样 MCP 加载后不需要重启整个应用。
        """
        user_input = request.state.get("user_input") if isinstance(request.state, dict) else None
        return handler(
            request.override(
                tools=self._tools(),
                system_message=self._system_message(user_input),
            )
        )

    def wrap_tool_call(self, request, handler):
        """拦截工具调用，处理子 Agent 委派和 MCP 懒加载。"""
        tool_name = request.tool_call.get("name") or ""

        # task_tool 是一个“委派入口”，不是普通业务工具。
        # 它的函数体不会执行；这里直接调用 SubagentExecutor，并把结果写回主图。
        if tool_name == "task_tool" and self._subagent_executor is not None:
            return self._run_subagent_task(request)

        # 优先使用最新 tool_map 中的工具对象。
        # request.tool 是 LangChain 编译时保存的工具对象，可作为静态工具兜底。
        tool = self._tool_map().get(tool_name) or request.tool
        if tool is not None:
            result = handler(request.override(tool=tool))

            # load_mcp_tools_tool 只是发出内部信号。收到信号后执行 MCP discovery，
            # 再返回一条 ToolMessage，提示模型基于新工具列表重新规划。
            if isinstance(result, ToolMessage) and str(result.content) == LOAD_MCP_TOOLS_SIGNAL:
                if not self._is_mcp_loaded():
                    self._ensure_mcp_tools()
                content = MCP_TOOLS_LOADED_MESSAGE if self._has_mcp_tools() else MCP_NO_TOOLS_MESSAGE
                return self._tool_message(request, content)
            return result

        # 模型有时会根据用户需求猜出一个尚未加载的 MCP 工具名。
        # 此时不要立刻报错，也不要使用可能错误的参数执行工具：
        # 先加载 MCP 工具，让模型拿到真实 schema 后重新规划。
        if not self._is_mcp_loaded():
            self._ensure_mcp_tools()
            content = MCP_TOOLS_LOADED_MESSAGE if self._has_mcp_tools() else MCP_NO_TOOLS_MESSAGE
            return self._tool_message(request, content)

        # MCP 已完成 discovery 仍找不到工具时，才返回明确错误。
        available = list(self._tool_map().keys())
        return self._tool_message(request, f"错误：未知工具 {tool_name}。可用工具：{available}", status="error")

    def _run_subagent_task(self, request) -> Command:
        """执行 task_tool 委派，并把子 Agent 结果写回 LangGraph 状态。"""
        args = request.tool_call.get("args") or {}
        description = str(args.get("description") or "").strip()
        subagent_type = str(args.get("subagent_type") or "image").strip() or "image"
        task_id = str(uuid4())
        parent_thread_id = self._thread_id(request)

        # description 是子 Agent 的首轮输入，缺失时无法启动任务。
        if not description:
            tool_message = self._tool_message(
                request,
                "错误：task_tool 缺少 description 参数。",
                status="error",
            )
            return Command(update={"messages": [tool_message]})

        # stream_writer 可用于未来的流式 UI 展示任务开始/完成事件。
        # 当前终端入口不依赖这些事件，因此写入失败时会静默忽略。
        self._write_stream_event(
            request,
            {"type": "task_started", "task_id": task_id, "description": description},
        )

        # SubagentExecutor 负责查找子 Agent、派生子线程 ID、初始化局部状态、
        # 调用子 Agent，并把返回值归一化成 SubagentExecution。
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

        # Command(update=...) 会把子 Agent 结果合并回主 Agent 的 LangGraph 状态：
        # - messages：把子 Agent 回复记录为 task_tool 的 ToolMessage
        # - current_subagent：流程未结束时保存续聊上下文，结束时为 None
        # - artifacts：保存图片等独立产物引用
        tool_message = self._tool_message(request, content, status=status)
        update = {
            "messages": [tool_message],
            "current_subagent": execution.context,
            "artifacts": execution.artifacts,
        }
        return Command(update=update)

    @staticmethod
    def _thread_id(request) -> str:
        """读取当前主图 thread_id，供子 Agent 派生隔离的子线程 ID。"""
        config = getattr(getattr(request, "runtime", None), "config", None) or {}
        configurable = config.get("configurable") if isinstance(config, dict) else None
        return str((configurable or {}).get("thread_id") or "default")

    @staticmethod
    def _write_stream_event(request, event: dict[str, Any]) -> None:
        """尽力写入自定义流式事件；没有 writer 或写入失败时不影响主流程。"""
        writer = getattr(getattr(request, "runtime", None), "stream_writer", None)
        if not callable(writer):
            return
        try:
            writer(event)
        except Exception:
            return

    @staticmethod
    def _tool_message(request, content: str, status: str = "success") -> ToolMessage:
        """构造与当前 tool_call 对应的 LangChain ToolMessage。"""
        return ToolMessage(
            content=content,
            name=request.tool_call.get("name"),
            tool_call_id=request.tool_call.get("id") or request.tool_call.get("name") or "",
            status=status,
        )


__all__ = ["MCP_NO_TOOLS_MESSAGE", "MCP_TOOLS_LOADED_MESSAGE", "RuntimeToolMiddleware"]
