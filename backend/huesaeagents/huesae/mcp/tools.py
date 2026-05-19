"""MCP 工具加载。

MCP 工具由 langchain-mcp-adapters 转为 LangChain BaseTool。
主Agent当前是同步 ReAct 循环，因此这里会为 async-only MCP tool 补同步包装。
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import logging
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool

from ..config.extensions_config import ExtensionsConfig
from .client import build_servers_config


logger = logging.getLogger(__name__)
_DISABLED_MCP_TOOL_NAME_PARTS = ("extract_video_frames",)

_SYNC_TOOL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=10,
    thread_name_prefix="huesae-mcp-tool",
)
atexit.register(lambda: _SYNC_TOOL_EXECUTOR.shutdown(wait=False))


def _make_sync_tool_wrapper(coro: Callable[..., Any], tool_name: str) -> Callable[..., Any]:
    """把 MCP 异步工具包装为同步函数。"""

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            future = _SYNC_TOOL_EXECUTOR.submit(asyncio.run, coro(*args, **kwargs))
            return future.result()
        return asyncio.run(coro(*args, **kwargs))

    sync_wrapper.__name__ = f"sync_{tool_name}"
    return sync_wrapper


async def get_mcp_tools(config_path: str | None = None) -> list[BaseTool]:
    """从启用的 MCP server 加载工具。"""
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.warning("未安装 langchain-mcp-adapters，跳过 MCP 工具加载")
        return []

    extensions_config = ExtensionsConfig.from_file(config_path)
    servers_config = build_servers_config(extensions_config)
    if not servers_config:
        return []

    try:
        client = MultiServerMCPClient(servers_config, tool_name_prefix=True)
        tools = await client.get_tools()
        tools = [
            tool
            for tool in tools
            if not any(part in tool.name for part in _DISABLED_MCP_TOOL_NAME_PARTS)
        ]
        for tool in tools:
            if getattr(tool, "func", None) is None and getattr(tool, "coroutine", None) is not None:
                tool.func = _make_sync_tool_wrapper(tool.coroutine, tool.name)
        return tools
    except Exception as exc:
        logger.warning("MCP 工具加载失败：%s", exc, exc_info=True)
        return []
