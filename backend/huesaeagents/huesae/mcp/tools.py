"""MCP 工具发现与适配层。

执行流程：

1. 读取扩展配置并转换为 ``MultiServerMCPClient`` 参数。
2. 由 ``langchain-mcp-adapters`` 启动或连接启用的 MCP server。
3. 通过 MCP discovery 获取服务端暴露的工具。
4. 将工具转换成 LangChain ``BaseTool``，供主 Agent 和子 Agent 使用。
5. 为只有异步入口的 MCP 工具补充同步包装，适配项目当前的同步调用链。

本模块负责一次实际发现过程；是否复用结果由 ``cache.py`` 决定。
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
# 当前项目主动隐藏名称中包含这些片段的 MCP 工具。
# 过滤发生在 discovery 之后，因此不会影响同一 MCP server 暴露的其他能力。
_DISABLED_MCP_TOOL_NAME_PARTS = ("extract_video_frames",)

# 同步工具调用发生在已有事件循环中时，使用独立线程运行异步 coroutine，
# 避免在同一线程中嵌套调用 ``asyncio.run()``。
_SYNC_TOOL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=10,
    thread_name_prefix="huesae-mcp-tool",
)
atexit.register(lambda: _SYNC_TOOL_EXECUTOR.shutdown(wait=False))


def _make_sync_tool_wrapper(coro: Callable[..., Any], tool_name: str) -> Callable[..., Any]:
    """把 MCP 异步工具包装为同步函数，供同步工具执行链调用。"""

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
    """从启用的 MCP server 发现并加载工具。

    MCP 属于可选扩展：依赖未安装、没有启用的 server 或发现过程异常时，统一返回
    空列表，让调用方继续使用项目内置工具。
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.warning("未安装 langchain-mcp-adapters，跳过 MCP 工具加载")
        return []

    # ExtensionsConfig.from_file() 会加载 .env，并展开 JSON 中的环境变量占位符。
    extensions_config = ExtensionsConfig.from_file(config_path)
    servers_config = build_servers_config(extensions_config)
    if not servers_config:
        return []

    try:
        # stdio server 通常会在 get_tools() 过程中启动子进程；
        # sse/http server 则会连接远程地址。adapter 最终返回 LangChain BaseTool。
        client = MultiServerMCPClient(servers_config, tool_name_prefix=True)
        tools = await client.get_tools()
        tools = [
            tool
            for tool in tools
            if not any(part in tool.name for part in _DISABLED_MCP_TOOL_NAME_PARTS)
        ]
        for tool in tools:
            # 部分 MCP 工具只有 coroutine，没有同步 func。为它们补充同步入口后，
            # 当前同步 Agent 运行时可以像调用普通 LangChain 工具一样调用它们。
            if getattr(tool, "func", None) is None and getattr(tool, "coroutine", None) is not None:
                tool.func = _make_sync_tool_wrapper(tool.coroutine, tool.name)
        return tools
    except Exception as exc:
        logger.warning("MCP 工具加载失败：%s", exc, exc_info=True)
        return []
