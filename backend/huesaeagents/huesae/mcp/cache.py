"""MCP 工具缓存层。

MCP 工具发现可能需要启动 stdio 子进程或连接远程服务，不适合在每轮模型调用前
重复执行。本模块为同步运行时提供一个带缓存的加载入口：

- 首次需要 MCP 能力时，调用 ``get_mcp_tools()`` 完成异步工具发现。
- 后续请求复用已发现的 LangChain ``BaseTool`` 列表。
- 扩展配置文件路径或修改时间变化时，自动重新发现工具。
- MCP 初始化失败时降级为空列表，让内置工具和主 Agent 仍可继续运行。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from langchain_core.tools import BaseTool

from ..config.extensions_config import ExtensionsConfig
from .tools import get_mcp_tools


logger = logging.getLogger(__name__)

# ``None`` 表示尚未执行 MCP discovery；空列表表示已经执行但没有可用工具。
_cached_tools: list[BaseTool] | None = None

# 缓存同时记录配置文件路径和修改时间。用户编辑 extensions_config.json 后，
# 下一次访问会自动重新发现工具，无需重启应用。
_cached_config_path: Path | None = None
_cached_config_mtime: float | None = None

# MCP 故障不会阻断主流程，因此额外保留最近一次错误，方便诊断配置或服务问题。
_last_mcp_error: str | None = None


def _config_signature(config_path: str | None = None) -> tuple[Path | None, float | None]:
    """返回当前配置文件路径与修改时间，用于判断缓存是否失效。"""
    path = ExtensionsConfig.resolve_config_path(config_path)
    if path is None:
        return None, None
    try:
        return path, path.stat().st_mtime
    except OSError:
        return path, None


def _run_async(coro):
    """在同步调用链中执行异步 MCP 初始化。

    MCP adapter 的工具发现接口是异步函数，但 ``SharedToolRuntime`` 当前通过同步
    方法加载工具。如果当前线程已经存在运行中的事件循环，不能再次直接调用
    ``asyncio.run()``，因此临时使用另一个线程运行协程。
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def initialize_mcp_tools(force: bool = False, config_path: str | None = None) -> list[BaseTool]:
    """初始化并缓存 MCP 工具。

    ``force=True`` 会忽略现有缓存。默认情况下，只有首次调用、配置文件路径变化
    或配置文件修改时间变化时，才会重新执行 MCP discovery。
    """
    global _cached_tools, _cached_config_path, _cached_config_mtime, _last_mcp_error

    config_path_obj, config_mtime = _config_signature(config_path)
    cache_valid = (
        not force
        and _cached_tools is not None
        and _cached_config_path == config_path_obj
        and _cached_config_mtime == config_mtime
    )
    if cache_valid:
        return _cached_tools

    try:
        # 真正的异步发现逻辑位于 tools.py。这里负责把结果接入同步运行时并缓存。
        _cached_tools = _run_async(get_mcp_tools(config_path))
        _cached_config_path = config_path_obj
        _cached_config_mtime = config_mtime
        _last_mcp_error = None
    except Exception as exc:
        # MCP 是可选扩展。初始化失败时保留错误信息并返回空工具列表，
        # 避免外部服务故障导致主 Agent 和内置工具一起不可用。
        logger.warning("初始化 MCP 工具失败：%s", exc, exc_info=True)
        _cached_tools = []
        _cached_config_path = config_path_obj
        _cached_config_mtime = config_mtime
        _last_mcp_error = str(exc)

    return _cached_tools


def get_cached_mcp_tools(config_path: str | None = None) -> list[BaseTool]:
    """读取 MCP 工具缓存，必要时自动初始化或刷新。"""
    return initialize_mcp_tools(force=False, config_path=config_path)


def reset_mcp_tools_cache() -> None:
    """清空 MCP 工具缓存，主要供测试使用。"""
    global _cached_tools, _cached_config_path, _cached_config_mtime, _last_mcp_error
    _cached_tools = None
    _cached_config_path = None
    _cached_config_mtime = None
    _last_mcp_error = None


def get_last_mcp_error() -> str | None:
    """返回最近一次 MCP 初始化错误。"""
    return _last_mcp_error
