"""MCP 工具缓存。"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from langchain_core.tools import BaseTool

from ..config.extensions_config import ExtensionsConfig
from .tools import get_mcp_tools


logger = logging.getLogger(__name__)

_cached_tools: list[BaseTool] | None = None
_cached_config_path: Path | None = None
_cached_config_mtime: float | None = None
_last_mcp_error: str | None = None


def _config_signature(config_path: str | None = None) -> tuple[Path | None, float | None]:
    """返回当前配置文件路径与 mtime，用于判断缓存是否失效。"""
    path = ExtensionsConfig.resolve_config_path(config_path)
    if path is None:
        return None, None
    try:
        return path, path.stat().st_mtime
    except OSError:
        return path, None


def _run_async(coro):
    """在同步调用链中执行异步 MCP 初始化。"""
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
    """初始化并缓存 MCP 工具。"""
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
        _cached_tools = _run_async(get_mcp_tools(config_path))
        _cached_config_path = config_path_obj
        _cached_config_mtime = config_mtime
        _last_mcp_error = None
    except Exception as exc:
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
