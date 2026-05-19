"""MCP client 配置转换。"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..config.extensions_config import ExtensionsConfig, McpServerConfig


logger = logging.getLogger(__name__)


def build_server_params(server_name: str, config: McpServerConfig) -> dict[str, Any]:
    """把项目配置转换成 MultiServerMCPClient 参数。"""
    transport_type = config.type or "stdio"
    params: dict[str, Any] = {"transport": transport_type}

    if transport_type == "stdio":
        if not config.command:
            raise ValueError(f"MCP server '{server_name}' 使用 stdio 时必须提供 command")
        params["command"] = config.command
        params["args"] = config.args
        # stdio MCP server 运行在子进程中，必须继承当前 PATH。
        # 否则 Windows 上即使命令行能找到 ffmpeg/ffprobe，Node MCP 子进程也可能找不到。
        params["env"] = {**os.environ, **config.env}
        return params

    if transport_type in ("sse", "http"):
        if not config.url:
            raise ValueError(f"MCP server '{server_name}' 使用 {transport_type} 时必须提供 url")
        params["url"] = config.url
        if config.headers:
            params["headers"] = config.headers
        return params

    raise ValueError(f"MCP server '{server_name}' 不支持传输类型：{transport_type}")


def build_servers_config(extensions_config: ExtensionsConfig) -> dict[str, dict[str, Any]]:
    """构建 MultiServerMCPClient 所需的 server 配置。"""
    servers_config: dict[str, dict[str, Any]] = {}
    for server_name, server_config in extensions_config.get_enabled_mcp_servers().items():
        try:
            servers_config[server_name] = build_server_params(server_name, server_config)
        except Exception as exc:
            logger.warning("跳过 MCP server %s：%s", server_name, exc)
    return servers_config
