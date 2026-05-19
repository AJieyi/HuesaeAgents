"""扩展配置。

当前只实现 MCP server 配置读取，保持 deerflow 风格的可选扩展机制：
配置不存在或 MCP 加载失败时，不影响内置工具与主Agent启动。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


_ENV_PATTERN = re.compile(r"\$(\w+)|\$\{([^}]+)\}")


def load_project_env() -> None:
    """加载项目根目录 `.env`，让扩展配置可以读取环境变量占位符。"""
    package_dir = Path(__file__).resolve().parents[1]
    backend_dir = package_dir.parents[1]
    repo_root = backend_dir.parent
    env_path = repo_root / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(env_path, override=False)
    _fill_douyin_mcp_api_key_alias()


def _fill_douyin_mcp_api_key_alias() -> None:
    """为抖音 MCP 准备独立密钥别名。

    douyin-mcp-server 子进程读取的变量名固定为 API_KEY；项目侧优先使用
    DOUYIN_MCP_API_KEY 隔离语义。如果用户只配置了阿里云百炼常用的
    DASHSCOPE_API_KEY，这里自动补成 DOUYIN_MCP_API_KEY。
    """
    if os.getenv("DOUYIN_MCP_API_KEY"):
        return
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if dashscope_api_key:
        os.environ["DOUYIN_MCP_API_KEY"] = dashscope_api_key


class McpServerConfig(BaseModel):
    """单个 MCP server 配置。"""

    enabled: bool = Field(default=True, description="是否启用该 MCP server")
    type: str = Field(default="stdio", description="传输类型，当前主要使用 stdio")
    command: str | None = Field(default=None, description="stdio MCP server 启动命令")
    args: list[str] = Field(default_factory=list, description="stdio MCP server 命令参数")
    env: dict[str, str] = Field(default_factory=dict, description="传给 MCP server 的环境变量")
    url: str | None = Field(default=None, description="sse/http MCP server 地址")
    headers: dict[str, str] = Field(default_factory=dict, description="sse/http 请求头")
    description: str = Field(default="", description="给用户和LLM看的 MCP server 能力说明")
    model_config = ConfigDict(extra="allow")


class ExtensionsConfig(BaseModel):
    """项目扩展配置。"""

    mcp_servers: dict[str, McpServerConfig] = Field(
        default_factory=dict,
        alias="mcpServers",
        description="MCP server 配置表",
    )
    source_path: Path | None = Field(default=None, exclude=True)
    model_config = ConfigDict(extra="allow", populate_by_name=True, arbitrary_types_allowed=True)

    @classmethod
    def resolve_config_path(cls, config_path: str | None = None) -> Path | None:
        """按约定顺序查找扩展配置文件。"""
        if config_path:
            path = Path(config_path)
            return path if path.exists() else None

        env_path = os.getenv("HUESAE_EXTENSIONS_CONFIG_PATH")
        if env_path:
            path = Path(env_path)
            return path if path.exists() else None

        package_dir = Path(__file__).resolve().parents[1]
        backend_dir = package_dir.parents[1]
        repo_root = backend_dir.parent
        for path in (
            repo_root / "extensions_config.json",
            backend_dir / "extensions_config.json",
            backend_dir / "huesaeagents" / "extensions_config.json",
        ):
            if path.exists():
                return path

        return None

    @classmethod
    def from_file(cls, config_path: str | None = None) -> "ExtensionsConfig":
        """从 JSON 文件读取扩展配置；配置不存在时返回空配置。"""
        load_project_env()
        resolved_path = cls.resolve_config_path(config_path)
        if resolved_path is None:
            return cls(mcp_servers={})

        try:
            with resolved_path.open(encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"扩展配置不是合法 JSON：{resolved_path}") from exc

        resolved_data = cls.resolve_env_variables(data)
        config = cls.model_validate(resolved_data)
        config.source_path = resolved_path
        return config

    @classmethod
    def resolve_env_variables(cls, value: Any) -> Any:
        """递归展开 `$VAR` 与 `${VAR}` 环境变量。"""
        if isinstance(value, str):
            return _ENV_PATTERN.sub(lambda match: os.getenv(match.group(1) or match.group(2) or "", ""), value)
        if isinstance(value, list):
            return [cls.resolve_env_variables(item) for item in value]
        if isinstance(value, dict):
            return {key: cls.resolve_env_variables(item) for key, item in value.items()}
        return value

    def get_enabled_mcp_servers(self) -> dict[str, McpServerConfig]:
        """返回启用的 MCP server。"""
        return {
            name: server
            for name, server in self.mcp_servers.items()
            if server.enabled
        }


_extensions_config: ExtensionsConfig | None = None


def get_extensions_config() -> ExtensionsConfig:
    """读取并缓存扩展配置。"""
    global _extensions_config
    if _extensions_config is None:
        _extensions_config = ExtensionsConfig.from_file()
    return _extensions_config


def reload_extensions_config(config_path: str | None = None) -> ExtensionsConfig:
    """重新加载扩展配置。"""
    global _extensions_config
    _extensions_config = ExtensionsConfig.from_file(config_path)
    return _extensions_config


def reset_extensions_config() -> None:
    """清空扩展配置缓存，主要供测试使用。"""
    global _extensions_config
    _extensions_config = None


def set_extensions_config(config: ExtensionsConfig) -> None:
    """注入扩展配置，主要供测试使用。"""
    global _extensions_config
    _extensions_config = config
