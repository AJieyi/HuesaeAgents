"""中间件配置。

配置保持轻量单例模式，方便测试时注入，也方便后续把开关迁移到文件或环境变量。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TokenUsageConfig(BaseModel):
    """Token 用量日志配置。"""

    enabled: bool = Field(default=True, description="是否启用 token 用量日志")


class MiddlewareConfig(BaseModel):
    """Agent 中间件总配置。"""

    token_usage: TokenUsageConfig = Field(default_factory=TokenUsageConfig)


_middleware_config: MiddlewareConfig | None = None


def get_middleware_config() -> MiddlewareConfig:
    """读取并缓存中间件配置。"""
    global _middleware_config
    if _middleware_config is None:
        _middleware_config = MiddlewareConfig()
    return _middleware_config


def set_middleware_config(config: MiddlewareConfig) -> None:
    """注入中间件配置，主要供测试和启动配置使用。"""
    global _middleware_config
    _middleware_config = config


def reset_middleware_config() -> None:
    """清空中间件配置缓存，主要供测试使用。"""
    global _middleware_config
    _middleware_config = None
