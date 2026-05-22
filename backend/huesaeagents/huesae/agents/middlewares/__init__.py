"""Agent 中间件入口。"""

from __future__ import annotations

from ...config.middleware_config import get_middleware_config
from .base import AgentMiddleware
from .pipeline import MiddlewarePipeline
from .token_usage_middleware import TokenUsageMiddleware


def build_middlewares() -> MiddlewarePipeline:
    """按项目配置组装中间件管道。"""
    config = get_middleware_config()
    pipeline = MiddlewarePipeline()
    if config.token_usage.enabled:
        pipeline.add(TokenUsageMiddleware())
    return pipeline


__all__ = [
    "AgentMiddleware",
    "MiddlewarePipeline",
    "TokenUsageMiddleware",
    "build_middlewares",
]
