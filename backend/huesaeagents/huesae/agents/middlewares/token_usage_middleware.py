"""Token 用量记录中间件。"""

from __future__ import annotations

import logging
from typing import Any

from .base import AgentMiddleware


class TokenUsageMiddleware(AgentMiddleware):
    """记录每次 LLM 响应携带的 token 用量。"""

    def after_model(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
        """模型响应后读取 usage_metadata 并写入日志。"""
        self._log_token_usage(state)
        return None

    async def aafter_model(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
        """异步模型响应后读取 usage_metadata 并写入日志。"""
        self._log_token_usage(state)
        return None

    @staticmethod
    def _log_token_usage(state: dict) -> None:
        messages = state.get("messages") or []
        if not messages:
            return

        latest_message = messages[-1]
        usage_metadata = getattr(latest_message, "usage_metadata", None) or {}
        if not usage_metadata:
            return

        input_tokens = usage_metadata.get("input_tokens", usage_metadata.get("prompt_tokens", 0))
        output_tokens = usage_metadata.get("output_tokens", usage_metadata.get("completion_tokens", 0))
        total_tokens = usage_metadata.get("total_tokens", 0)
        if not total_tokens:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        logging.info(
            "LLM token usage: input=%s output=%s total=%s",
            input_tokens,
            output_tokens,
            total_tokens,
        )


__all__ = ["TokenUsageMiddleware"]
