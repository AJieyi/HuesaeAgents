"""Compatibility helpers for LangChain agent model inputs."""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool


class ChatModelAdapter(BaseChatModel):
    """Wrap lightweight test doubles so ``create_agent`` can call them."""

    wrapped: Any

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        if hasattr(self.wrapped, "bind_tools"):
            return self.wrapped.bind_tools(tools)
        return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)

    def bind(self, **kwargs: Any):
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self.wrapped.invoke(messages)
        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(getattr(response, "content", response)))
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "huesae-chat-model-adapter"


def ensure_chat_model(model: Any) -> BaseChatModel:
    """Return a BaseChatModel, adapting small local fakes when needed."""

    if isinstance(model, BaseChatModel):
        return model
    return ChatModelAdapter(wrapped=model)


__all__ = ["ChatModelAdapter", "ensure_chat_model"]
