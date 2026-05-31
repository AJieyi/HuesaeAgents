"""LangChain Agent 模型输入兼容层。

项目里正式运行的模型通常已经是 LangChain 的 ``BaseChatModel``，
但测试或局部调试时可能会传入只有 ``invoke`` 方法的轻量假模型。
``create_agent`` 需要标准 ChatModel 接口，因此这里提供一个很薄的适配器。


正常 DeepSeek 路径下，它对功能几乎没有实际影响；但它对测试、兼容自定义模型、保证 create_agent 不报类型/接口错误有影响。
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool


class ChatModelAdapter(BaseChatModel):
    """把轻量模型/测试替身包装成 LangChain 可调用的 ChatModel。

    适配目标很克制：只补齐 ``create_agent`` 实际依赖的最小接口，
    不改变被包装模型本身的推理逻辑。
    """

    wrapped: Any

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        """绑定工具。

        如果被包装模型自己支持 ``bind_tools``，优先交给原模型处理；
        否则回退到 ``BaseChatModel`` 的默认实现，让 LangChain 继续走标准流程。
        """
        if hasattr(self.wrapped, "bind_tools"):
            return self.wrapped.bind_tools(tools)
        return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)

    def bind(self, **kwargs: Any):
        """兼容 LangChain 的 ``bind`` 调用。

        测试替身通常不需要真的绑定运行参数，因此直接返回自身。
        """
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """把 ``invoke`` 的返回值转换成 LangChain 标准 ``ChatResult``。

        被包装对象可以返回 ``AIMessage``，也可以返回普通字符串或带
        ``content`` 属性的对象；这里统一包装成 ``AIMessage``，避免
        ``create_agent`` 在后续处理消息时拿到非标准类型。
        """
        response = self.wrapped.invoke(messages)
        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(getattr(response, "content", response)))
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "huesae-chat-model-adapter"


def ensure_chat_model(model: Any) -> BaseChatModel:
    """确保传给 Agent 的模型一定是 ``BaseChatModel``。

    正式模型原样返回；非标准轻量模型会被 ``ChatModelAdapter`` 包一层。
    """

    if isinstance(model, BaseChatModel):
        return model
    return ChatModelAdapter(wrapped=model)


__all__ = ["ChatModelAdapter", "ensure_chat_model"]
