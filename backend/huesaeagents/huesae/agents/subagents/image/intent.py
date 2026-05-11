"""意图识别模块

使用 LLM 结构化输出识别用户意图，替代关键词匹配。
支持：直接生图、转Danbooru标签、扩写提示词、普通对话
"""
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .prompts import INTENT_SYSTEM_MESSAGE


class ImageIntent(BaseModel):
    """用户意图识别结果"""

    intent: Literal["direct_image", "convert_tags", "expand_prompt", "chat"] = Field(
        description="用户意图类型：direct_image=直接生图, convert_tags=转Danbooru标签, expand_prompt=扩写提示词, chat=普通对话"
    )
    extracted_prompt: str = Field(
        description="提取的纯净提示词，移除动作词（如'画'、'生成'），只保留描述内容"
    )
    needs_clarification: bool = Field(
        description="是否需要澄清（提示词太短或不明确时设为true）"
    )
    clarification_message: str = Field(
        default="",
        description="如果需要澄清，提供给用户的提示消息"
    )


def recognize_intent(user_input: str, llm: BaseChatModel) -> ImageIntent:
    """使用LLM识别用户意图

    通过结构化输出让LLM返回意图类型和提取的提示词。

    Args:
        user_input: 用户原始输入
        llm: 大语言模型实例

    Returns:
        ImageIntent: 意图识别结果

    Example:
        >>> result = recognize_intent("画一个银发红瞳的少女在河边拿着笔画画", llm)
        >>> print(result.intent)  # "direct_image"
        >>> print(result.extracted_prompt)  # "银发红瞳的少女在河边拿着笔画画"
    """
    # 使用结构化输出
    structured_llm = llm.with_structured_output(ImageIntent)

    # 构建消息
    messages = [
        INTENT_SYSTEM_MESSAGE,
        HumanMessage(content=f"用户输入：{user_input}\n\n请以 JSON 格式输出分析结果。"),
    ]

    # 调用LLM
    result = structured_llm.invoke(messages)
    return result

