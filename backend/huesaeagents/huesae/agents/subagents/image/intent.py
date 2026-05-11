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
        HumanMessage(content=f"用户输入：{user_input}"),
    ]

    # 调用LLM
    result = structured_llm.invoke(messages)
    return result


def recognize_intent_simple(user_input: str, llm: BaseChatModel) -> dict:
    """简化版意图识别（不使用结构化输出，兼容旧模型）

    如果模型不支持结构化输出，使用此方法作为fallback。

    Args:
        user_input: 用户原始输入
        llm: 大语言模型实例

    Returns:
        dict: 包含 intent, extracted_prompt, needs_clarification
    """
    prompt = f"""分析以下用户输入，返回JSON格式结果：

用户输入：{user_input}

请返回以下JSON格式（不要包含其他内容）：
{{
    "intent": "direct_image|convert_tags|expand_prompt|chat",
    "extracted_prompt": "提取的纯净提示词",
    "needs_clarification": true|false,
    "clarification_message": "如果需要澄清，提供提示消息"
}}

规则：
- intent=direct_image: 用户想要生成图片
- intent=convert_tags: 用户想要转成Danbooru标签
- intent=expand_prompt: 用户想要扩写提示词
- intent=chat: 普通对话
- extracted_prompt: 移除"画"、"生成"等动作词后的纯净描述
- needs_clarification: 提示词少于5个字时设为true"""

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    # 尝试解析JSON
    try:
        import json
        # 提取JSON部分
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())
        return {
            "intent": result.get("intent", "chat"),
            "extracted_prompt": result.get("extracted_prompt", ""),
            "needs_clarification": result.get("needs_clarification", False),
            "clarification_message": result.get("clarification_message", ""),
        }
    except Exception:
        # 解析失败，返回默认值
        return {
            "intent": "chat",
            "extracted_prompt": "",
            "needs_clarification": False,
            "clarification_message": "",
        }
