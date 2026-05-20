"""DeepSeek Provider"""
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

load_dotenv()


def create_deepseek_model(
    model: str = "deepseek-v4-flash",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> ChatDeepSeek:
    """
    创建 DeepSeek 模型实例

    Args:
        model: 模型名称，默认为 deepseek-v4-flash
        temperature: 温度参数
        max_tokens: 最大 token 数
        **kwargs: 其他参数

    Returns:
        ChatDeepSeek 实例
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DeepSeek API密钥未找到。请配置 DEEPSEEK_API_KEY 环境变量")

    kwargs.setdefault("request_timeout", 120)

    return ChatDeepSeek(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
