"""模型工厂"""
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from .providers import DoubaoVisionClient, create_deepseek_model, create_doubao_vision_client

# 加载 .env 文件
load_dotenv()

# 使用默认参数
# model = create_chat_model("deepseek")

# # 指定模型名
# model = create_chat_model("deepseek", model="deepseek-chat")

# # 额外传参
# model = create_chat_model("deepseek", model="deepseek-v4-flash", temperature=0.5, max_tokens=2048)
def create_chat_model(
    provider: str = "deepseek",
    model: str = "deepseek-v4-flash",
    **kwargs,
) -> BaseChatModel:
    """
    创建聊天模型实例

    Args:
        provider: 模型提供商，如 "deepseek"
        model: 模型名称，如 "deepseek-v4-flash"
        **kwargs: 其他参数（如 temperature, max_tokens 等）
    Returns:
        BaseChatModel 实例
    """
    if provider == "deepseek":
        return create_deepseek_model(model=model, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def create_vision_client(
    provider: str = "doubao",
    **kwargs,
) -> DoubaoVisionClient:
    """创建视觉理解客户端。

    视觉模型不包装成 LangChain ChatModel，保持和 DeerFlow 类似的独立
    provider + 工厂结构，避免把图片 base64 混入普通对话消息。
    """
    if provider == "doubao":
        return create_doubao_vision_client(**kwargs)
    raise ValueError(f"Unsupported vision provider: {provider}")
