"""模型工厂"""
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from .providers import create_deepseek_model

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
