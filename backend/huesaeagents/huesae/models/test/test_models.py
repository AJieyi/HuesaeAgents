"""模型工厂测试。

默认只验证本地工厂逻辑，不直接调用真实模型 API。
"""
import sys
from pathlib import Path

import pytest
from langchain_deepseek import ChatDeepSeek

backend_dir = Path(__file__).resolve().parents[4]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from huesaeagents.huesae.models.models_factory import create_chat_model
from huesaeagents.huesae.models.providers import create_deepseek_model


def test_create_deepseek_model_requires_api_key(monkeypatch):
    """没有配置 API Key 时，应给出明确错误。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        create_deepseek_model()


def test_create_chat_model_deepseek(monkeypatch):
    """工厂函数应能创建 DeepSeek ChatModel 实例。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    model = create_chat_model(provider="deepseek", model="deepseek-v4-flash")

    assert isinstance(model, ChatDeepSeek)


def test_create_chat_model_unknown_provider():
    """未知模型提供商应快速失败。"""
    with pytest.raises(ValueError, match="Unsupported provider"):
        create_chat_model(provider="unknown")
