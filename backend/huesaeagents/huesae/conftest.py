"""huesae 测试共享 fixture。"""

import pytest
from langchain.messages import AIMessage


@pytest.fixture(autouse=True)
def isolated_extensions_config(monkeypatch, tmp_path):
    """测试默认使用空 MCP 配置，避免启动真实 npm MCP server。"""
    config_path = tmp_path / "extensions_config.json"
    config_path.write_text('{"mcpServers": {}}', encoding="utf-8")
    monkeypatch.setenv("HUESAE_EXTENSIONS_CONFIG_PATH", str(config_path))

    from huesaeagents.huesae.mcp.cache import reset_mcp_tools_cache

    reset_mcp_tools_cache()
    yield
    reset_mcp_tools_cache()


class FakeLLM:
    """覆盖测试所需的最小 LLM 接口。"""

    def with_structured_output(self, schema, method=None):
        from huesaeagents.huesae.agents.test.test_agents import FakeStructuredLLM

        return FakeStructuredLLM(schema)

    def invoke(self, messages):
        text = "\n".join(str(getattr(message, "content", "")) for message in messages)

        if "Danbooru标签" in text:
            return AIMessage(content="1girl, silver hair, red eyes, cherry blossoms, anime style")
        if "请扩写以下描述" in text:
            return AIMessage(content="夕阳下的战舰停泊在金色海面上，云层被晚霞染亮，画面具有二次元插画质感。")
        if "用户请求的图片已经生成完成了" in text:
            return AIMessage(content="这是生成好的图片哦~")
        return AIMessage(content="你好呀，我在这里~")


@pytest.fixture(scope="module")
def llm():
    """共享的本地假模型。"""
    return FakeLLM()
