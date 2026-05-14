"""Agent 行为测试。

测试范围：
1. 主Agent工具注册和子Agent委派
2. 生图子Agent的标准化接口
3. 标签生成、提示词扩写、Provider 注册等独立模块

本文件使用本地假模型，不访问真实 LLM 或生图 API。
"""
import sys
from pathlib import Path

import pytest
from langchain.messages import AIMessage, HumanMessage

backend_dir = Path(__file__).resolve().parents[4]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from huesaeagents.huesae.agents.lead_agent import HuesaeMainAgent
from huesaeagents.huesae.subagents.image_agent import (
    ImageDecision,
    ImageSubAgent,
    create_image_agent,
)
from huesaeagents.huesae.subagents.image import (
    DoubaoProvider,
    expand_prompt,
    generate_tags,
)
from huesaeagents.huesae.tools.tools import Action


class FakeStructuredLLM:
    """按 Pydantic Schema 返回固定结构化结果的假模型。"""

    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        user_input = self._latest_user_text(messages)

        if self.schema is Action:
            return self._decide_main_action(user_input)

        if self.schema is ImageDecision:
            return self._decide_image_action(user_input)

        raise AssertionError(f"测试假模型不支持的结构化输出：{self.schema}")

    @staticmethod
    def _latest_user_text(messages) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                content = str(message.content)
                for line in content.splitlines():
                    if line.startswith("- 用户最新输入："):
                        return line.removeprefix("- 用户最新输入：").strip()
                return content
        return ""

    @staticmethod
    def _decide_main_action(user_input: str) -> Action:
        if user_input == "我想生成图片":
            return Action(
                thought="用户想生图但没有给出描述，委派生图子Agent追问。",
                type="tool_call",
                tool_name="task_tool",
                tool_args={"description": user_input, "subagent_type": "image"},
            )

        return Action(
            thought="普通对话，直接回复。",
            type="reply",
            response="你好呀，我在这里~",
        )

    @staticmethod
    def _decide_image_action(user_input: str) -> ImageDecision:
        if "推荐" in user_input:
            return ImageDecision(
                thought="用户需要推荐图片主题。",
                action="recommend",
                response="可以试试：樱花树下的少女、雨夜霓虹街道、星空下的魔法书。",
                prompt=None,
                provider="doubao",
            )

        if "扩展" in user_input or "扩写" in user_input:
            return ImageDecision(
                thought="用户要求扩写提示词。",
                action="expand",
                response="我来帮你扩写一下~",
                prompt="夏天的图",
                provider="doubao",
            )

        if "谢谢" in user_input or "好看" in user_input:
            return ImageDecision(
                thought="用户表示满意，结束子Agent对话。",
                action="finish",
                response="喜欢就好，下次还可以继续来找我画图~",
            )

        if user_input == "我想生成图片":
            return ImageDecision(
                thought="缺少图片描述，需要追问。",
                action="ask_prompt",
                response="请告诉我您想要生成什么样的图片？",
                prompt=None,
                provider="doubao",
            )

        return ImageDecision(
            thought="用户已经给出可生成的描述。",
            action="generate",
            response="图片正在生成中，请稍等~",
            prompt=user_input,
            provider="doubao",
            size="2K",
            output_format="jpeg",
            is_batch=False,
        )


class FakeLLM:
    """覆盖测试所需的 LLM 接口，避免真实网络调用。"""

    def with_structured_output(self, schema, method=None):
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


@pytest.fixture(scope="module")
def main_agent(llm):
    """共享的主Agent实例。"""
    agent = HuesaeMainAgent(llm=llm)
    agent.register_sub_agent(create_image_agent(llm=llm, providers=[]))
    return agent


@pytest.fixture(scope="module")
def image_agent(llm):
    """共享的生图子Agent实例。"""
    return ImageSubAgent(llm=llm, providers=[])


class TestMainAgentHarness:
    """测试主Agent的工具注册和子Agent委派能力。"""

    def test_tools_are_available(self, main_agent):
        """主Agent应暴露当前可用工具列表。"""
        tool_names = {tool.name for tool in main_agent.tools}
        assert "generate_image_tool" in tool_names
        assert "generate_images_tool" in tool_names
        assert "task_tool" in tool_names

    def test_image_subagent_registered(self, main_agent):
        """生图子Agent应注册到子Agent注册表。"""
        assert main_agent.subagent_registry.get("image") is not None


class TestImageSubAgent:
    """测试生图子Agent标准化接口。"""

    def test_ask_prompt_when_no_description(self, image_agent):
        """用户只说想生成图片时，子Agent应追问具体描述。"""
        result = image_agent.process({}, "我想生成图片")

        assert result["action"] == "ask_prompt"
        assert "请告诉我" in result["response"]

    def test_generate_when_has_prompt(self, image_agent):
        """用户提供明确提示词时，子Agent应进入生成流程。"""
        state = {
            "messages": [
                HumanMessage(content="我想生成图片"),
                AIMessage(content="请告诉我您想要生成什么样的图片？"),
            ],
        }
        result = image_agent.process(state, "夕阳下看大海的少女，穿着水手服")

        assert result["action"] == "generate"
        assert result["prompt"].startswith("图片风格为 二次元")
        assert result["data"]["size"] == "2K"

    def test_recommend_when_asked(self, image_agent):
        """用户要求推荐时，子Agent应返回推荐内容。"""
        result = image_agent.process({}, "你帮我推荐一些吧")

        assert result["action"] == "recommend"
        assert "樱花树下" in result["response"]

    def test_expand_when_asked(self, image_agent):
        """用户要求扩写时，子Agent应扩写并进入确认状态。"""
        state = {
            "messages": [HumanMessage(content="夏天的图")],
            "image_prompt": "夏天的图",
        }
        result = image_agent.process(state, "你帮我扩展一下吧")

        assert result["action"] == "ask_confirm"
        assert "扩写后的描述" in result["response"]
        assert result["data"]["expanded_prompt"].startswith("图片风格为 二次元")

    def test_finish_when_satisfied(self, image_agent):
        """用户满意后，子Agent应返回结束动作。"""
        result = image_agent.process({}, "真好看，谢谢")

        assert result["action"] == "finish"
        assert "喜欢就好" in result["response"]

    def test_decide_structured_output(self, image_agent):
        """LLM 决策结果应符合 ImageDecision 结构。"""
        decision = image_agent._decide({}, "画一个猫娘")

        assert isinstance(decision, ImageDecision)
        assert decision.action == "generate"
        assert decision.response is not None

    def test_standardized_result_format(self, image_agent):
        """子Agent返回结果应符合统一格式。"""
        result = image_agent.process({}, "我想生成图片")

        assert set(result) == {"action", "response", "prompt", "provider", "data"}


class TestMainAgentIntegration:
    """测试主Agent与生图子Agent集成。"""

    def test_chat_directly(self, main_agent):
        """普通聊天应由主Agent直接回复。"""
        result = main_agent.process({"messages": []}, "你好")

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "你好呀" in result["messages"][0].content

    def test_delegate_to_image_agent(self, main_agent):
        """模糊生图需求应委派给生图子Agent追问。"""
        result = main_agent.process({"messages": []}, "我想生成图片")

        assert len(result["messages"]) == 1
        assert "请告诉我" in result["messages"][0].content
        assert result["active_subagent"]["agent_type"] == "image"

    def test_chat_after_image_generation(self, main_agent):
        """没有 active_subagent 时，普通反馈不应再次进入生图流程。"""
        messages = [
            HumanMessage(content="我想生成图片"),
            AIMessage(content="请告诉我您想要生成什么样的图片？"),
            HumanMessage(content="夕阳下看大海的少女"),
            AIMessage(content="图片已生成完成"),
        ]
        result = main_agent.process({"messages": messages}, "真好看")

        assert "image_url" not in result
        assert len(result["messages"]) == 1
        assert "你好呀" in result["messages"][0].content


class TestDanbooruTags:
    """测试 Danbooru 标签生成。"""

    def test_generate_tags(self, llm):
        """标签生成函数应返回清洗后的标签列表。"""
        tags = generate_tags("一个银发红瞳的少女在樱花树下", llm)

        assert tags[:3] == ["1girl", "silver hair", "red eyes"]
        assert "anime style" in tags

    def test_tags_to_prompt(self):
        """标签列表应拼接为逗号分隔提示词。"""
        from huesaeagents.huesae.subagents.image import tags_to_prompt

        tags = ["1girl", "silver hair", "red eyes"]
        prompt = tags_to_prompt(tags)

        assert prompt == "1girl, silver hair, red eyes"


class TestExpandPrompt:
    """测试提示词扩写。"""

    def test_expand_prompt(self, llm):
        """扩写函数应返回更长的自然语言描述。"""
        expanded = expand_prompt("夕阳下的战舰", llm)

        assert isinstance(expanded, str)
        assert len(expanded) > len("夕阳下的战舰")
        assert "战舰" in expanded


class TestProviders:
    """测试 Provider 注册。"""

    def test_register_provider(self, llm):
        """生图Agent应支持动态注册Provider。"""
        agent = ImageSubAgent(llm=llm, providers=[])

        assert len(agent.providers) == 0

        agent.register_provider(DoubaoProvider())

        assert "doubao" in agent.providers

    def test_default_provider(self, llm):
        """默认创建生图Agent时应注册豆包Provider。"""
        agent = create_image_agent(llm=llm)
        names = agent.get_available_providers()

        assert "doubao" in names
