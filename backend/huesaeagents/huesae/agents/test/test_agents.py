"""Agent 测试

测试内容：
1. 主Agent工具选择和子Agent委派
2. 子Agent标准化接口（追问、推荐、扩写、确认、生图）
3. 主Agent集成测试（聊天、委派子Agent、包装展示）
"""
import sys
from pathlib import Path

# 将 backend 目录添加到 Python 路径
backend_dir = Path(__file__).resolve().parents[4]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from huesaeagents.huesae.agents.lead_agent import HuesaeMainAgent
from huesaeagents.huesae.agents.subagents.image_agent import (
    ImageSubAgent,
    ImageDecision,
    create_image_agent,
)
from huesaeagents.huesae.agents.subagents.image import (
    generate_tags,
    expand_prompt,
    DoubaoProvider,
)
from huesaeagents.huesae.models.models_factory import create_chat_model


# ============== 夹具 ==============

@pytest.fixture(scope="module")
def llm():
    """共享的LLM实例"""
    return create_chat_model("deepseek")


@pytest.fixture(scope="module")
def main_agent(llm):
    """共享的主Agent实例"""
    agent = HuesaeMainAgent(llm=llm)
    agent.register_sub_agent(create_image_agent(llm=llm))
    return agent


@pytest.fixture(scope="module")
def image_agent(llm):
    """共享的ImageSubAgent实例（不调用实际API）"""
    return ImageSubAgent(llm=llm, providers=[])


# ============== 测试：主Agent工具和子Agent注册 ==============

class TestMainAgentHarness:
    """测试主Agent的工具注册和子Agent委派能力"""

    def test_tools_are_available(self, main_agent):
        """测试主Agent暴露当前工具列表"""
        tool_names = {tool.name for tool in main_agent.tools}
        assert "generate_image_tool" in tool_names
        assert "generate_images_tool" in tool_names
        assert "task_tool" in tool_names

    def test_image_subagent_registered(self, main_agent):
        """测试生图子Agent已注册到注册表"""
        assert main_agent.subagent_registry.get("image") is not None


# ============== 测试：子Agent标准化接口 ==============

class TestImageSubAgent:
    """测试 ImageSubAgent 标准化接口"""

    def test_ask_prompt_when_no_description(self, image_agent):
        """测试：用户只说'我想生成图片'，应该追问"""
        result = image_agent.process({}, "我想生成图片")

        assert result["action"] == "ask_prompt"
        assert "请告诉我" in result["response"] or "描述" in result["response"]
        print(f"追问: {result['response']}")

    def test_generate_when_has_prompt(self, image_agent):
        """测试：用户提供了明确提示词"""
        state = {
            "messages": [
                HumanMessage(content="我想生成图片"),
                AIMessage(content="请告诉我您想要生成什么样的图片？"),
                HumanMessage(content="夕阳下看大海的少女，穿着水手服"),
            ],
        }
        result = image_agent.process(state, "夕阳下看大海的少女，穿着水手服")

        # 应该进入generate或ask_confirm
        assert result["action"] in ("generate", "ask_confirm")
        print(f"有提示词: action={result['action']}, prompt={result.get('prompt')}")

    def test_recommend_when_asked(self, image_agent):
        """测试：用户要求推荐"""
        state = {
            "messages": [
                HumanMessage(content="我想生成图片"),
                AIMessage(content="请描述一下？"),
                HumanMessage(content="你帮我推荐一些吧"),
            ],
        }
        result = image_agent.process(state, "你帮我推荐一些吧")

        assert result["action"] == "recommend"
        print(f"推荐: {result['response'][:60]}...")

    def test_expand_when_asked(self, image_agent):
        """测试：用户要求扩写"""
        state = {
            "messages": [HumanMessage(content="夏天的图")],
            "image_prompt": "夏天的图",
        }
        result = image_agent.process(state, "你帮我扩展一下吧")

        # 扩写后进入ask_confirm
        assert result["action"] == "ask_confirm"
        assert "扩写" in result["response"]
        print(f"扩写: {result['response'][:60]}...")

    def test_finish_when_satisfied(self, image_agent):
        """测试：用户满意后，子Agent返回finish"""
        state = {
            "messages": [
                HumanMessage(content="我想生成图片"),
                AIMessage(content="图片已生成完成，您满意吗？"),
            ],
        }
        result = image_agent.process(state, "真好看，谢谢")

        # 应该返回finish（子Agent认为对话结束）
        assert result["action"] == "finish"
        print(f"结束: {result['response'][:60]}...")

    def test_decide_structured_output(self, image_agent):
        """测试：LLM决策结构化输出"""
        decision = image_agent._decide({}, "画一个猫娘")

        assert isinstance(decision, ImageDecision)
        assert decision.action in (
            "ask_prompt", "recommend", "expand",
            "ask_confirm", "generate", "show_image", "finish",
        )
        assert decision.response is not None
        print(f"决策: action={decision.action}, prompt={decision.prompt}")

    def test_standardized_result_format(self, image_agent):
        """测试：返回结果符合标准化格式"""
        result = image_agent.process({}, "我想生成图片")

        assert "action" in result
        assert "response" in result
        assert "prompt" in result
        assert "provider" in result
        assert "data" in result
        print(f"标准化结果: {list(result.keys())}")


# ============== 测试：主Agent集成 ==============

class TestMainAgentIntegration:
    """测试主Agent集成（聊天 + 委派子Agent）"""

    def test_chat_directly(self, main_agent):
        """测试：主Agent直接聊天回复"""
        result = main_agent.process({"messages": []}, "你好")

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        print(f"聊天回复: {result['messages'][0].content[:60]}...")

    def test_delegate_to_image_agent(self, main_agent):
        """测试：主Agent委派给生图Agent"""
        result = main_agent.process({"messages": []}, "我想生成图片")

        # 应该返回子Agent的追问
        assert len(result["messages"]) == 1
        assert "请告诉我" in result["messages"][0].content or "描述" in result["messages"][0].content
        print(f"委派生图: {result['messages'][0].content[:60]}...")

    def test_chat_after_image_generation(self, main_agent):
        """测试：生图对话历史存在时，用户说无关内容，主Agent直接聊天"""
        messages = [
            HumanMessage(content="我想生成图片"),
            AIMessage(content="请告诉我您想要生成什么样的图片？"),
            HumanMessage(content="夕阳下看大海的少女"),
            AIMessage(content="图片已生成完成"),
        ]
        result = main_agent.process({"messages": messages}, "真好看")

        # 主Agent应该直接聊天回复，不进入生图Agent
        assert "image_url" not in result
        assert len(result["messages"]) == 1
        print(f"生图后聊天: {result['messages'][0].content[:60]}...")


# ============== 测试：独立功能模块 ==============

class TestDanbooruTags:
    """测试 Danbooru 标签生成"""

    def test_generate_tags(self, llm):
        """测试生成Danbooru标签"""
        tags = generate_tags("一个银发红瞳的少女在樱花树下", llm)
        assert isinstance(tags, list)
        assert len(tags) > 0
        print(f"生成的标签: {tags[:10]}")

    def test_tags_to_prompt(self):
        """测试标签拼接为提示词"""
        from huesaeagents.huesae.agents.subagents.image import tags_to_prompt
        tags = ["1girl", "silver hair", "red eyes"]
        prompt = tags_to_prompt(tags)
        assert prompt == "1girl, silver hair, red eyes"


class TestExpandPrompt:
    """测试提示词扩写"""

    def test_expand_prompt(self, llm):
        """测试扩写提示词"""
        expanded = expand_prompt("夕阳下的战舰", llm)
        assert isinstance(expanded, str)
        assert len(expanded) > len("夕阳下的战舰")
        print(f"扩写结果: {expanded[:80]}...")


class TestProviders:
    """测试 Provider 注册"""

    def test_register_provider(self, llm):
        """测试注册Provider"""
        agent = ImageSubAgent(llm=llm, providers=[])
        assert len(agent.providers) == 0

        agent.register_provider(DoubaoProvider())
        assert "doubao" in agent.providers

    def test_default_provider(self, llm):
        """测试默认Provider"""
        agent = create_image_agent(llm=llm)
        names = agent.get_available_providers()
        assert "doubao" in names


# ============== 辅助函数：直接运行测试 ==============

def run_all_tests():
    """运行核心测试（不需要pytest）"""
    llm = create_chat_model("deepseek")

    print("=" * 60)
    print("HuesaeAgents 架构重构测试")
    print("=" * 60)

    # 1. 主Agent工具注册
    print("\n--- 测试1: 主Agent工具注册 ---")
    main = HuesaeMainAgent(llm=llm)
    main.register_sub_agent(create_image_agent(llm=llm))

    tool_names = {tool.name for tool in main.tools}
    print(f"✓ 可用工具: {sorted(tool_names)}")
    assert "task_tool" in tool_names
    assert main.subagent_registry.get("image") is not None

    # 2. 子Agent标准化接口
    print("\n--- 测试2: 子Agent标准化接口 ---")
    image_agent = ImageSubAgent(llm=llm, providers=[])
    result = image_agent.process({}, "我想生成图片")
    print(f"✓ 追问: action={result['action']}")
    assert "action" in result and "response" in result

    # 3. 主Agent集成
    print("\n--- 测试3: 主Agent集成 ---")
    result = main.process({"messages": []}, "你好")
    print(f"✓ 直接聊天: {result['messages'][0].content[:40]}...")

    result = main.process({"messages": []}, "我想生成图片")
    print(f"✓ 委派生图: {result['messages'][0].content[:40]}...")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
