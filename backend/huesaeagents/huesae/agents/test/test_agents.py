"""Agent 测试

测试内容：
1. 主图意图识别（LLM粗分类 + 子图保持）
2. 子图对话管理器（追问、推荐、扩写、确认、生图）
3. Graph多轮对话流程
4. Provider注册
"""
import sys
from pathlib import Path

# 将 backend 目录添加到 Python 路径
backend_dir = Path(__file__).resolve().parents[4]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from huesaeagents.huesae.agents.subagents.image_agent import (
    ImageConversationManager,
    ImageDecision,
    create_image_agent,
)
from huesaeagents.huesae.agents.subagents.image import (
    generate_tags,
    expand_prompt,
    DoubaoProvider,
    JimengProvider,
)
from huesaeagents.huesae.models.models_factory import create_chat_model
from huesaeagents.huesae.graph.conditional_logic import (
    classify_intent,
    route_by_intent,
    Intent,
)


# ============== 夹具 ==============

@pytest.fixture(scope="module")
def llm():
    """共享的LLM实例"""
    return create_chat_model("deepseek")


@pytest.fixture(scope="module")
def image_manager(llm):
    """共享的ImageConversationManager实例（不调用实际API）"""
    return ImageConversationManager(llm=llm, providers=[])


# ============== 测试：主图意图识别 ==============

class TestIntentRecognition:
    """测试主图LLM粗分类意图识别"""

    def test_classify_image_intent(self):
        """测试识别生图意图"""
        state = {
            "messages": [HumanMessage(content="我想生成图片")],
            "image_step": None,
        }
        result = classify_intent(state)
        assert result == Intent.IMAGE
        print(f"意图识别: {result}")

    def test_classify_chat_intent(self):
        """测试识别普通对话意图"""
        state = {
            "messages": [HumanMessage(content="今天天气怎么样？")],
            "image_step": None,
        }
        result = classify_intent(state)
        assert result == Intent.CHAT
        print(f"意图识别: {result}")

    def test_keep_image_intent_in_conversation(self):
        """测试子图对话中保持IMAGE意图"""
        state = {
            "messages": [HumanMessage(content="夕阳下看大海的少女")],
            "image_step": "ask_prompt",
        }
        result = classify_intent(state)
        assert result == Intent.IMAGE
        print(f"子图保持: {result}")

    def test_route_by_intent(self):
        """测试条件路由"""
        assert route_by_intent({"intent": Intent.CHAT}) == "chat_agent"
        assert route_by_intent({"intent": Intent.IMAGE}) == "image_agent"
        assert route_by_intent({"intent": Intent.VOICE}) == "voice_agent"


# ============== 测试：子图对话管理器 ==============

class TestImageConversationManager:
    """测试 ImageConversationManager 对话流程"""

    def test_ask_prompt_when_no_description(self, image_manager):
        """测试：用户只说'我想生成图片'，应该追问"""
        result = image_manager.process({}, "我想生成图片")

        assert result["image_step"] == "ask_prompt"
        assert "请告诉我" in result["messages"][0].content or "描述" in result["messages"][0].content
        print(f"追问: {result['messages'][0].content}")

    def test_generate_when_has_prompt(self, image_manager):
        """测试：用户提供了明确提示词"""
        state = {
            "messages": [
                HumanMessage(content="我想生成图片"),
                AIMessage(content="请告诉我您想要生成什么样的图片？"),
                HumanMessage(content="夕阳下看大海的少女，穿着水手服"),
            ],
        }
        result = image_manager.process(state, "夕阳下看大海的少女，穿着水手服")

        # 应该进入generate或ask_confirm
        assert result["image_step"] in ("generate", "ask_confirm")
        print(f"有提示词: step={result['image_step']}, prompt={result.get('image_prompt')}")

    def test_recommend_when_asked(self, image_manager):
        """测试：用户要求推荐"""
        state = {
            "messages": [
                HumanMessage(content="我想生成图片"),
                AIMessage(content="请描述一下？"),
                HumanMessage(content="你帮我推荐一些吧"),
            ],
        }
        result = image_manager.process(state, "你帮我推荐一些吧")

        assert result["image_step"] == "recommend"
        print(f"推荐: {result['messages'][0].content[:60]}...")

    def test_expand_when_asked(self, image_manager):
        """测试：用户要求扩写"""
        state = {
            "messages": [HumanMessage(content="夏天的图")],
            "image_prompt": "夏天的图",
        }
        result = image_manager.process(state, "你帮我扩展一下吧")

        # 扩写后进入ask_confirm
        assert result["image_step"] == "ask_confirm"
        assert "扩写" in result["messages"][0].content
        print(f"扩写: {result['messages'][0].content[:60]}...")

    def test_decide_structured_output(self, image_manager):
        """测试：LLM决策结构化输出"""
        decision = image_manager._decide({}, "画一个猫娘")

        assert isinstance(decision, ImageDecision)
        assert decision.action in (
            "ask_prompt", "recommend", "expand",
            "ask_confirm", "generate", "show_image", "finish",
        )
        assert decision.response is not None
        print(f"决策: action={decision.action}, prompt={decision.prompt}")


# ============== 测试：Graph 多轮对话 ==============

class TestGraphMultiTurn:
    """测试 Graph 级别的多轮对话"""

    def test_graph_first_turn_ask_prompt(self):
        """测试 Graph 第一轮：用户说'我想生成图片'→应该追问"""
        from huesaeagents.huesae.graph.huesae_graph import create_huesae_graph

        graph = create_huesae_graph()
        config = {"configurable": {"thread_id": "test_user_1"}}

        result = graph.invoke(
            {"messages": [HumanMessage(content="我想生成图片")]},
            config=config,
        )

        assert result.get("image_step") == "ask_prompt"
        ai_msg = result["messages"][-1]
        assert "请告诉我" in ai_msg.content or "描述" in ai_msg.content
        print(f"Graph第一轮: {ai_msg.content[:60]}...")

    def test_graph_second_turn_generate(self):
        """测试 Graph 第二轮：用户提供提示词"""
        from huesaeagents.huesae.graph.huesae_graph import create_huesae_graph

        graph = create_huesae_graph()
        config = {"configurable": {"thread_id": "test_user_2"}}

        # 第一轮
        graph.invoke(
            {"messages": [HumanMessage(content="我想生成图片")]},
            config=config,
        )

        # 第二轮：提供提示词
        result = graph.invoke(
            {"messages": [HumanMessage(content="夕阳下看大海的少女，穿着水手服")]},
            config=config,
        )

        assert result.get("image_step") in ("generate", "ask_confirm")
        print(f"Graph第二轮: step={result.get('image_step')}")

    def test_graph_keep_intent_in_conversation(self):
        """测试：子图对话中，主图保持IMAGE意图"""
        from huesaeagents.huesae.graph.huesae_graph import create_huesae_graph

        graph = create_huesae_graph()
        config = {"configurable": {"thread_id": "test_user_3"}}

        # 第一轮：进入生图
        graph.invoke(
            {"messages": [HumanMessage(content="我想生成图片")]},
            config=config,
        )

        # 第二轮：提供一个没有生图关键词的描述
        result = graph.invoke(
            {"messages": [HumanMessage(content="夕阳下看大海的少女")]},
            config=config,
        )

        assert result.get("image_step") in ("generate", "ask_confirm", "ask_prompt")
        print(f"意图保持: step={result.get('image_step')}")


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
        manager = ImageConversationManager(llm=llm, providers=[])
        assert len(manager.providers) == 0

        manager.register_provider(DoubaoProvider())
        assert "doubao" in manager.providers

        manager.register_provider(JimengProvider())
        assert "jimeng" in manager.providers
        assert len(manager.providers) == 2

    def test_available_providers(self, llm):
        """测试获取可用Provider列表"""
        manager = ImageConversationManager(
            llm=llm,
            providers=[DoubaoProvider(), JimengProvider()],
        )
        names = manager.get_available_providers()
        assert "doubao" in names
        assert "jimeng" in names


# ============== 辅助函数：直接运行测试 ==============

def run_all_tests():
    """运行核心测试（不需要pytest）"""
    llm = create_chat_model("deepseek")
    manager = ImageConversationManager(llm=llm, providers=[])

    print("=" * 60)
    print("ImageConversationManager 测试")
    print("=" * 60)

    # 1. 意图保持
    print("\n--- 测试1: 子图对话中保持意图 ---")
    state = {"messages": [HumanMessage(content="我想生成图片")], "image_step": "ask_prompt"}
    result = classify_intent(state)
    print(f"✓ 子图保持: {result}")

    # 2. 追问
    print("\n--- 测试2: 缺少提示词时追问 ---")
    result = manager.process({}, "我想生成图片")
    print(f"✓ 追问: {result['messages'][0].content[:60]}...")

    # 3. 有提示词
    print("\n--- 测试3: 有明确提示词 ---")
    state = {
        "messages": [
            HumanMessage(content="我想生成图片"),
            AIMessage(content="请描述一下？"),
            HumanMessage(content="夕阳下看大海的少女"),
        ],
    }
    result = manager.process(state, "夕阳下看大海的少女")
    print(f"✓ 有提示词: step={result['image_step']}, prompt={result.get('image_prompt')}")

    # 4. Graph多轮
    print("\n--- 测试4: Graph 多轮对话 ---")
    from huesaeagents.huesae.graph.huesae_graph import create_huesae_graph

    graph = create_huesae_graph()
    config = {"configurable": {"thread_id": "demo_test"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="我想生成图片")]},
        config=config,
    )
    print(f"✓ 第一轮: step={result.get('image_step')}")

    result = graph.invoke(
        {"messages": [HumanMessage(content="夕阳下看大海的少女")]},
        config=config,
    )
    print(f"✓ 第二轮: step={result.get('image_step')}, prompt={result.get('image_prompt')}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
