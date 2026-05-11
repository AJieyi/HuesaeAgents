"""Agent 测试

测试内容：
1. 意图识别（LLM结构化输出）
2. ImageAgent 三种模式（直接生图、转Danbooru标签、扩写提示词）
3. 多轮对话流程（Graph级别）
4. Provider 注册和调用
"""
import sys
from pathlib import Path

# 将 backend 目录添加到 Python 路径
backend_dir = Path(__file__).resolve().parents[4]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import pytest
import asyncio
from langchain_core.messages import HumanMessage

from huesaeagents.huesae.agents.subagents.image_agent import (
    ImageAgent,
    ImageMode,
    ImageStep,
    create_image_agent,
)
from huesaeagents.huesae.agents.subagents.image.intent import (
    recognize_intent,
    ImageIntent,
)
from huesaeagents.huesae.agents.subagents.image import (
    generate_tags,
    expand_prompt,
    DoubaoProvider,
    JimengProvider,
)
from huesaeagents.huesae.models.models_factory import create_chat_model


# ============== 夹具 ==============

@pytest.fixture(scope="module")
def llm():
    """共享的LLM实例"""
    return create_chat_model("deepseek")


@pytest.fixture(scope="module")
def image_agent(llm):
    """共享的ImageAgent实例（不调用实际API）"""
    return ImageAgent(llm=llm, providers=[])


# ============== 测试：意图识别 ==============

class TestIntentRecognition:
    """测试LLM意图识别"""

    def test_recognize_direct_image(self, llm):
        """测试识别直接生图意图"""
        result = recognize_intent("画一个银发红瞳的少女在河边", llm)
        assert isinstance(result, ImageIntent)
        assert result.intent == ImageMode.DIRECT_IMAGE
        assert "银发" in result.extracted_prompt
        assert "红瞳" in result.extracted_prompt
        print(f"意图识别: {result.intent}, 提示词: {result.extracted_prompt}")

    def test_recognize_convert_tags(self, llm):
        """测试识别转Danbooru标签意图"""
        result = recognize_intent("把这句话转成Danbooru标签：一个猫娘在咖啡馆", llm)
        assert isinstance(result, ImageIntent)
        assert result.intent == ImageMode.CONVERT_TAGS
        assert "猫娘" in result.extracted_prompt or "咖啡馆" in result.extracted_prompt
        print(f"意图识别: {result.intent}, 提示词: {result.extracted_prompt}")

    def test_recognize_expand_prompt(self, llm):
        """测试识别扩写提示词意图"""
        result = recognize_intent("扩写：夕阳下的战舰", llm)
        assert isinstance(result, ImageIntent)
        assert result.intent == ImageMode.EXPAND_PROMPT
        assert "夕阳" in result.extracted_prompt or "战舰" in result.extracted_prompt
        print(f"意图识别: {result.intent}, 提示词: {result.extracted_prompt}")

    def test_recognize_chat(self, llm):
        """测试识别普通对话意图"""
        result = recognize_intent("今天天气怎么样？", llm)
        assert isinstance(result, ImageIntent)
        assert result.intent == ImageMode.CHAT
        print(f"意图识别: {result.intent}")


# ============== 测试：ImageAgent 流程 ==============

class TestImageAgentFlow:
    """测试 ImageAgent 各模式流程"""

    # ---- 模式A：直接生图 ----

    def test_direct_image_input_short(self, image_agent):
        """测试直接生图 - 提示词太短，需要补充"""
        result = image_agent.process_input("画一个猫")
        assert result["step"] == ImageStep.INPUT
        assert result["need_more_input"] is True
        assert "太短" in result["message"] or "详细" in result["message"]
        print(f"短提示词处理: {result['message']}")

    def test_direct_image_input_ok(self, image_agent):
        """测试直接生图 - 正常提示词，进入选择工具"""
        result = image_agent.process_input("画一个银发红瞳的少女在樱花树下")
        assert result["step"] == ImageStep.SELECT_TOOL
        assert result["mode"] == ImageMode.DIRECT_IMAGE
        assert result["need_more_input"] is False
        assert "请选择" in result["message"] or "工具" in result["message"]
        print(f"正常提示词: {result['message']}")

    def test_direct_image_select_tool(self, image_agent):
        """测试选择生图工具"""
        # 先进入选择工具状态
        state = {
            "step": ImageStep.SELECT_TOOL,
            "mode": ImageMode.DIRECT_IMAGE,
            "prompt": "银发红瞳的少女",
        }
        result = image_agent.process_step(state, "doubao")
        assert result["step"] == ImageStep.GENERATE_IMAGE
        assert result["selected_provider"] == "doubao"
        print(f"选择工具: {result['message']}")

    def test_direct_image_show_image_ok(self, image_agent):
        """测试展示图片后用户说可以"""
        state = {
            "step": ImageStep.SHOW_IMAGE,
            "mode": ImageMode.DIRECT_IMAGE,
            "prompt": "银发红瞳的少女",
        }
        result = image_agent.process_step(state, "可以了")
        assert result["step"] == ImageStep.FINISH
        print(f"用户确认: {result['message']}")

    def test_direct_image_show_image_regenerate(self, image_agent):
        """测试展示图片后用户要求换一张"""
        state = {
            "step": ImageStep.SHOW_IMAGE,
            "mode": ImageMode.DIRECT_IMAGE,
            "prompt": "银发红瞳的少女",
        }
        result = image_agent.process_step(state, "换一张")
        assert result["step"] == ImageStep.GENERATE_IMAGE
        assert result["selected_provider"] == "doubao"
        print(f"用户换图: {result['message']}")

    # ---- 模式B：转Danbooru标签 ----

    def test_convert_tags_input(self, image_agent):
        """测试转Danbooru标签 - 初始输入"""
        result = image_agent.process_input("转成Danbooru标签：一个猫娘在咖啡馆")
        assert result["step"] == ImageStep.SHOW_TAGS
        assert result["mode"] == ImageMode.CONVERT_TAGS
        assert "danbooru_tags" in result
        print(f"标签生成: {result['message'][:50]}...")

    def test_convert_tags_regenerate(self, image_agent):
        """测试标签不满意，重新生成"""
        state = {
            "step": ImageStep.SHOW_TAGS,
            "mode": ImageMode.CONVERT_TAGS,
            "prompt": "一个猫娘在咖啡馆",
        }
        result = image_agent.process_step(state, "换一版")
        assert result["step"] == ImageStep.SHOW_TAGS
        assert "danbooru_tags" in result
        print(f"重新生成标签: {result['message'][:50]}...")

    def test_convert_tags_finish(self, image_agent):
        """测试标签满意，结束"""
        state = {
            "step": ImageStep.SHOW_TAGS,
            "mode": ImageMode.CONVERT_TAGS,
            "prompt": "一个猫娘在咖啡馆",
        }
        result = image_agent.process_step(state, "可以了")
        assert result["step"] == ImageStep.FINISH
        print(f"标签确认: {result['message']}")

    # ---- 模式C：扩写提示词 ----

    def test_expand_prompt_input(self, image_agent):
        """测试扩写提示词 - 初始输入"""
        result = image_agent.process_input("扩写：夕阳下的战舰")
        assert result["step"] == ImageStep.SHOW_EXPANDED
        assert result["mode"] == ImageMode.EXPAND_PROMPT
        assert "expanded_prompt" in result
        print(f"扩写结果: {result['expanded_prompt'][:50]}...")

    def test_expand_prompt_regenerate(self, image_agent):
        """测试扩写不满意，重新扩写"""
        state = {
            "step": ImageStep.SHOW_EXPANDED,
            "mode": ImageMode.EXPAND_PROMPT,
            "prompt": "夕阳下的战舰",
        }
        result = image_agent.process_step(state, "再写一版")
        assert result["step"] == ImageStep.SHOW_EXPANDED
        assert "expanded_prompt" in result
        print(f"重新扩写: {result['expanded_prompt'][:50]}...")

    def test_expand_prompt_finish(self, image_agent):
        """测试扩写满意，结束"""
        state = {
            "step": ImageStep.SHOW_EXPANDED,
            "mode": ImageMode.EXPAND_PROMPT,
            "prompt": "夕阳下的战舰",
        }
        result = image_agent.process_step(state, "接受")
        assert result["step"] == ImageStep.FINISH
        print(f"扩写确认: {result['message']}")

    # ---- 边界情况 ----

    def test_need_more_input_then_continue(self, image_agent):
        """测试补充输入后继续流程"""
        # 第一次输入太短
        result1 = image_agent.process_input("画一个猫")
        assert result1["need_more_input"] is True

        # 补充输入
        state = {
            "step": ImageStep.INPUT,
            "mode": ImageMode.DIRECT_IMAGE,
            "prompt": "",
            "need_more_input": True,
        }
        result2 = image_agent.process_step(state, "画一个银发红瞳的少女在樱花树下")
        assert result2["step"] == ImageStep.SELECT_TOOL
        print(f"补充输入后: {result2['message']}")

    def test_chat_fallback(self, image_agent):
        """测试非生图意图的fallback"""
        result = image_agent.process_input("你好，今天过得怎么样？")
        assert result["mode"] == ImageMode.CHAT
        assert result["step"] == ImageStep.FINISH
        print(f"Chat fallback: {result['message']}")


# ============== 测试：Graph 多轮对话 ==============

class TestGraphMultiTurn:
    """测试 Graph 级别的多轮对话"""

    def test_graph_first_turn(self):
        """测试 Graph 第一轮：用户输入生图请求"""
        from huesaeagents.huesae.graph.huesae_graph import create_huesae_graph

        graph = create_huesae_graph()
        config = {"configurable": {"thread_id": "test_user_1"}}

        result = graph.invoke(
            {"messages": [HumanMessage(content="画一个银发红瞳的少女")]},
            config=config,
        )

        # 应该进入image_agent，步骤为 SELECT_TOOL
        assert result.get("image_step") == ImageStep.SELECT_TOOL
        assert result.get("image_mode") == ImageMode.DIRECT_IMAGE
        ai_msg = result["messages"][-1]
        assert "请选择" in ai_msg.content or "工具" in ai_msg.content
        print(f"Graph第一轮: {ai_msg.content[:60]}...")

    def test_graph_second_turn(self):
        """测试 Graph 第二轮：用户选择工具"""
        from huesaeagents.huesae.graph.huesae_graph import create_huesae_graph

        graph = create_huesae_graph()
        config = {"configurable": {"thread_id": "test_user_2"}}

        # 第一轮
        graph.invoke(
            {"messages": [HumanMessage(content="画一个银发红瞳的少女")]},
            config=config,
        )

        # 第二轮：选择工具
        result = graph.invoke(
            {"messages": [HumanMessage(content="doubao")]},
            config=config,
        )

        assert result.get("image_step") == ImageStep.GENERATE_IMAGE
        assert result.get("selected_provider") == "doubao"
        ai_msg = result["messages"][-1]
        assert "doubao" in ai_msg.content.lower() or "豆包" in ai_msg.content
        print(f"Graph第二轮: {ai_msg.content[:60]}...")

    def test_graph_convert_tags_flow(self):
        """测试 Graph 转Danbooru标签完整流程"""
        from huesaeagents.huesae.graph.huesae_graph import create_huesae_graph

        graph = create_huesae_graph()
        config = {"configurable": {"thread_id": "test_user_3"}}

        # 第一轮：请求转标签
        result1 = graph.invoke(
            {"messages": [HumanMessage(content="转成Danbooru标签：一个猫娘")]},
            config=config,
        )
        assert result1.get("image_step") == ImageStep.SHOW_TAGS
        assert result1.get("image_mode") == ImageMode.CONVERT_TAGS
        assert result1.get("danbooru_tags") is not None

        # 第二轮：确认
        result2 = graph.invoke(
            {"messages": [HumanMessage(content="可以了")]},
            config=config,
        )
        assert result2.get("image_step") == ImageStep.FINISH
        print(f"标签流程完成: {result2['messages'][-1].content[:60]}...")

    def test_graph_expand_prompt_flow(self):
        """测试 Graph 扩写提示词完整流程"""
        from huesaeagents.huesae.graph.huesae_graph import create_huesae_graph

        graph = create_huesae_graph()
        config = {"configurable": {"thread_id": "test_user_4"}}

        # 第一轮：请求扩写
        result1 = graph.invoke(
            {"messages": [HumanMessage(content="扩写：夕阳下的战舰")]},
            config=config,
        )
        assert result1.get("image_step") == ImageStep.SHOW_EXPANDED
        assert result1.get("image_mode") == ImageMode.EXPAND_PROMPT
        assert result1.get("expanded_prompt") is not None

        # 第二轮：确认
        result2 = graph.invoke(
            {"messages": [HumanMessage(content="接受")]},
            config=config,
        )
        assert result2.get("image_step") == ImageStep.FINISH
        print(f"扩写流程完成: {result2['messages'][-1].content[:60]}...")


# ============== 测试：独立功能模块 ==============

class TestDanbooruTags:
    """测试 Danbooru 标签生成"""

    def test_generate_tags(self, llm):
        """测试生成Danbooru标签"""
        tags = generate_tags("一个银发红瞳的少女在樱花树下", llm)
        assert isinstance(tags, list)
        assert len(tags) > 0
        # 应该包含一些常见标签
        tags_lower = [t.lower() for t in tags]
        assert any("girl" in t or "1girl" in t for t in tags_lower)
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
        assert "夕阳" in expanded or "战舰" in expanded
        print(f"扩写结果: {expanded[:80]}...")


class TestProviders:
    """测试 Provider 注册"""

    def test_register_provider(self, llm):
        """测试注册Provider"""
        agent = ImageAgent(llm=llm, providers=[])
        assert len(agent.providers) == 0

        agent.register_provider(DoubaoProvider())
        assert "doubao" in agent.providers

        agent.register_provider(JimengProvider())
        assert "jimeng" in agent.providers
        assert len(agent.providers) == 2

    def test_available_providers(self, llm):
        """测试获取可用Provider列表"""
        agent = ImageAgent(
            llm=llm,
            providers=[DoubaoProvider(), JimengProvider()],
        )
        names = agent.get_available_providers()
        assert "doubao" in names
        assert "jimeng" in names


# ============== 辅助函数：直接运行测试 ==============

def run_all_tests():
    """运行所有测试（不需要pytest）"""
    llm = create_chat_model("deepseek")
    agent = ImageAgent(llm=llm, providers=[])

    print("=" * 60)
    print("ImageAgent 测试")
    print("=" * 60)

    # 1. 意图识别
    print("\n--- 测试1: 意图识别 ---")
    for text, expected in [
        ("画一个银发红瞳的少女", ImageMode.DIRECT_IMAGE),
        ("转成Danbooru标签：猫娘", ImageMode.CONVERT_TAGS),
        ("扩写：夕阳下的战舰", ImageMode.EXPAND_PROMPT),
        ("今天天气怎么样", ImageMode.CHAT),
    ]:
        try:
            intent = recognize_intent(text, llm)
            status = "✓" if intent.intent == expected else "✗"
            print(f"{status} '{text}' → {intent.intent} (期望: {expected})")
        except Exception as e:
            print(f"✗ '{text}' → 错误: {e}")

    # 2. 直接生图流程
    print("\n--- 测试2: 直接生图流程 ---")
    result = agent.process_input("画一个银发红瞳的少女在樱花树下")
    print(f"✓ 输入处理: step={result['step']}, mode={result['mode']}")

    state = {
        "step": ImageStep.SELECT_TOOL,
        "mode": ImageMode.DIRECT_IMAGE,
        "prompt": "银发红瞳的少女",
    }
    result = agent.process_step(state, "doubao")
    print(f"✓ 选择工具: step={result['step']}, provider={result.get('selected_provider')}")

    # 3. 转标签流程
    print("\n--- 测试3: 转Danbooru标签流程 ---")
    result = agent.process_input("转成Danbooru标签：一个猫娘在咖啡馆")
    print(f"✓ 标签生成: step={result['step']}, tags数={len(result.get('danbooru_tags', []))}")

    # 4. 扩写流程
    print("\n--- 测试4: 扩写提示词流程 ---")
    result = agent.process_input("扩写：夕阳下的战舰")
    print(f"✓ 扩写完成: step={result['step']}")
    print(f"  扩写内容: {result.get('expanded_prompt', '')[:60]}...")

    # 5. 多轮对话（Graph级别）
    print("\n--- 测试5: Graph 多轮对话 ---")
    from huesaeagents.huesae.graph.huesae_graph import create_huesae_graph

    graph = create_huesae_graph()
    config = {"configurable": {"thread_id": "demo_test"}}

    # 第一轮
    result = graph.invoke(
        {"messages": [HumanMessage(content="画一个银发红瞳的少女")]},
        config=config,
    )
    print(f"✓ 第一轮: step={result.get('image_step')}")

    # 第二轮
    result = graph.invoke(
        {"messages": [HumanMessage(content="doubao")]},
        config=config,
    )
    print(f"✓ 第二轮: step={result.get('image_step')}, provider={result.get('selected_provider')}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
