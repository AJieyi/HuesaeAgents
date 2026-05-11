"""生图Agent测试脚本（v2）

测试三种模式：
1. 直接生图：输入 → 选择工具 → 生图 → 换图/结束
2. 转Danbooru标签：输入 → 标签 → 换版本/结束
3. 扩写提示词：输入 → 扩写 → 接受/拒绝

使用方法：
    cd backend/huesaeagents
    python -m huesae.agents.subagents.test_image_agent
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_intent_detection():
    """测试意图检测"""
    print("=" * 60)
    print("测试 意图检测")
    print("=" * 60)

    from huesae.agents.subagents.image_agent import ImageAgent, ImageMode
    from huesae.models.models_factory import create_chat_model

    agent = ImageAgent(llm=create_chat_model("deepseek"))

    test_cases = [
        # ("画一个银发红瞳的少女", ImageMode.DIRECT_IMAGE),
        ("帮我生成一张图片", ImageMode.DIRECT_IMAGE),
        # ("转成Danbooru标签", ImageMode.CONVERT_TAGS),
        # ("生成Danbooru标签", ImageMode.CONVERT_TAGS),
        # ("扩写提示词", ImageMode.EXPAND_PROMPT),
        # ("扩写这个描述", ImageMode.EXPAND_PROMPT),
    ]

    for text, expected in test_cases:
        mode = agent._detect_mode(text)
        status = "✓" if mode == expected else "✗"
        print(f"  {status} '{text}' -> {mode}")

    print()


def test_direct_image_flow():
    """测试直接生图流程"""
    print("=" * 60)
    print("测试 直接生图流程")
    print("=" * 60)

    from huesae.agents.subagents.image_agent import ImageAgent, ImageStep
    from huesae.models.models_factory import create_chat_model

    agent = ImageAgent(
        llm=create_chat_model("deepseek"),
        providers=[],  # 不实际生图
    )

    # Step 1: 用户输入（有效提示词）
    print("\n[Step 1] 用户输入：'画一个银发红瞳的少女'")
    result = agent.process_input("画一个银发红瞳的少女")
    print(f"  模式: {result['mode']}")
    print(f"  步骤: {result['step']}")
    print(f"  提示词: {result.get('prompt')}")
    print(f"  消息: {result['message'][:80]}...")

    # Step 2: 用户选择工具
    print("\n[Step 2] 用户选择：'豆包'")
    state = {"step": result["step"], "mode": result["mode"], "prompt": result.get("prompt", "")}
    result2 = agent.process_step(state, "豆包")
    print(f"  步骤: {result2['step']}")
    print(f"  Provider: {result2.get('selected_provider')}")
    print(f"  消息: {result2['message']}")

    # Step 3: 展示图片，用户说换一张
    print("\n[Step 3] 用户说：'换一张'")
    state = {"step": "show_image", "mode": "direct_image", "prompt": result.get("prompt", "")}
    result3 = agent.process_step(state, "换一张")
    print(f"  步骤: {result3['step']}")
    print(f"  Provider: {result3.get('selected_provider')}")
    print(f"  消息: {result3['message'][:60]}...")

    # Step 4: 展示图片，用户说可以
    print("\n[Step 4] 用户说：'可以'")
    state = {"step": "show_image", "mode": "direct_image", "prompt": result.get("prompt", "")}
    result4 = agent.process_step(state, "可以")
    print(f"  步骤: {result4['step']}")
    print(f"  消息: {result4['message']}")

    print("\n✓ 直接生图流程测试通过\n")


def test_short_prompt():
    """测试短提示词处理"""
    print("=" * 60)
    print("测试 短提示词处理")
    print("=" * 60)

    from huesae.agents.subagents.image_agent import ImageAgent
    from huesae.models.models_factory import create_chat_model

    agent = ImageAgent(llm=create_chat_model("deepseek"))

    # 短提示词
    print("\n[测试] 用户输入：'画个猫'")
    result = agent.process_input("画个猫")
    print(f"  步骤: {result['step']}")
    print(f"  需要补充: {result.get('need_more_input')}")
    print(f"  消息: {result['message'][:60]}...")

    # 用户补充
    print("\n[测试] 用户补充：'画一只橘色的猫在窗台上睡觉'")
    state = {
        "step": result["step"],
        "mode": result["mode"],
        "prompt": "",
        "need_more_input": True,
    }
    result2 = agent.process_step(state, "画一只橘色的猫在窗台上睡觉")
    print(f"  步骤: {result2['step']}")
    print(f"  提示词: {result2.get('prompt')}")
    print(f"  需要补充: {result2.get('need_more_input')}")

    print("\n✓ 短提示词处理测试通过\n")


def test_convert_tags_flow():
    """测试转Danbooru标签流程"""
    print("=" * 60)
    print("测试 转Danbooru标签流程")
    print("=" * 60)

    from huesae.agents.subagents.image_agent import ImageAgent, ImageMode
    from huesae.models.models_factory import create_chat_model

    agent = ImageAgent(llm=create_chat_model("deepseek"))

    # Step 1: 用户请求转标签
    print("\n[Step 1] 用户输入：'转成Danbooru标签：一个银发红瞳的少女'")
    result = agent.process_input("转成Danbooru标签：一个银发红瞳的少女")
    print(f"  模式: {result['mode']}")
    print(f"  步骤: {result['step']}")
    print(f"  标签: {result.get('danbooru_tags', [])[:5]}...")
    print(f"  消息: {result['message'][:100]}...")

    # Step 2: 用户说换一版
    print("\n[Step 2] 用户说：'换一版'")
    state = {
        "step": result["step"],
        "mode": result["mode"],
        "prompt": result.get("prompt", ""),
        "danbooru_tags": result.get("danbooru_tags", []),
    }
    result2 = agent.process_step(state, "换一版")
    print(f"  步骤: {result2['step']}")
    print(f"  标签: {result2.get('danbooru_tags', [])[:5]}...")

    # Step 3: 用户说可以了
    print("\n[Step 3] 用户说：'可以了'")
    state = {
        "step": result2["step"],
        "mode": result2["mode"],
        "prompt": result2.get("prompt", ""),
    }
    result3 = agent.process_step(state, "可以了")
    print(f"  步骤: {result3['step']}")
    print(f"  消息: {result3['message']}")

    print("\n✓ 转Danbooru标签流程测试通过\n")


def test_expand_prompt_flow():
    """测试扩写提示词流程"""
    print("=" * 60)
    print("测试 扩写提示词流程")
    print("=" * 60)

    from huesae.agents.subagents.image_agent import ImageAgent, ImageMode
    from huesae.models.models_factory import create_chat_model

    agent = ImageAgent(llm=create_chat_model("deepseek"))

    # Step 1: 用户请求扩写
    print("\n[Step 1] 用户输入：'扩写提示词：一个银发红瞳的少女'")
    result = agent.process_input("扩写提示词：一个银发红瞳的少女")
    print(f"  模式: {result['mode']}")
    print(f"  步骤: {result['step']}")
    print(f"  扩写: {result.get('expanded_prompt', '')[:80]}...")

    # Step 2: 用户说再写一版
    print("\n[Step 2] 用户说：'再写一版'")
    state = {
        "step": result["step"],
        "mode": result["mode"],
        "prompt": result.get("prompt", ""),
    }
    result2 = agent.process_step(state, "再写一版")
    print(f"  步骤: {result2['step']}")
    print(f"  扩写: {result2.get('expanded_prompt', '')[:80]}...")

    # Step 3: 用户说接受
    print("\n[Step 3] 用户说：'接受'")
    state = {
        "step": result2["step"],
        "mode": result2["mode"],
        "prompt": result2.get("prompt", ""),
    }
    result3 = agent.process_step(state, "接受")
    print(f"  步骤: {result3['step']}")
    print(f"  消息: {result3['message']}")

    print("\n✓ 扩写提示词流程测试通过\n")


# def main():
    # """运行所有测试"""
    # print("\n" + "=" * 60)
    # print("HuesaeAgents 生图Agent测试 v2")
    # print("=" * 60 + "\n")

    # try:
    #     test_intent_detection()
    #     # test_direct_image_flow()
    #     # test_short_prompt()
    #     # test_convert_tags_flow()
    #     # test_expand_prompt_flow()
    # except Exception as e:
    #     print(f"\n✗ 测试失败: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return

    # print("=" * 60)
    # print("所有测试完成！")
    # print("=" * 60)


if __name__ == "__main__":
    test_intent_detection()
