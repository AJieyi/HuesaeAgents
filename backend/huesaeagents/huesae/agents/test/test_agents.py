"""
测试 agents 模块 - Phase1 表情包与语C能力

使用方法:
1. 激活虚拟环境: conda activate HuesaeAgents
2. 安装依赖: pip install langchain langgraph langchain-core langchain-deepseek python-dotenv
3. 确保 .env 文件中有 DEEPSEEK_API_KEY
4. 运行: python huesae/agents/test/test_agents.py
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

# 添加项目路径（指向 backend 目录，使 huesaeagents.huesae 可被导入）
# 文件: backend/huesaeagents/huesae/agents/test/test_agents.py
# test -> agents -> huesae -> huesaeagents -> backend (向上5层到backend)
_test_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_test_file)))))
# project_root = f:/agent/code/agent/HuesaeAgents/ben/HuesaeAgents/backend/
os.chdir(project_root)  # 切换到项目根目录
sys.path.insert(0, project_root)

# 加载 .env 文件
load_dotenv()

from langchain_core.messages import HumanMessage
from huesaeagents.huesae.agents.state import ThreadState
from huesaeagents.huesae.agents.agent_factory import create_huesae_agent, create_image_agent


def test_create_agent():
    """测试创建 Agent"""
    print("\n1. 测试创建 Agent")
    try:
        agent = create_huesae_agent()
        print(f"   Agent type: {type(agent).__name__}")
        print("   ✅ create_huesae_agent 成功")
        return agent
    except Exception as e:
        print(f"   ❌ create_huesae_agent 失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        return None


def test_emotion_chat(agent, message: str, character_id: str = "gentle_sister"):
    """
    测试带情绪的对话

    Args:
        agent: Agent实例
        message: 用户消息
        character_id: 角色ID
    """
    try:
        print(f"   用户: {message}")
        result = agent.invoke({
            "messages": [HumanMessage(content=message)],
            "character_id": character_id,
            "emotion_state": None,
            "emotion_score": None,
            "user_id": None,
            "thread_id": None
        })
        # 获取最后一条 AI 消息
        ai_message = result["messages"][-1]
        print(f"   AI: {ai_message.content}")
        print(f"   情绪: {result.get('emotion_state', 'N/A')}")
        print()
    except Exception as e:
        print(f"   ❌ 对话失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        print()


def test_emotion_detection(agent):
    """测试情绪检测"""
    print("\n2. 测试情绪检测")

    test_cases = [
        ("今天考试考砸了，好难过...", "gentle_sister"),
        ("哈哈哈哈太开心了!", "gentle_sister"),
        ("一个人好寂寞...", "gentle_sister"),
        ("我不喜欢你!", "tsundere"),
        ("有点不好意思...", "furry_fox"),
    ]

    for message, character_id in test_cases:
        test_emotion_chat(agent, message, character_id)


def test_character_switch(agent):
    """测试角色切换"""
    print("\n3. 测试角色切换")

    message = "今天好开心!"

    characters = ["gentle_sister", "tsundere", "furry_fox"]
    for character_id in characters:
        print(f"   --- 角色: {character_id} ---")
        test_emotion_chat(agent, message, character_id)


def test_create_image_agent():
    """测试创建图片 Agent"""
    print("\n4. 测试创建图片 Agent")
    try:
        agent = create_image_agent()
        print(f"   Agent type: {type(agent).__name__}")
        print("   ✅ create_image_agent 成功")
        return agent
    except Exception as e:
        print(f"   ❌ create_image_agent 失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        return None


def test_image_generation(agent, message: str):
    """
    测试图片生成

    Args:
        agent: Agent实例
        message: 用户消息
    """
    try:
        print(f"   用户: {message}")
        result = agent.invoke({
            "messages": [HumanMessage(content=message)],
        })
        # 获取最后一条 AI 消息
        ai_message = result["messages"][-1]
        print(f"   AI 响应类型: {type(ai_message).__name__}")

        # 检查是否有 tool_calls
        if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            print(f"   调用了工具: {[tc['name'] for tc in ai_message.tool_calls]}")

        # 检查返回的内容
        if hasattr(ai_message, "content") and ai_message.content:
            content = ai_message.content
            if len(content) > 200:
                print(f"   AI: {content[:200]}...")
            else:
                print(f"   AI: {content}")
        print()
    except Exception as e:
        print(f"   ❌ 图片生成失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        print()


def test_image_agent():
    """测试图片 Agent"""
    print("=" * 60)
    print("测试图片 Agent - 图片生成能力集成")
    print("=" * 60)

    # 检查环境变量
    print("\n0. 检查环境变量")
    doubao_key = os.environ.get("DOUBAO_SEEDREAM_API_KEY")
    if doubao_key:
        print(f"   DOUBAO_SEEDREAM_API_KEY: 已设置 (长度={len(doubao_key)})")
    else:
        print("   DOUBAO_SEEDREAM_API_KEY: 未设置")

    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        print(f"   DEEPSEEK_API_KEY: 已设置 (长度={len(deepseek_key)})")
    else:
        print("   DEEPSEEK_API_KEY: 未设置")
        print("   ⚠️ 跳过测试")

    # 测试创建 Agent
    agent = test_create_image_agent()

    if not agent:
        print("\n❌ Agent创建失败，退出测试")
        return

    # 测试图片生成
    print("\n5. 测试图片生成")
    test_cases = [
        "画一个银发红瞳的少女在樱花树下",
        "生成一张科幻风格的图片，太空站",
    ]

    for message in test_cases:
        test_image_generation(agent, message)

    print("=" * 60)
    print("图片 Agent 测试完成")
    print("=" * 60)


def test_models():
    print("=" * 60)
    print("测试 agents 模块 - Phase1 表情包与语C能力")
    print("=" * 60)

    # 检查环境变量
    print("\n0. 检查环境变量")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        print(f"   DEEPSEEK_API_KEY: 已设置 (长度={len(api_key)})")
        print("   ✅ 环境变量检查通过")
    else:
        print("   DEEPSEEK_API_KEY: 未设置")
        print("   ⚠️ 跳过API调用测试")

    # 测试创建 Agent
    agent = test_create_agent()

    if not agent:
        print("\n❌ Agent创建失败，退出测试")
        return

    # 测试情绪检测
    test_emotion_detection(agent)

    # 测试角色切换
    test_character_switch(agent)

    print("=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "image":
        test_image_agent()
    else:
        test_models()
