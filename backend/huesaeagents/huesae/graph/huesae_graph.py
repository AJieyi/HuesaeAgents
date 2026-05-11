"""HuesaeAgents 主工作流

LangGraph 状态图，编排所有智能体节点。
支持多轮对话：通过 checkpoint 保存状态，每次 invoke 恢复继续执行。

工作流结构：
    input → classify_intent → [条件路由]
                              ├─ chat → chat_agent → END
                              ├─ image → image_agent → END
                              ├─ voice → voice_agent → END
                              ├─ memory → memory_agent → END
                              ├─ search → search_agent → END
                              ├─ remind → remind_agent → END
                              └─ safe → safe_agent → END

多轮对话使用方式：
    # 第一次调用
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "我想生成图片"}]},
        config={"configurable": {"thread_id": "user_123"}}
    )
    # 第二次调用（自动恢复状态）
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "夕阳下看大海的少女"}]},
        config={"configurable": {"thread_id": "user_123"}}
    )
"""
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage

from .state import HuesaeState
from .conditional_logic import classify_intent, route_by_intent, Intent


# ============== 节点定义 ==============

def input_node(state: HuesaeState) -> dict:
    """输入处理节点：初始化默认值"""
    return {
        "character_id": state.get("character_id", "gentle_sister"),
        "emotion_state": state.get("emotion_state", "平静"),
        "emotion_score": state.get("emotion_score", 0.0),
        "safety_flag": False,
        "high_risk_flag": False,
    }


def classify_intent_node(state: HuesaeState) -> dict:
    """意图分类节点（LLM驱动）"""
    intent = classify_intent(state)
    return {"intent": intent}


def chat_agent_node(state: HuesaeState) -> dict:
    """对话智能体节点（占位）"""
    return {
        "messages": [AIMessage(content="我是对话Agent，还在开发中~ (≧▽≦)/")],
    }


def image_agent_node(state: HuesaeState) -> dict:
    """生图智能体节点

    使用LLM驱动的对话管理器，支持多轮对话：
    - 追问：用户没提供提示词时主动询问
    - 推荐：用户要求推荐时主动生成建议
    - 扩写：用户描述太短时帮助扩写
    - 确认：推荐/扩写后询问用户是否满意
    - 生图：用户确认后调用Provider生成图片
    - 换图：用户不满意时重新生成

    多轮状态通过 checkpoint 保存和恢复。
    """
    import asyncio
    from ..agents.subagents.image_agent import create_image_agent
    from ..models.models_factory import create_chat_model

    # 初始化对话管理器
    llm = create_chat_model("deepseek")
    manager = create_image_agent(llm=llm)

    # 获取最后一条用户消息
    last_user_msg = ""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (
            isinstance(msg, dict) and msg.get("role") == "user"
        ):
            last_user_msg = (
                msg.content if hasattr(msg, "content") else msg.get("content", "")
            )
            break

    # 交给对话管理器处理
    result = manager.process(state, last_user_msg)

    # 如果决策是生图，执行异步生图
    if result.get("image_step") == "generate":
        prompt = result.get("image_prompt", "")
        provider = result.get("selected_provider", "doubao")

        try:
            generation = asyncio.run(manager.generate_image(prompt, provider))

            return {
                "image_step": "show_image",
                "generated_image_url": generation.url,
                "messages": [
                    AIMessage(
                        content=(
                            f"{result['messages'][0].content}\n\n"
                            f"图片生成完成！\n\n"
                            f"工具：{provider}\n"
                            f"提示词：{prompt}\n\n"
                            f"{generation.url}\n\n"
                            f"是否满意？不满意的话我可以重新生成哦~"
                        )
                    )
                ],
            }
        except Exception as e:
            return {
                "image_step": "finish",
                "messages": [AIMessage(content=f"图片生成失败：{str(e)}")],
            }

    return result


def voice_agent_node(state: HuesaeState) -> dict:
    """语音智能体节点（占位）"""
    return {
        "messages": [AIMessage(content="语音功能还在开发中~")],
    }


def memory_agent_node(state: HuesaeState) -> dict:
    """记忆智能体节点（占位）"""
    return {
        "messages": [AIMessage(content="记忆功能还在开发中~")],
    }


def search_agent_node(state: HuesaeState) -> dict:
    """搜索智能体节点（占位）"""
    return {
        "messages": [AIMessage(content="搜索功能还在开发中~")],
    }


def remind_agent_node(state: HuesaeState) -> dict:
    """提醒智能体节点（占位）"""
    return {
        "messages": [AIMessage(content="提醒功能还在开发中~")],
    }


def safe_agent_node(state: HuesaeState) -> dict:
    """安全智能体节点"""
    return {
        "safety_flag": True,
        "high_risk_flag": True,
        "messages": [
            AIMessage(
                content=(
                    "*轻轻握住你的手*\n\n"
                    "我在这里陪着你，你不是一个人...\n\n"
                    "如果你感到痛苦或绝望，请一定要寻求专业帮助：\n"
                    "- 心理危机干预热线：400-161-9995\n"
                    "- 北京心理危机研究与干预中心：010-82951332\n"
                    "- 生命热线：400-821-1215\n\n"
                    "你的生命很珍贵，请不要独自承受这些。"
                )
            )
        ],
    }


# ============== 工作流构建 ==============

def create_huesae_graph(checkpointer: Any = None) -> StateGraph:
    """创建 HuesaeAgents 主工作流

    Args:
        checkpointer: 持久化检查点，默认使用内存存储

    Returns:
        编译后的 StateGraph

    Example:
        >>> graph = create_huesae_graph()
        >>> config = {"configurable": {"thread_id": "user_123"}}
        >>>
        >>> # 第一轮：用户输入
        >>> result = graph.invoke(
        ...     {"messages": [{"role": "user", "content": "我想生成图片"}]},
        ...     config=config
        ... )
        >>> print(result["messages"][-1].content)  # Agent回复
        >>>
        >>> # 第二轮：用户提供提示词（自动恢复状态）
        >>> result = graph.invoke(
        ...     {"messages": [{"role": "user", "content": "夕阳下看大海的少女"}]},
        ...     config=config
        ... )
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    # 创建状态图
    workflow = StateGraph(HuesaeState)

    # 添加节点
    workflow.add_node("input", input_node)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("chat_agent", chat_agent_node)
    workflow.add_node("image_agent", image_agent_node)
    workflow.add_node("voice_agent", voice_agent_node)
    workflow.add_node("memory_agent", memory_agent_node)
    workflow.add_node("search_agent", search_agent_node)
    workflow.add_node("remind_agent", remind_agent_node)
    workflow.add_node("safe_agent", safe_agent_node)

    # 设置入口点
    workflow.set_entry_point("input")

    # 定义边
    workflow.add_edge("input", "classify_intent")

    # 条件路由
    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "chat_agent": "chat_agent",
            "image_agent": "image_agent",
            "voice_agent": "voice_agent",
            "memory_agent": "memory_agent",
            "search_agent": "search_agent",
            "remind_agent": "remind_agent",
            "safe_agent": "safe_agent",
        },
    )

    # 所有Agent都指向END
    for node in [
        "chat_agent",
        "image_agent",
        "voice_agent",
        "memory_agent",
        "search_agent",
        "remind_agent",
        "safe_agent",
    ]:
        workflow.add_edge(node, END)

    # 编译
    return workflow.compile(checkpointer=checkpointer)


# ============== 便捷函数：演示多轮对话 ==============

def run_chat_loop():
    """演示多轮对话循环

    这是一个示例，展示如何在主程序中实现循环调用。
    """
    from langchain_core.messages import HumanMessage

    graph = create_huesae_graph()
    config = {"configurable": {"thread_id": "demo_user"}}

    print("=" * 50)
    print("HuesaeAgents 多轮对话演示")
    print("=" * 50)
    print("输入 'exit' 退出\n")

    while True:
        user_input = input("用户: ").strip()
        if user_input.lower() == "exit":
            break

        # 调用图（自动恢复之前的状态）
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        # 打印AI回复
        ai_message = result["messages"][-1]
        print(f"AI: {ai_message.content}\n")

        # 如果生图流程结束，重置状态
        if result.get("image_step") == "finish":
            print("[对话已结束，开始新的话题]\n")


if __name__ == "__main__":
    run_chat_loop()
