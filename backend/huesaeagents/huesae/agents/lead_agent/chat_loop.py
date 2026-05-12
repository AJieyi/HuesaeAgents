"""HuesaeAgents 终端交互入口

LangChain + 主智能体委派架构。
主Agent始终是对话核心，子Agent作为可调用组件。

对话流：
    用户输入 → 主Agent意图分类 → [子Agent|直接聊天]
    子Agent完成 → 主Agent包装展示 → 继续对话
"""
# 修复直接运行时的包路径（python chat_loop.py）
if __package__ is None:
    import sys
    from pathlib import Path

    _backend_dir = Path(__file__).resolve().parents[4]
    if str(_backend_dir) not in sys.path:
        sys.path.insert(0, str(_backend_dir))
    __package__ = "huesaeagents.huesae.agents.lead_agent"

import asyncio
import time

from langchain_core.messages import HumanMessage, AIMessage


def print_stream(text: str, prefix: str = "AI: ", delay: float = 0.025) -> None:
    """逐字打印，模拟流式输出

    Args:
        text: 要打印的文本
        prefix: 前缀（如"AI: "）
        delay: 每个字符的延迟（秒）
    """
    print(prefix, end="", flush=True)
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()


def run_chat_loop():
    """终端交互循环

    模拟真实使用场景：
    - 用户持续输入消息
    - 主Agent流式回复显示在终端（打字机效果）
    - 生图时先显示"生成中"提示，完成后展示图片
    - 输入 exit/quit 退出
    """
    from .lead_agent import create_main_agent
    from ..subagents.image_agent import create_image_agent
    from ..state_manager import StateManager

    # 创建主Agent并注册子Agent
    main_agent = create_main_agent()
    main_agent.register_sub_agent(create_image_agent())

    # 使用状态管理器（支持文件持久化）
    state_manager = StateManager(persist_path="./conversations")
    session_id = "terminal_user"
    conv_state = state_manager.get_state(session_id)

    print("=" * 50)
    print("HuesaeAgents 终端交互")
    print("=" * 50)
    print("提示：")
    print("  - 输入消息与Agent对话")
    print("  - 说'我想生成图片'进入生图模式")
    print("  - 输入 exit/quit 退出\n")

    while True:
        try:
            user_input = input("用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见~")
            break

        if user_input.lower() in ("exit", "quit"):
            print("再见~")
            break

        if not user_input:
            continue

        # 构建 state dict（从 StateManager 读取当前状态）
        state = {
            "messages": conv_state.messages,
            "image_goal": conv_state.image_goal,
        }

        # 调用主Agent
        result = main_agent.process(state, user_input)

        # 流式打印AI回复（打字机效果）
        for msg in result.get("messages", []):
            content = msg.content if hasattr(msg, "content") else str(msg)
            if content.strip():
                print_stream(content)
                print()

        # 处理待执行的生图
        if result.get("pending_generation"):
            prompt = result.get("prompt", "")
            print_stream("图片正在生成中，请稍等~")
            print()

            # 先保存用户输入和"生成中"提示到对话历史
            conv_state.messages.append(HumanMessage(content=user_input))
            generating_msg = result["messages"][0].content if result.get("messages") else "图片正在生成中，请稍等~"
            conv_state.messages.append(AIMessage(content=generating_msg))

            try:
                # 异步执行生图
                image_result = asyncio.run(main_agent.execute_image_generation(prompt))

                # 流式打印包装语
                print_stream(image_result["wrap_message"])
                print()

                # 显示图片URL
                print(f"[图片] {image_result['image_url']}\n")

                # 生图完成后更新对话历史（包装语作为AI回复）
                conv_state.messages.append(AIMessage(content=image_result["wrap_message"]))

                # 清除 image_goal（生图流程结束）
                conv_state.image_goal = None

            except Exception as e:
                error_msg = f"图片生成失败：{str(e)}"
                print_stream(error_msg)
                print()
                conv_state.messages.append(AIMessage(content=error_msg))

            # 保存状态并跳过常规更新
            state_manager.save_state(session_id)
            continue

        # 更新对话历史到状态管理器
        conv_state.messages.append(HumanMessage(content=user_input))
        conv_state.messages.extend(result.get("messages", []))

        # 管理 image_goal 生命周期
        if "image_goal" in result:
            conv_state.image_goal = result["image_goal"]
        if result.get("clear_image_goal"):
            conv_state.image_goal = None

        # 持久化状态
        state_manager.save_state(session_id)


if __name__ == "__main__":
    run_chat_loop()
