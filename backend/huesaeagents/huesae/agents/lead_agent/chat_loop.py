"""HuesaeAgents 终端交互入口

DeerFlow Harness Engineering 模式：
- 主Agent通过 ReAct 循环让 LLM 自主选择和调用工具
- 子Agent作为可委托组件处理复杂多轮对话
- 工具选择完全由 LLM 决定

对话流：
    用户输入 → 主Agent ReAct 循环 → [直接回复 | 调用工具 | 委托子Agent]
    工具/子Agent完成 → 主Agent包装展示 → 继续对话
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
import warnings

# 抑制 langgraph 内部弃用警告
warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change",
    category=UserWarning,
)

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

    # 使用状态管理器（仅内存存储）
    state_manager = StateManager()
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
            "active_subagent": conv_state.active_subagent,
        }

        # 调用主Agent（ReAct 循环）
        result = main_agent.process(state, user_input)

        # 处理 pending_generation（子Agent或工具触发的异步生图）
        if result.get("pending_generation"):
            prompt = result.get("prompt", "")
            size = result.get("size", "2K")
            output_format = result.get("output_format", "jpeg")
            is_batch = result.get("is_batch", False)

            print_stream("图片正在生成中，请稍等~")
            print()

            try:
                # 异步执行生图
                image_result = asyncio.run(main_agent.execute_image_generation(
                    prompt=prompt,
                    size=size,
                    output_format=output_format,
                    is_batch=is_batch,
                ))

                # 构造完整回复（包装语 + 图片URL）
                wrap_msg = image_result["wrap_message"]
                if image_result.get("image_urls"):
                    images_text = "\n".join(
                        [f"[图片] {url}" for url in image_result["image_urls"]]
                    )
                    full_msg = f"{wrap_msg}\n\n{images_text}"
                else:
                    full_msg = f"{wrap_msg}\n\n[图片] {image_result['image_url']}"

                # 替换 result 中的消息为完整回复
                result["messages"] = [AIMessage(content=full_msg)]

                # 流式打印包装语
                print_stream(wrap_msg)
                print()

                # 显示图片URL
                if image_result.get("image_urls"):
                    for url in image_result["image_urls"]:
                        print(f"[图片] {url}\n")
                else:
                    print(f"[图片] {image_result['image_url']}\n")

            except Exception as e:
                error_msg = f"图片生成失败：{str(e)}"
                print_stream(error_msg)
                print()
                result["messages"] = [AIMessage(content=error_msg)]

        else:
            # 流式打印AI回复（打字机效果）
            for msg in result.get("messages", []):
                content = msg.content if hasattr(msg, "content") else str(msg)
                if content.strip():
                    print_stream(content)
                    print()

        # 更新子Agent状态
        if "active_subagent" in result:
            conv_state.active_subagent = result["active_subagent"]
        if result.get("clear_subagent"):
            conv_state.active_subagent = None

        # 更新主对话历史
        conv_state.messages.append(HumanMessage(content=user_input))
        conv_state.messages.extend(result.get("messages", []))

        # 持久化状态
        state_manager.save_state(session_id)

    # 退出时清除持久化状态
    state_manager.clear_state(session_id)


if __name__ == "__main__":
    run_chat_loop()
