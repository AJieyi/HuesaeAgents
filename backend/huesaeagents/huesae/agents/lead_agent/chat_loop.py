"""HuesaeAgents 终端交互入口

DeerFlow Harness Engineering 模式：
- 主Agent通过 ReAct 循环让 LLM 自主选择和调用工具
- 子Agent作为可委托组件处理复杂多轮对话
- 工具选择完全由 LLM 决定

对话流：
    用户输入 → 主Agent ReAct 循环 → [直接回复 | 调用工具 | 委托子Agent]
    工具/子Agent完成 → 主Agent包装展示 → 继续对话

永久关闭token用量，改 chat_loop.py 这一行默认值：
os.getenv("HUESAE_SHOW_TOKEN_USAGE_LOGS", "1")

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
import logging
import os
import sys
import threading
import time
import warnings

# 抑制 langgraph 内部弃用警告
warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change",
    category=UserWarning,
)

from langchain.messages import HumanMessage, AIMessage


_CONSOLE_OUTPUT_LOCK = threading.RLock()


class _ConsoleLogHandler(logging.StreamHandler):
    """Write logs through the same lock used by streaming terminal output."""

    def emit(self, record: logging.LogRecord) -> None:
        with _CONSOLE_OUTPUT_LOCK:
            super().emit(record)
            self.flush()


def configure_chat_loop_logging() -> None:
    """配置终端日志，让 TokenUsageMiddleware 的 token 用量能在 chat_loop 中显示。"""
    show_token_usage_logs = os.getenv("HUESAE_SHOW_TOKEN_USAGE_LOGS", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    handler = _ConsoleLogHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(
        level=logging.INFO if show_token_usage_logs else logging.WARNING,
        handlers=[handler],
        force=True,
    )


def print_stream(text: str, prefix: str = "AI: ", delay: float = 0.025) -> None:
    """逐字打印，模拟流式输出

    Args:
        text: 要打印的文本
        prefix: 前缀（如"AI: "）
        delay: 每个字符的延迟（秒）
    """
    with _CONSOLE_OUTPUT_LOCK:
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
    configure_chat_loop_logging()

    from .lead_agent import create_main_agent
    from ...skills.registry import SkillRegistry
    from ...services import create_honcho_memory_service
    from ...subagents.general_agent import create_general_agent
    from ...subagents.image_agent import create_image_agent

    # 创建主Agent并注册子Agent
    skill_registry = SkillRegistry()
    memory_service = create_honcho_memory_service()
    main_agent = create_main_agent(skill_registry=skill_registry, memory_service=memory_service)
    main_agent.register_sub_agent(create_image_agent())
    main_agent.register_sub_agent(create_general_agent(skill_registry=skill_registry, runtime=main_agent._runtime))

    # 终端交互保留当前进程状态，同时把跨终端记忆写入 Honcho。
    messages = []
    active_subagent = None
    vision_context = None

    print("=" * 50)
    print("HuesaeAgents 终端交互")
    print("=" * 50)
    print("提示：")
    print("  - 输入消息与Agent对话")
    print("  - 说'我想生成图片'进入生图模式")
    print("  - 输入 exit/quit 退出\n")
    print(f"记忆状态：{memory_service.status}")
    if memory_service.enabled and memory_service.session_id:
        print(f"记忆会话：{memory_service.session_id}\n")

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

        state = {
            "messages": messages,
            "active_subagent": active_subagent,
            "vision_context": vision_context,
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
                confirm_msg = image_result.get("confirm_message", "")
                if image_result.get("image_urls"):
                    images_text = "\n".join(
                        [f"[图片] {url}" for url in image_result["image_urls"]]
                    )
                    full_msg = f"{wrap_msg}\n\n{images_text}\n\n{confirm_msg}"
                else:
                    full_msg = f"{wrap_msg}\n\n[图片] {image_result['image_url']}\n\n{confirm_msg}"

                # 替换 result 中的消息为完整回复
                result["messages"] = [AIMessage(content=full_msg)]
                if result.get("active_subagent"):
                    active_state = result["active_subagent"].setdefault("state", {})
                    active_state.update(image_result.get("subagent_state_update", {}))
                    history = result["active_subagent"].setdefault("history", [])
                    if history and isinstance(history[-1], AIMessage):
                        history[-1] = AIMessage(content=full_msg)

                # 流式打印包装语
                print_stream(wrap_msg)
                print()

                # 显示图片URL
                if image_result.get("image_urls"):
                    for url in image_result["image_urls"]:
                        print(f"[图片] {url}\n")
                else:
                    print(f"[图片] {image_result['image_url']}\n")
                if confirm_msg:
                    print_stream(confirm_msg)
                    print()

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
            active_subagent = result["active_subagent"]
        if result.get("clear_subagent"):
            active_subagent = None
        if "vision_context" in result:
            vision_context = result["vision_context"]

        assistant_response = "\n".join(
            msg.content if hasattr(msg, "content") else str(msg)
            for msg in result.get("messages", [])
            if str(getattr(msg, "content", msg)).strip()
        ).strip()

        if assistant_response:
            memory_service.store_exchange(user_input, assistant_response)

        # 更新主对话历史
        messages.append(HumanMessage(content=user_input))
        messages.extend(result.get("messages", []))


if __name__ == "__main__":
    run_chat_loop()
