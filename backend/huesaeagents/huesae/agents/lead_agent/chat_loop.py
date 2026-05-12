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

from langchain_core.messages import HumanMessage


def run_chat_loop():
    """终端交互循环

    模拟真实使用场景：
    - 用户持续输入消息
    - 主Agent回复显示在终端
    - 生图完成后显示图片URL（只显示一次）
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

        # 打印AI回复
        for msg in result.get("messages", []):
            content = msg.content if hasattr(msg, "content") else str(msg)
            if content.strip():
                print(f"AI: {content}\n")

        # 显示图片URL（如果有，只显示一次）
        if result.get("image_url"):
            print(f"[图片] {result['image_url']}\n")

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
