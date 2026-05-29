"""通用子Agent。

用于处理主Agent委派的复杂通用任务：多步骤工具调用、信息整理、
资料加工、报告汇总等。该子Agent只执行，不追问用户，也不再委派其他子Agent。
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from ..agents.model_adapter import ensure_chat_model
from ..agents.thread_state import HuesaeThreadState
from ..skills.registry import SkillRegistry
from ..tools.runtime import GENERAL_AGENT_EXCLUDED_TOOL_NAMES, build_shared_runtime
from ..agents.middlewares import RuntimeToolMiddleware
from .base import BaseSubAgent


GENERAL_AGENT_SYSTEM_PROMPT = """你是一个任务执行Agent，负责完成主Agent委派的复杂任务。

## 你的职责
- 只专注于完成任务，不偏离主题
- 不要向用户追问，不要请求澄清
- 能用工具就直接使用工具，尽量把任务做完
- 输出要简洁、专业、清晰
- 不要使用角色语气，不要卖萌，不要展开闲聊
- 不要委派其他子Agent，不要调用 task_tool

## 工作原则
1. 只使用当前可见工具完成任务。
2. 如果已经能完成任务，就直接给出结果，不要继续循环。
3. 如果步骤已经很多，最后要给出简洁总结。
4. 如果遇到错误，直接说明错误原因并结束。

## 可用工具
{tools_description}

## 工具约束
{tool_constraints}

## MCP 工具原则
{mcp_tool_principles}

## 可用 Skills
{skills_section}
"""


class GeneralSubAgent(BaseSubAgent):
    """通用任务子Agent。"""

    name = "general"
    MAX_STEPS = 15

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        runtime=None,
        skill_registry: SkillRegistry | None = None,
    ):
        self.llm = llm
        self._agent_model = ensure_chat_model(llm)
        self.skill_registry = skill_registry
        self.runtime = runtime or build_shared_runtime(
            llm,
            None,
            skill_registry=skill_registry,
        )
        self.tools: list[BaseTool] = []
        self.tool_map: dict[str, BaseTool] = {}
        self._checkpointer = InMemorySaver()
        self.agent = None
        self._refresh_tools()

    def _refresh_tools(self) -> None:
        """刷新通用子Agent可见工具。"""
        self.runtime.refresh_builtin_tools()
        self.tools = self.runtime.get_tools(
            include_mcp=self.runtime.mcp_loaded,
            include_task_tool=False,
            exclude_names=GENERAL_AGENT_EXCLUDED_TOOL_NAMES,
        )
        self.tool_map = self.runtime.get_tool_map(
            include_mcp=self.runtime.mcp_loaded,
            include_task_tool=False,
            exclude_names=GENERAL_AGENT_EXCLUDED_TOOL_NAMES,
        )
        self._rebuild_agent()

    def _refresh_tools_with_mcp(self) -> None:
        """加载 MCP 工具后刷新通用子Agent工具视图。"""
        self.runtime.refresh_mcp_tools(force=False)
        self._refresh_tools()

    def _rebuild_agent(self) -> None:
        """Compile the LangChain graph-backed general agent."""
        self.agent = create_agent(
            model=self._agent_model,
            tools=self.tools,
            system_prompt=self._build_system_prompt(),
            middleware=[
                RuntimeToolMiddleware(
                    tools=lambda: self.tools,
                    tool_map=lambda: self.tool_map,
                    system_message=self._build_system_prompt,
                    ensure_mcp_tools=self._refresh_tools_with_mcp,
                    is_mcp_loaded=lambda: self.runtime.mcp_loaded,
                    has_mcp_tools=lambda: bool(self.runtime._mcp_tools),
                )
            ],
            state_schema=HuesaeThreadState,
            checkpointer=self._checkpointer,
            name="huesae_general_agent",
        )

    def process(self, state: dict, user_input: str) -> dict:
        """执行主Agent委派的通用任务。"""
        self._refresh_tools()
        try:
            graph_state = self.agent.invoke(
                {"messages": [HumanMessage(content=user_input)], "user_input": user_input},
                config={"configurable": {"thread_id": str(state.get("thread_id") or f"general-{id(state)}")}},
            )
        except Exception as exc:
            return self._make_result("error", f"通用任务执行失败：{exc}")

        messages = graph_state.get("messages") or []
        new_messages = self._new_messages_for_turn(messages, user_input)
        latest_tool_result = self._latest_tool_content(new_messages)
        content = self._last_ai_content(new_messages) or self._last_ai_content(messages)
        if not content and latest_tool_result:
            content = latest_tool_result
        if not content:
            content = "任务已完成。"
        return self._make_result("finish", content)

    def _build_system_prompt(self, user_input: str | None = None) -> SystemMessage:
        """构建通用任务执行提示词。"""
        tools_description = self.runtime.format_tools_for_prompt(
            include_mcp=self.runtime.mcp_loaded,
            include_task_tool=False,
            exclude_names=GENERAL_AGENT_EXCLUDED_TOOL_NAMES,
        )
        tool_constraints = "\n".join(
            [
                "- 工具名称、描述和参数 schema 是选择工具的主要依据；只调用当前可见工具，不要编造工具名或参数名。",
                "- 每轮只选择一个行动：直接回复，或调用一个最合适的工具。",
                "- 不要调用 task_tool，不要调用任何生图工具。",
            ]
        )
        mcp_tool_principles = self.runtime.format_mcp_tool_principles()
        skills_section = (
            self.skill_registry.format_for_prompt()
            if self.skill_registry is not None
            else "暂无可用 Skills。"
        )

        return SystemMessage(
            content=GENERAL_AGENT_SYSTEM_PROMPT.format(
                tools_description=tools_description,
                tool_constraints=tool_constraints,
                mcp_tool_principles=mcp_tool_principles,
                skills_section=skills_section,
            )
        )

    def _build_messages(self, state: dict, user_input: str) -> list:
        """构建通用任务执行所需消息。"""
        messages = [self._build_system_prompt(user_input)]
        messages.extend(state.get("messages", [])[-10:])
        messages.append(HumanMessage(content=user_input))
        return messages

    @staticmethod
    def _new_messages_for_turn(messages: list, user_input: str) -> list:
        start_index = 0
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if getattr(message, "type", None) == "human" and str(message.content) == user_input:
                start_index = index + 1
                break
        return messages[start_index:]

    @staticmethod
    def _latest_tool_content(messages: list) -> str | None:
        for message in reversed(messages):
            if getattr(message, "type", None) == "tool":
                return str(message.content)
        return None

    @staticmethod
    def _last_ai_content(messages: list) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return str(message.content or "").strip()
        return ""

    @staticmethod
    def _make_result(action: str, response: str) -> dict:
        """构造通用子Agent标准结果。"""
        return {
            "action": action,
            "response": response,
            "prompt": None,
            "provider": None,
            "data": {},
        }


def create_general_agent(
    llm: BaseChatModel | None = None,
    *,
    runtime=None,
    skill_registry: SkillRegistry | None = None,
) -> GeneralSubAgent:
    """创建通用子Agent工厂函数。"""
    if llm is None:
        try:
            from huesae.models.models_factory import create_chat_model
        except ImportError:
            from huesaeagents.huesae.models.models_factory import create_chat_model
        llm = create_chat_model("deepseek")

    return GeneralSubAgent(
        llm=llm,
        runtime=runtime,
        skill_registry=skill_registry,
    )


__all__ = ["GeneralSubAgent", "create_general_agent"]
