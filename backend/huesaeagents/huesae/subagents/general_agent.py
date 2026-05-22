"""通用子Agent。

用于处理主Agent委派的复杂通用任务：多步骤工具调用、信息整理、
资料加工、报告汇总等。该子Agent只执行，不追问用户，也不再委派其他子Agent。
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..skills.registry import SkillRegistry
from ..tools.runtime import GENERAL_AGENT_EXCLUDED_TOOL_NAMES, build_shared_runtime
from ..tools.tools import LOAD_MCP_TOOLS_SIGNAL, is_load_mcp_tools_signal
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
        self.skill_registry = skill_registry
        self.runtime = runtime or build_shared_runtime(
            llm,
            None,
            skill_registry=skill_registry,
        )
        self.tools: list[BaseTool] = []
        self.tool_map: dict[str, BaseTool] = {}
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

    def _refresh_tools_with_mcp(self) -> None:
        """加载 MCP 工具后刷新通用子Agent工具视图。"""
        self.runtime.refresh_mcp_tools(force=False)
        self._refresh_tools()

    def process(self, state: dict, user_input: str) -> dict:
        """执行主Agent委派的通用任务。"""
        self._refresh_tools()
        working_messages = self._build_messages(state, user_input)
        tool_results: list[str] = []

        for step in range(self.MAX_STEPS):
            try:
                ai_message = self._invoke_with_tools(working_messages)
            except Exception as exc:
                return self._make_result(
                    "error",
                    f"通用任务执行失败：{exc}",
                )

            tool_calls = getattr(ai_message, "tool_calls", None) or []
            if not tool_calls:
                content = str(ai_message.content or "").strip()
                if not content and tool_results:
                    content = tool_results[-1]
                if not content:
                    content = "任务已完成。"
                return self._make_result("finish", content)

            working_messages.append(ai_message)
            for tool_call in tool_calls:
                tool_name = tool_call.get("name") or ""
                tool_args = tool_call.get("args") or {}
                tool_call_id = tool_call.get("id") or tool_name
                result = self._execute_tool(tool_name, tool_args)

                if is_load_mcp_tools_signal(result):
                    if not self.runtime.mcp_loaded:
                        self._refresh_tools_with_mcp()
                    working_messages[0] = self._build_system_prompt(user_input)
                    result = "MCP 扩展工具已加载，请继续根据任务选择最合适的工具。"

                result_text = str(result)
                tool_results.append(result_text)
                working_messages.append(ToolMessage(content=result_text, tool_call_id=tool_call_id))

            working_messages[0] = self._build_system_prompt(user_input)

        summary = self._summarize_completion(working_messages, tool_results, user_input)
        return self._make_result("finish", summary)

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

    def _invoke_with_tools(self, messages: list) -> AIMessage:
        """使用当前可见工具执行模型调用。"""
        bound_llm = self.llm.bind_tools(self.tools)
        response = bound_llm.invoke(messages)
        if isinstance(response, AIMessage):
            return response
        return AIMessage(content=str(getattr(response, "content", response)))

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """执行通用任务中的指定工具。"""
        if tool_name not in self.tool_map:
            if not self.runtime.mcp_loaded:
                self._refresh_tools_with_mcp()
                return LOAD_MCP_TOOLS_SIGNAL
            return f"错误：未知工具 {tool_name}。可用工具：{list(self.tool_map.keys())}"

        tool = self.tool_map[tool_name]
        try:
            return str(tool.invoke(tool_args))
        except Exception as exc:
            return f"工具执行失败：{exc}"

    def _summarize_completion(self, messages: list, tool_results: list[str], user_input: str) -> str:
        """在步数耗尽时，让 LLM 汇总已完成的工作。"""
        summary_prompt = (
            "你已经执行了一个复杂任务，请总结你已经完成的工作，"
            "输出简洁清晰的最终结果，不要再提问，不要扩展到无关内容。"
        )
        summary_messages = [
            self._build_system_prompt(user_input),
            *messages[-8:],
            HumanMessage(content=summary_prompt),
        ]
        try:
            response = self.llm.invoke(summary_messages)
            content = str(getattr(response, "content", response) or "").strip()
            if content:
                return content
        except Exception:
            pass

        if tool_results:
            return tool_results[-1]
        return "任务已完成。"

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
