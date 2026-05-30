"""Lead agent runtime built on LangChain create_agent and LangGraph state."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from ..middlewares import (
    MiddlewarePipeline,
    RuntimeToolMiddleware,
    adapt_middlewares,
    build_middlewares,
)
from ..model_adapter import ensure_chat_model
from ..thread_state import HuesaeThreadState
from ...services import HonchoMemoryService
from ...skills.registry import SkillRegistry
from ...subagents.base import BaseSubAgent
from ...subagents.executor import SubagentExecutor
from ...subagents.registry import SubAgentRegistry
from ...tools.runtime import MAIN_AGENT_EXCLUDED_TOOL_NAMES, build_shared_runtime
from ...tools.tools import parse_subagent_task


_SAFE_KEYWORDS = [
    "自杀",
    "自残",
    "想死",
    "不想活",
    "结束生命",
    "活着没意思",
    "kill myself",
    "suicide",
    "self-harm",
]

_FALLBACK_RESPONSE = "抱歉，我刚刚有点卡住了，请稍后再试一次，或者把需求再发我一遍。"

IMAGE_CONTEXT_PATH_KEYS = (
    "image_path",
    "image_paths",
    "last_image_path",
    "last_image_paths",
)


class HuesaeMainAgent:
    """Global Huesae agent.

    The public runtime entry is :meth:`invoke`. LangChain owns model/tool
    execution through ``create_agent`` and LangGraph owns per-thread state via
    the compiled graph checkpointer.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        character_id: str = "gentle_sister",
        mcp_tools_loader=None,
        skill_registry: SkillRegistry | None = None,
        memory_service: HonchoMemoryService | None = None,
        middleware_pipeline: MiddlewarePipeline | None = None,
    ):
        if llm is None:
            try:
                from huesae.models.models_factory import create_chat_model
            except ImportError:
                from huesaeagents.huesae.models.models_factory import create_chat_model
            llm = create_chat_model("deepseek")

        self.llm = llm
        self._agent_model = ensure_chat_model(llm)
        self.character_id = character_id
        self.skill_registry = skill_registry
        self.memory_service = memory_service
        self._middleware_pipeline = middleware_pipeline or build_middlewares()
        self.subagent_registry = SubAgentRegistry()
        self.subagent_executor = SubagentExecutor(self.subagent_registry)
        self._checkpointer = InMemorySaver()
        self.agent = None
        runtime_kwargs = {}
        if mcp_tools_loader is not None:
            runtime_kwargs["mcp_tools_loader"] = mcp_tools_loader
        if skill_registry is not None:
            runtime_kwargs["skill_registry"] = skill_registry
        self._runtime = build_shared_runtime(self.llm, self.subagent_registry, **runtime_kwargs)
        self.tools = []
        self.tool_map = {}
        self._vision_context: dict[str, Any] = {}
        self._refresh_tools()

    def _refresh_tools(self) -> None:
        """Refresh the main-agent tool schema and rebuild the graph."""
        self._runtime.subagent_registry = self.subagent_registry
        self._runtime.skill_registry = self.skill_registry
        self._runtime.refresh_builtin_tools()
        self.tools = self._runtime.get_tools(
            include_mcp=False,
            include_task_tool=True,
            exclude_names=MAIN_AGENT_EXCLUDED_TOOL_NAMES,
        )
        self.tool_map = self._runtime.get_tool_map(
            include_mcp=False,
            include_task_tool=True,
            exclude_names=MAIN_AGENT_EXCLUDED_TOOL_NAMES,
        )
        self._rebuild_agent()

    def _refresh_tools_with_mcp(self) -> None:
        """Load MCP tools lazily, then expose the refreshed schema."""
        self._runtime.subagent_registry = self.subagent_registry
        self._runtime.skill_registry = self.skill_registry
        self._runtime.refresh_builtin_tools()
        self.tools = self._runtime.get_tools(
            include_mcp=True,
            include_task_tool=True,
            exclude_names=MAIN_AGENT_EXCLUDED_TOOL_NAMES,
        )
        self.tool_map = self._runtime.get_tool_map(
            include_mcp=True,
            include_task_tool=True,
            exclude_names=MAIN_AGENT_EXCLUDED_TOOL_NAMES,
        )
        self._rebuild_agent()

    def _rebuild_agent(self) -> None:
        """Compile the LangChain agent backed by LangGraph."""
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
                    is_mcp_loaded=lambda: self._runtime.mcp_loaded,
                    has_mcp_tools=lambda: bool(self._runtime._mcp_tools),
                    subagent_executor=self.subagent_executor,
                ),
                *adapt_middlewares(self._middleware_pipeline.middlewares),
            ],
            state_schema=HuesaeThreadState,
            checkpointer=self._checkpointer,
            name="huesae_main_agent",
        )

    def register_sub_agent(self, agent: BaseSubAgent) -> None:
        """Register a subagent as a task_tool target."""
        description = None
        if agent.name == "image":
            description = "生图对话Agent，处理追问、推荐、扩写、确认、单图和组图生成。"
        elif agent.name == "general":
            description = "通用任务Agent，处理复杂通用任务、工具链执行、资料加工和结果汇总。"
        agent.runtime = self._runtime
        agent.skill_registry = self.skill_registry
        self.subagent_registry.register(agent, description=description)
        self._refresh_tools()

    def invoke(self, user_input: str, *, thread_id: str = "default") -> dict:
        """Run one user turn.

        Args:
            user_input: The latest user message.
            thread_id: LangGraph checkpoint thread id.

        Returns:
            ``{"messages": [AIMessage(...)]}`` plus optional ``artifacts`` and
            ``vision_context``. Long-running subagent state stays inside the
            LangGraph checkpoint under ``current_subagent``.
        """
        thread_id = str(thread_id or "default")
        config = self._graph_config(thread_id)

        if self._check_safety(user_input):
            message = AIMessage(content=self._safety_response())
            self._append_graph_messages(config, [HumanMessage(content=user_input), message])
            return {"messages": [message], "safety_flag": True}

        current_subagent = self._checkpoint_value(config, "current_subagent")
        if current_subagent:
            return self._continue_subagent(user_input, thread_id, config, current_subagent)

        vision_context = self._checkpoint_value(config, "vision_context") or {}
        self._vision_context = dict(vision_context)
        graph_input = {
            "messages": [HumanMessage(content=user_input)],
            "user_input": user_input,
            "vision_context": vision_context,
        }

        try:
            graph_state = self.agent.invoke(graph_input, config=config)
        except Exception:
            message = AIMessage(content=_FALLBACK_RESPONSE)
            self._append_graph_messages(config, [message])
            return {"messages": [message], "vision_context": vision_context}

        return self._format_graph_result(graph_state, user_input, thread_id, config, vision_context)

    def stream(self, user_input: str, *, thread_id: str = "default"):
        """Yield the final messages for callers that want a stream-like API."""
        result = self.invoke(user_input, thread_id=thread_id)
        for message in result.get("messages", []):
            yield message

    def get_state(self, thread_id: str = "default") -> dict:
        """Return the persisted LangGraph state values for a thread."""
        return dict(self._snapshot_values(self._graph_config(thread_id)))

    def _format_graph_result(
        self,
        graph_state: dict,
        user_input: str,
        thread_id: str,
        config: dict,
        vision_context: dict,
    ) -> dict:
        messages = graph_state.get("messages") or []
        new_messages = self._new_messages_for_turn(messages, user_input)
        latest_tool_result = self._latest_tool_content(new_messages)

        task = self._subagent_task_from_messages(new_messages)
        if task is not None:
            subagent_type, description = task
            execution = self.subagent_executor.start(
                subagent_type,
                description,
                parent_thread_id=thread_id,
            )
            return self._commit_subagent_execution(execution, config)

        updated_vision_context = self._vision_context_from_messages(vision_context, new_messages)
        self._persist_runtime_state(config, vision_context=updated_vision_context)

        delegated_context = graph_state.get("current_subagent")
        delegated_artifacts = list(graph_state.get("artifacts") or [])
        if delegated_context is not None or delegated_artifacts:
            ai_message = self._last_ai_message(new_messages) or self._last_ai_message(messages)
            content = str(getattr(ai_message, "content", "") or "")
            if not content and latest_tool_result:
                content = latest_tool_result
            if not content:
                content = _FALLBACK_RESPONSE
            result = {
                "messages": [AIMessage(content=content)],
                "vision_context": updated_vision_context,
                "artifacts": delegated_artifacts,
            }
            if delegated_context:
                result["current_subagent"] = delegated_context
            return result

        ai_message = self._last_ai_message(new_messages) or self._last_ai_message(messages)
        content = str(getattr(ai_message, "content", "") or "") if ai_message is not None else ""
        if not content and latest_tool_result:
            content = self._format_last_tool_result(latest_tool_result)
        if not content:
            content = _FALLBACK_RESPONSE
        return {
            "messages": [AIMessage(content=content)],
            "vision_context": updated_vision_context,
        }

    def _continue_subagent(
        self,
        user_input: str,
        parent_thread_id: str,
        config: dict,
        context: dict,
    ) -> dict:
        execution = self.subagent_executor.resume(
            user_input,
            parent_thread_id=parent_thread_id,
            context=context,
        )
        return self._commit_subagent_execution(
            execution,
            config,
            leading_messages=[HumanMessage(content=user_input)],
        )

    def _commit_subagent_execution(
        self,
        execution,
        config: dict,
        *,
        leading_messages: list | None = None,
    ) -> dict:
        self._persist_runtime_state(
            config,
            current_subagent=execution.context,
            artifacts=execution.artifacts,
        )
        self._append_graph_messages(config, [*(leading_messages or []), execution.message])
        return execution.to_result()

    def _persist_runtime_state(
        self,
        config: dict,
        *,
        current_subagent: dict | None | object = ...,
        vision_context: dict | None | object = ...,
        artifacts: list[dict] | object = ...,
    ) -> None:
        update: dict[str, Any] = {}
        if current_subagent is not ...:
            update["current_subagent"] = current_subagent
        if vision_context is not ...:
            update["vision_context"] = vision_context or {}
        if artifacts is not ...:
            update["artifacts"] = list(artifacts or [])
        if update:
            self.agent.update_state(config, update)

    def _append_graph_messages(self, config: dict, messages: list) -> None:
        if messages:
            self.agent.update_state(config, {"messages": messages})

    def _checkpoint_value(self, config: dict, key: str):
        return self._snapshot_values(config).get(key)

    def _snapshot_values(self, config: dict) -> dict:
        try:
            snapshot = self.agent.get_state(config)
        except Exception:
            return {}
        return dict(getattr(snapshot, "values", {}) or {})

    @staticmethod
    def _graph_config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": str(thread_id or "default")}}

    def _build_system_prompt(self, user_input: str | None = None) -> SystemMessage:
        """Build the dynamic main-agent system message."""
        from .prompts import build_main_system_message

        tools_description = self._runtime.format_tools_for_prompt(
            include_mcp=self._runtime.mcp_loaded,
            include_task_tool=True,
            exclude_names=MAIN_AGENT_EXCLUDED_TOOL_NAMES,
        )
        tool_constraints = self._runtime.format_tool_constraints(
            include_mcp=self._runtime.mcp_loaded,
            include_task_tool=True,
            exclude_names=MAIN_AGENT_EXCLUDED_TOOL_NAMES,
        )
        mcp_tool_principles = self._runtime.format_mcp_tool_principles()
        subagents_description = self.subagent_registry.format_for_prompt()
        skills_section = (
            self.skill_registry.format_for_prompt()
            if self.skill_registry is not None
            else "暂无可用 Skills。"
        )
        memory_context_section = (
            self._get_memory_context(user_input)
            if self.memory_service is not None and self.memory_service.enabled
            else "暂无可用用户记忆。"
        )
        vision_context_section = self._format_vision_context_for_prompt(self._get_vision_context())

        return build_main_system_message(
            character_id=self.character_id,
            tools_description=tools_description,
            tool_constraints=tool_constraints,
            mcp_tool_principles=mcp_tool_principles,
            subagents_description=subagents_description,
            skills_section=skills_section,
            memory_context_section=memory_context_section,
            vision_context_section=vision_context_section,
        )

    def _get_memory_context(self, user_input: str | None) -> str:
        if self.memory_service is None:
            return "暂无可用用户记忆。"
        try:
            return self.memory_service.get_context(user_input=user_input)
        except TypeError:
            return self.memory_service.get_context()

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
    def _last_ai_message(messages: list) -> AIMessage | None:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None

    @staticmethod
    def _latest_tool_content(messages: list) -> str | None:
        for message in reversed(messages):
            if getattr(message, "type", None) == "tool":
                return str(message.content)
        return None

    @staticmethod
    def _subagent_task_from_messages(messages: list) -> tuple[str, str] | None:
        for message in reversed(messages):
            if getattr(message, "type", None) == "tool" or isinstance(message, AIMessage):
                task = parse_subagent_task(str(getattr(message, "content", "") or ""))
                if task is not None:
                    return task
        return None

    def _vision_context_from_messages(self, vision_context: dict, messages: list) -> dict:
        updated = dict(vision_context or {})
        ai_tool_calls: dict[str, dict] = {}
        for message in messages:
            if isinstance(message, AIMessage):
                for tool_call in getattr(message, "tool_calls", None) or []:
                    call_id = tool_call.get("id") or tool_call.get("name") or ""
                    ai_tool_calls[call_id] = tool_call
                continue
            if getattr(message, "type", None) != "tool":
                continue
            tool_call_id = getattr(message, "tool_call_id", "")
            tool_call = ai_tool_calls.get(tool_call_id) or {}
            tool_name = tool_call.get("name") or getattr(message, "name", "") or ""
            tool_args = tool_call.get("args") or {}
            updated = self._update_vision_context(updated, tool_name, tool_args, str(message.content))
        return updated

    @staticmethod
    def _format_last_tool_result(result) -> str:
        text = str(result).strip()
        if not text:
            return "工具已经执行完成，但没有返回可展示的内容。"
        return text

    def _get_vision_context(self) -> dict:
        return getattr(self, "_vision_context", {}) or {}

    def _update_vision_context(self, vision_context: dict, tool_name: str, tool_args: dict, result_text: str) -> dict:
        updated = dict(vision_context or {})
        if tool_name == "reverse_image_prompt":
            image_path = str(tool_args.get("image_path") or "").strip()
            if image_path:
                updated["image_path"] = image_path
            updated["last_reverse_prompt"] = result_text
            updated["last_vision_tool"] = tool_name
        elif self._looks_like_image_input(tool_name, tool_args):
            paths = self._collect_image_paths(tool_args)
            if paths:
                updated["image_path"] = paths[-1]
                updated["image_paths"] = paths
            updated["last_vision_tool"] = tool_name
        self._vision_context = updated
        return updated

    @staticmethod
    def _looks_like_image_input(tool_name: str, tool_args: dict) -> bool:
        if tool_name == "reverse_image_prompt":
            return True
        return any(key in tool_args for key in IMAGE_CONTEXT_PATH_KEYS)

    @staticmethod
    def _collect_image_paths(tool_args: dict) -> list[str]:
        collected: list[str] = []
        for key in ("image_path", "last_image_path"):
            value = tool_args.get(key)
            if isinstance(value, str) and value.strip():
                collected.append(value.strip())
        for key in ("image_paths", "last_image_paths"):
            values = tool_args.get(key)
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str) and value.strip():
                        collected.append(value.strip())
        return collected

    @staticmethod
    def _format_vision_context_for_prompt(vision_context: dict) -> str:
        if not vision_context:
            return "暂无图像上下文。"
        lines = ["当前图像上下文："]
        image_path = vision_context.get("image_path")
        if image_path:
            lines.append(f"- 最近图片路径：{image_path}")
        image_paths = vision_context.get("image_paths")
        if image_paths:
            lines.append(f"- 最近图片列表：{image_paths}")
        last_prompt = vision_context.get("last_reverse_prompt")
        if last_prompt:
            lines.append(f"- 最近反推提示词：{last_prompt}")
        return "\n".join(lines)

    @staticmethod
    def _check_safety(user_input: str) -> bool:
        content = user_input.lower()
        return any(keyword in content for keyword in _SAFE_KEYWORDS)

    @staticmethod
    def _safety_response() -> str:
        return (
            "*轻轻握住你的手*\n\n"
            "我在这里陪着你，你不是一个人...\n\n"
            "如果你感到痛苦或绝望，请一定要寻求专业帮助：\n"
            "- 心理危机干预热线：400-161-9995\n"
            "- 北京心理危机研究与干预中心：010-82951332\n"
            "- 生命热线：400-821-1215\n\n"
            "你的生命很珍贵，请不要独自承受这些。"
        )

__all__ = ["HuesaeMainAgent"]
