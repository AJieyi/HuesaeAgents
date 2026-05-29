"""主Agent（Lead Agent）- DeerFlow Harness Engineering 模式。"""
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from ..middlewares import MiddlewarePipeline, RuntimeToolMiddleware, adapt_middlewares, build_middlewares
from ..model_adapter import ensure_chat_model
from ..thread_state import HuesaeThreadState
from ...subagents.base import BaseSubAgent
from ...subagents.registry import SubAgentRegistry
from ...services import HonchoMemoryService
from ...skills.registry import SkillRegistry
from ...tools.runtime import MAIN_AGENT_EXCLUDED_TOOL_NAMES, build_shared_runtime
from ...tools.tools import (
    parse_subagent_task,
)


# ============== 安全检查 ==============

_SAFE_KEYWORDS = [
    "自杀", "自残", "想死", "不想活", "结束生命", "活着没意思",
    "kill myself", "suicide", "self-harm",
]

_FALLBACK_RESPONSE = "抱歉，我刚刚有点卡住了，请稍后再试一次，或者把需求再发我一遍。"

IMAGE_CONTEXT_PATH_KEYS = (
    "image_path",
    "image_paths",
    "last_image_path",
    "last_image_paths",
)


# ============== 主Agent ==============

class HuesaeMainAgent:
    """主Agent：LLM 自主工具选择的 ReAct 循环

    每轮接收用户输入，让 LLM 自主决策：
    - 直接回复
    - 调用工具
    - 委托子Agent

    典型调用：外层传入 messages 与用户最新输入，主Agent返回新的
    AIMessage；如果需要异步生图，会额外返回 pending_generation。
    """

    MAX_STEPS = 3  # ReAct 循环最大步数

    def __init__(
        self,
        llm: BaseChatModel,
        character_id: str = "gentle_sister",
        mcp_tools_loader=None,
        skill_registry: SkillRegistry | None = None,
        memory_service: HonchoMemoryService | None = None,
        middleware_pipeline: MiddlewarePipeline | None = None,
    ):
        self.llm = llm
        self._agent_model = ensure_chat_model(llm)
        self.character_id = character_id
        self.skill_registry = skill_registry
        self.memory_service = memory_service
        self._middleware_pipeline = middleware_pipeline or build_middlewares()
        self.subagent_registry = SubAgentRegistry()
        self._checkpointer = InMemorySaver()
        self._transient_thread_index = 0
        self.agent = None
        runtime_kwargs = {}
        if mcp_tools_loader is not None:
            runtime_kwargs["mcp_tools_loader"] = mcp_tools_loader
        if skill_registry is not None:
            runtime_kwargs["skill_registry"] = skill_registry
        self._runtime = build_shared_runtime(self.llm, self.subagent_registry, **runtime_kwargs)
        self.tools = []
        self.tool_map = {}
        self._vision_context: dict = {}
        self._refresh_tools()

    def _refresh_tools(self) -> None:
        """刷新工具列表。

        子Agent注册变化会影响 task_tool 的可用描述，因此注册后刷新一次。
        """
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
        """懒加载 MCP 工具后刷新完整工具视图。"""
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
                ),
                *adapt_middlewares(self._middleware_pipeline.middlewares),
            ],
            state_schema=HuesaeThreadState,
            checkpointer=self._checkpointer,
            name="huesae_main_agent",
        )

    def register_sub_agent(self, agent: BaseSubAgent) -> None:
        """注册子Agent"""
        description = None
        if agent.name == "image":
            description = "生图对话Agent，处理追问、推荐、扩写、确认、单图和组图生成。"
        elif agent.name == "general":
            description = "通用任务Agent，处理复杂通用任务、工具链执行、资料加工和结果汇总。"
        # 通用子Agent后续可通过 runtime 读取共享工具池；
        # 子Agent视图应使用 include_task_tool=False，避免子Agent继续委派子Agent。
        agent.runtime = self._runtime
        agent.skill_registry = self.skill_registry
        self.subagent_registry.register(agent, description=description)
        self._refresh_tools()

    # ============== 主入口 ==============

    def process(self, state: dict, user_input: str) -> dict:
        """处理用户输入（ReAct 循环）

        Args:
            state: 当前状态，包含 messages 对话历史等
            user_input: 用户最新输入

        Returns:
            dict: 包含 messages 列表，可选 pending_generation/prompt 等
        """
        # 1. 安全检查（最高优先级）
        if self._check_safety(user_input):
            return {
                "messages": [AIMessage(content=self._safety_response())],
                "safety_flag": True,
            }

        # 2. 如果在子Agent上下文中，直接委托给子Agent
        if state.get("active_subagent"):
            return self._handle_subagent(state, user_input)

        # 3. DeerFlow 风格 create_agent 全局运行时
        image_context = self._extract_vision_context(state)
        self._vision_context = image_context
        config = self._graph_config(state)
        graph_input = {
            "messages": self._graph_messages_for_input(state, user_input, config),
            "user_input": user_input,
            "vision_context": image_context,
        }

        try:
            graph_state = self.agent.invoke(graph_input, config=config)
        except Exception:
            return {
                "messages": [AIMessage(content=_FALLBACK_RESPONSE)],
                "vision_context": image_context,
            }

        return self._format_graph_result(graph_state, state, user_input, image_context)

    # ============== 系统提示词构建 ==============

    def _build_system_prompt(self, user_input: str | None = None) -> SystemMessage:
        """构建含工具描述的系统提示词"""
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

    def _build_messages(self, state: dict, user_input: str) -> list:
        """构建函数调用循环所需消息。"""
        messages = [self._build_system_prompt(user_input)]
        messages.extend(state.get("messages", [])[-10:])
        messages.append(HumanMessage(content=user_input))
        return messages

    def _graph_config(self, state: dict) -> dict:
        """Build LangGraph runnable config with thread isolation."""
        thread_id = state.get("thread_id")
        if thread_id:
            return {"configurable": {"thread_id": str(thread_id)}}

        default_config = {"configurable": {"thread_id": "default"}}
        if self._can_use_default_thread(state, default_config):
            return default_config
        return {"configurable": {"thread_id": self._next_transient_thread_id()}}

    def _next_transient_thread_id(self) -> str:
        self._transient_thread_index += 1
        return f"turn-{self._transient_thread_index}"

    def _can_use_default_thread(self, state: dict, config: dict) -> bool:
        """Keep the default thread for a matching legacy conversation only."""
        checkpoint_messages = self._checkpoint_messages(config)
        if not checkpoint_messages:
            return True
        history = list(state.get("messages", [])[-10:])
        if not history:
            return False
        return self._legacy_history_matches_checkpoint(history, checkpoint_messages)

    def _graph_messages_for_input(self, state: dict, user_input: str, config: dict) -> list:
        """Seed the graph from legacy history only when the checkpoint is empty."""
        checkpoint_messages = self._checkpoint_messages(config)
        history = list(state.get("messages", [])[-10:])
        if checkpoint_messages:
            return self._missing_legacy_messages(history, checkpoint_messages) + [HumanMessage(content=user_input)]
        return history + [HumanMessage(content=user_input)]

    def _checkpoint_messages(self, config: dict) -> list:
        """Read checkpoint messages without depending on saver internals."""
        try:
            snapshot = self.agent.get_state(config)
        except Exception:
            return []
        values = getattr(snapshot, "values", {}) or {}
        return values.get("messages") or []

    @staticmethod
    def _legacy_history_matches_checkpoint(history: list, checkpoint_messages: list) -> bool:
        history_signatures = HuesaeMainAgent._visible_message_signatures(history)
        checkpoint_signatures = HuesaeMainAgent._visible_message_signatures(checkpoint_messages)
        if not history_signatures:
            return False
        count = len(history_signatures)
        return checkpoint_signatures[-count:] == history_signatures

    @staticmethod
    def _missing_legacy_messages(history: list, checkpoint_messages: list) -> list:
        """Return visible legacy messages that are not already checkpointed."""
        checkpoint_visible = HuesaeMainAgent._visible_message_signatures(checkpoint_messages)
        history_signatures = HuesaeMainAgent._visible_message_signatures(history)
        if not history_signatures:
            return []

        matched_count = 0
        max_count = min(len(history_signatures), len(checkpoint_visible))
        for count in range(max_count, 0, -1):
            if history_signatures[:count] == checkpoint_visible[-count:]:
                matched_count = count
                break
            for start in range(0, len(checkpoint_visible) - count + 1):
                if history_signatures[:count] == checkpoint_visible[start : start + count]:
                    matched_count = count
                    break
            if matched_count:
                break
        return history[matched_count:]

    @staticmethod
    def _visible_message_signatures(messages: list) -> list[tuple[str, str]]:
        return [
            HuesaeMainAgent._message_signature(message)
            for message in messages
            if getattr(message, "type", None) in ("human", "ai")
            and str(getattr(message, "content", "") or "").strip()
        ]

    @staticmethod
    def _message_signature(message) -> tuple[str, str]:
        return getattr(message, "type", ""), str(getattr(message, "content", ""))

    def _format_graph_result(
        self,
        graph_state: dict,
        outer_state: dict,
        user_input: str,
        image_context: dict,
    ) -> dict:
        """Translate create_agent state back to the legacy process() contract."""
        messages = graph_state.get("messages") or []
        new_messages = self._new_messages_for_turn(messages, user_input)
        latest_tool_result = self._latest_tool_content(new_messages)

        task = self._subagent_task_from_messages(new_messages)
        if task is not None:
            subagent_type, description = task
            return self._start_subagent(outer_state, subagent_type, description)

        image_context = self._vision_context_from_messages(image_context, new_messages)
        self._vision_context = image_context
        ai_message = self._last_ai_message(new_messages) or self._last_ai_message(messages)
        content = str(getattr(ai_message, "content", "") or "") if ai_message is not None else ""
        if not content and latest_tool_result:
            content = self._format_last_tool_result(latest_tool_result)
        if not content:
            content = _FALLBACK_RESPONSE
        return {
            "messages": [AIMessage(content=content)],
            "vision_context": image_context,
        }

    @staticmethod
    def _new_messages_for_turn(messages: list, user_input: str) -> list:
        """Return messages appended by the current create_agent invocation."""
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

    def _vision_context_from_messages(self, image_context: dict, messages: list) -> dict:
        updated = dict(image_context or {})
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

    def _get_memory_context(self, user_input: str | None) -> str:
        """Fetch Honcho memory, using the current user input as a retrieval query."""
        if self.memory_service is None:
            return "暂无可用用户记忆。"
        try:
            return self.memory_service.get_context(user_input=user_input)
        except TypeError:
            return self.memory_service.get_context()

    @staticmethod
    def _format_last_tool_result(result) -> str:
        """ReAct 步数耗尽时，优先把已获得的工具结果返回给用户。"""
        text = str(result).strip()
        if not text:
            return "工具已经执行完成，但没有返回可展示的内容。"
        if text.startswith("工具执行失败") or text.startswith("错误："):
            return text
        return text

    # ============== 子Agent处理 ==============

    def _start_subagent(
        self,
        state: dict,
        subagent_type: str,
        description: str,
        initial_state: dict | None = None,
    ) -> dict:
        """启动子Agent处理任务
        """
        agent = self.subagent_registry.get(subagent_type)
        if not agent:
            available = self.subagent_registry.names()
            return {
                "messages": [AIMessage(
                    content=f"抱歉，暂时没有处理这种任务的子Agent~ 可用的子Agent：{available}"
                )]
            }

        # 创建子Agent的初始状态
        if subagent_type == "general":
            sub_state = {
                "messages": [],
            }
        else:
            sub_state = {
                "messages": [],
                "image_task_type": "generate_image",
                "image_phase": "collecting_prompt",
            }
        if initial_state:
            sub_state.update(initial_state)
        sub_state = self._sanitize_subagent_state(sub_state)

        # 调用子Agent
        sub_result = agent.process(sub_state, description)

        # 构建子Agent上下文
        subagent_context = {
            "agent_type": subagent_type,
            "state": sub_state,
            "history": [
                HumanMessage(content=description),
                AIMessage(content=sub_result.get("response", "")),
            ],
        }
        self._apply_subagent_state_update(subagent_context, sub_result)

        return self._format_subagent_result(sub_result, subagent_context)

    def _handle_subagent(self, state: dict, user_input: str) -> dict:
        """继续子Agent的对话"""
        subagent_context = dict(state.get("active_subagent", {}))
        subagent_context.pop("agent", None)
        agent_type = subagent_context.get("agent_type")
        agent = self.subagent_registry.get(agent_type)
        sub_state = self._sanitize_subagent_state(subagent_context.get("state", {}))
        history = subagent_context.get("history", [])

        if not agent:
            return {"messages": [AIMessage(content="子Agent状态异常，请重新开始~")]}

        # 更新子Agent状态
        sub_state["messages"] = history
        subagent_context["state"] = sub_state

        # 调用子Agent
        sub_result = agent.process(sub_state, user_input)
        self._apply_subagent_state_update(subagent_context, sub_result)
        sub_state = subagent_context.get("state", sub_state)

        # 更新历史
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=sub_result.get("response", "")))
        subagent_context["history"] = history
        sub_state["messages"] = history
        subagent_context["state"] = sub_state

        return self._format_subagent_result(sub_result, subagent_context)

    @staticmethod
    def _apply_subagent_state_update(subagent_context: dict, sub_result: dict) -> None:
        """把子Agent返回的状态更新合并到 active_subagent 中。"""
        state_update = (sub_result.get("data") or {}).get("state_update") or {}
        if not state_update:
            return
        state = subagent_context.setdefault("state", {})
        state.update(state_update)
        subagent_context["state"] = HuesaeMainAgent._sanitize_subagent_state(state)

    @staticmethod
    def _sanitize_subagent_state(state: dict) -> dict:
        """Keep active_subagent state checkpoint-friendly."""
        sanitized = dict(state or {})
        sanitized.pop("agent", None)
        sanitized.pop("runtime", None)
        sanitized.pop("skill_registry", None)
        return sanitized

    def _format_subagent_result(self, sub_result: dict, subagent_context: dict) -> dict:
        """把子Agent标准结果转换成主Agent对外返回格式。"""
        action = sub_result.get("action", "")
        response = sub_result.get("response", "")
        agent_type = subagent_context.get("agent_type", "")

        if agent_type == "general":
            return {
                "messages": [AIMessage(content=response)],
                "clear_subagent": True,
            }

        if action in ("ask_prompt", "recommend", "ask_confirm"):
            return {
                "messages": [AIMessage(content=response)],
                "active_subagent": subagent_context,
            }

        if action == "generate":
            return {
                "messages": [AIMessage(content=response or "图片正在生成中，请稍等~")],
                "pending_generation": True,
                "prompt": sub_result.get("prompt", ""),
                "size": self._sub_result_data_value(sub_result, "size", "2K"),
                "output_format": self._sub_result_data_value(sub_result, "output_format", "jpeg"),
                "is_batch": self._sub_result_data_value(sub_result, "is_batch", False),
                "active_subagent": subagent_context,
            }

        if action == "finish":
            return {
                "messages": [AIMessage(content=response)],
                "clear_subagent": True,
            }

        return {
            "messages": [AIMessage(content=response or "请告诉我您想要生成什么样的图片？")],
            "active_subagent": subagent_context,
        }

    @staticmethod
    def _sub_result_data_value(result: dict, key: str, default):
        """读取子Agent标准返回结果中的 data 字段。"""
        data = result.get("data") or {}
        return data.get(key, default)

    # ============== 异步生图（供 chat_loop 调用）=============

    async def execute_image_generation(
        self,
        prompt: str,
        size: str = "2K",
        output_format: str = "jpeg",
        is_batch: bool = False,
    ) -> dict:
        """执行生图并返回结果（供外部调用）

        保留此方法供 chat_loop 在 pending_generation 场景下调用。
        """
        agent = self.subagent_registry.get("image")
        if not agent:
            raise ValueError("Image agent not registered")

        wrap_msg = self._create_wrap_message()

        if is_batch:
            generations = await agent.generate_images(
                prompt=prompt,
                size=size,
                output_format=output_format,
            )
            return {
                "wrap_message": wrap_msg,
                "image_urls": [g.url for g in generations],
                "confirm_message": "这些图片可以吗？如果满意请回复“可以”，也可以说“换一组”或重新输入提示词~",
                "subagent_state_update": {
                    "image_phase": "awaiting_image_confirm",
                    "last_image_urls": [g.url for g in generations],
                    "last_generation_succeeded": True,
                },
            }

        generation = await agent.generate_image(
            prompt=prompt,
            size=size,
            output_format=output_format,
        )
        return {
            "wrap_message": wrap_msg,
            "image_url": generation.url,
            "confirm_message": "这张图片可以吗？如果满意请回复“可以”，也可以说“换一张”或重新输入提示词~",
            "subagent_state_update": {
                "image_phase": "awaiting_image_confirm",
                "last_image_urls": [generation.url],
                "last_generation_succeeded": True,
            },
        }

    # ============== 角色语气包装 ==============

    def _create_wrap_message(self) -> str:
        """用主Agent的角色语气生成图片展示语"""
        from .prompts import get_character_system_message

        character_msg = get_character_system_message(self.character_id)
        wrap_prompt = (
            "用户请求的图片已经生成完成了！"
            "请用你温柔可爱的语气说一句简短的展示语，"
            "比如'这是生成好的图片哦~'、'快来看看吧~'等"
        )
        messages = [character_msg, HumanMessage(content=wrap_prompt)]
        response = self.llm.invoke(messages)
        return response.content

    # ============== 聊天回复 ==============

    def _chat_reply(self, state: dict, user_input: str) -> str:
        """主Agent直接聊天回复"""
        from .prompts import get_character_system_message

        character_msg = get_character_system_message(self.character_id)
        messages = [character_msg] + state.get("messages", []) + [HumanMessage(content=user_input)]
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception:
            return _FALLBACK_RESPONSE

    def _extract_vision_context(self, state: dict) -> dict:
        """从状态中提取图像上下文。"""
        vision_context = state.get("vision_context")
        if isinstance(vision_context, dict):
            return vision_context
        if self._vision_context:
            return dict(self._vision_context)
        return {}

    def _get_vision_context(self) -> dict:
        """从运行时缓存中读取图像上下文。"""
        return getattr(self, "_vision_context", {}) or {}

    def _update_vision_context(self, image_context: dict, tool_name: str, tool_args: dict, result_text: str) -> dict:
        """根据识图工具结果更新轻量图像上下文。"""
        updated = dict(image_context or {})
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

    def _looks_like_image_input(self, tool_name: str, tool_args: dict) -> bool:
        """判断当前工具调用是否带有图片路径。"""
        if tool_name == "reverse_image_prompt":
            return True
        return any(key in tool_args for key in IMAGE_CONTEXT_PATH_KEYS)

    def _collect_image_paths(self, tool_args: dict) -> list[str]:
        """把工具参数中的图片路径收集成列表。"""
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
        """把图像上下文压缩成系统提示词可读的摘要。"""
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

    # ============== 安全处理 ==============

    def _check_safety(self, user_input: str) -> bool:
        """安全检查"""
        content = user_input.lower()
        return any(kw in content for kw in _SAFE_KEYWORDS)

    def _safety_response(self) -> str:
        """安全回复"""
        return (
            "*轻轻握住你的手*\n\n"
            "我在这里陪着你，你不是一个人...\n\n"
            "如果你感到痛苦或绝望，请一定要寻求专业帮助：\n"
            "- 心理危机干预热线：400-161-9995\n"
            "- 北京心理危机研究与干预中心：010-82951332\n"
            "- 生命热线：400-821-1215\n\n"
            "你的生命很珍贵，请不要独自承受这些。"
        )


# ============== 工厂函数 ==============

def create_main_agent(
    llm: BaseChatModel | None = None,
    character_id: str = "gentle_sister",
    mcp_tools_loader=None,
    skill_registry: SkillRegistry | None = None,
    memory_service: HonchoMemoryService | None = None,
    middleware_pipeline: MiddlewarePipeline | None = None,
) -> HuesaeMainAgent:
    """创建主Agent工厂函数

    Args:
        llm: 大语言模型，默认使用DeepSeek
        character_id: 角色ID

    Returns:
        HuesaeMainAgent: 主Agent实例
    """
    if llm is None:
        try:
            from huesae.models.models_factory import create_chat_model
        except ImportError:
            from huesaeagents.huesae.models.models_factory import create_chat_model
        llm = create_chat_model("deepseek")

    return HuesaeMainAgent(
        llm=llm,
        character_id=character_id,
        mcp_tools_loader=mcp_tools_loader,
        skill_registry=skill_registry,
        memory_service=memory_service,
        middleware_pipeline=middleware_pipeline,
    )
