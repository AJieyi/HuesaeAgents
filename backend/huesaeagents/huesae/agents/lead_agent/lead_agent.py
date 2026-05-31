"""主 Agent 运行时。

这个文件是 HuesaeAgents 的主调度层：
- 用 LangChain ``create_agent`` 创建可调用工具的 ReAct Agent。
- 用 LangGraph checkpointer 按 ``thread_id`` 保存每个会话的短期状态。
- 负责安全检查、工具视图刷新、MCP 懒加载、Skill/记忆/视觉上下文注入。
- 负责通过 ``task_tool`` 把生图、复杂任务等委派给子 Agent。

注意：主 Agent 不直接把图片二进制塞进 messages，上下文里只保存轻量引用
（例如图片路径、反推提示词、artifacts）。
"""

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

# 主 Agent 兜底回复。只在 LangGraph/模型调用异常、或者没有可展示结果时使用。
_FALLBACK_RESPONSE = "抱歉，我刚刚有点卡住了，请稍后再试一次，或者把需求再发我一遍。"

# 这些字段名用于从工具参数中识别“用户提供了图片路径/图片列表”。
# 识别到后会写入 vision_context，供下一轮系统提示词引用。
IMAGE_CONTEXT_PATH_KEYS = (
    "image_path",
    "image_paths",
    "last_image_path",
    "last_image_paths",
)


class HuesaeMainAgent:
    """Huesae 主 Agent。

    对外主要入口是 :meth:`invoke`。

    主 Agent 自己不硬编码“用户说什么就调用什么工具”的分流逻辑，而是把：
    - 当前可见工具列表
    - 子 Agent 描述
    - Skill 列表
    - 记忆上下文
    - 图像上下文

    动态注入系统提示词，让 LLM 通过 LangChain tool calling 自己选择行动。
    LangGraph 负责按 ``thread_id`` 保存 messages、current_subagent、artifacts 等状态。
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
        # 没有显式传入模型时，默认创建 DeepSeek 聊天模型。
        # 这里放在函数内部导入，是为了减少模块加载时的依赖耦合，也方便测试替换。
        if llm is None:
            try:
                from huesae.models.models_factory import create_chat_model
            except ImportError:
                from huesaeagents.huesae.models.models_factory import create_chat_model
            llm = create_chat_model("deepseek")

        # self.llm 保留原始模型，供工具/子模块复用。
        # self._agent_model 必须是 LangChain BaseChatModel，供 create_agent 使用。
        self.llm = llm
        self._agent_model = ensure_chat_model(llm)

        # 角色 ID 控制主 Agent 的回复语气，由 prompts.py 拼进系统提示词。
        self.character_id = character_id

        # SkillRegistry 不是在这里扫描路径；调用方通常传入已经初始化好的注册表。
        # 如果 chat_loop.py 中调用 SkillRegistry()，它会默认扫描项目根目录 skills/。
        self.skill_registry = skill_registry

        # 长期记忆服务可选。不可用时系统提示词里只会注入“暂无可用用户记忆”。
        self.memory_service = memory_service

        # 中间件管道用于挂载 token usage 等横切逻辑。
        self._middleware_pipeline = middleware_pipeline or build_middlewares()

        # 子 Agent 注册表：只负责“有哪些子 Agent、名字是什么、描述是什么”。
        self.subagent_registry = SubAgentRegistry()

        # 子 Agent 执行器：负责根据注册表查找子 Agent，并处理启动/续聊/结果归一化。
        self.subagent_executor = SubagentExecutor(self.subagent_registry)

        # LangGraph 内存检查点。状态按 thread_id 隔离；进程退出后会丢失。
        self._checkpointer = InMemorySaver()
        self.agent = None

        # SharedToolRuntime 的可选依赖。
        # 把“加载器函数”和 Skill 注册表传给工具运行时。
        # 真正的 MCP 懒加载发生在 load_mcp_tools_tool 被调用之后。
        runtime_kwargs = {}
        if mcp_tools_loader is not None:
            runtime_kwargs["mcp_tools_loader"] = mcp_tools_loader
        if skill_registry is not None:
            runtime_kwargs["skill_registry"] = skill_registry

        # 共享工具运行时统一管理内置工具、MCP 工具、Skill 工具入口。
        # 主 Agent 和 general 子 Agent 可以共享这套工具视图。
        self._runtime = build_shared_runtime(self.llm, self.subagent_registry, **runtime_kwargs)

        # tools 是传给 create_agent 的工具列表；tool_map 用于中间件根据工具名找到工具对象。
        self.tools = []
        self.tool_map = {}

        # 当前轮临时视觉上下文缓存。持久版本会写入 LangGraph 的 vision_context。
        self._vision_context: dict[str, Any] = {}

        # 初始化内置工具并编译 LangGraph Agent。
        self._refresh_tools()

    def _refresh_tools(self) -> None:
        """刷新主 Agent 的基础工具视图，并重新编译 LangGraph Agent。

        初始工具视图不包含 MCP 工具，避免启动时就拉起所有 MCP server。
        同时会隐藏底层生图工具，强制生图任务走 image 子 Agent 的确认闭环。
        """
        # 子 Agent 或 Skill 注册表可能在运行时变化，刷新工具前先同步到 runtime。
        self._runtime.subagent_registry = self.subagent_registry
        self._runtime.skill_registry = self.skill_registry
        self._runtime.refresh_builtin_tools()

        # include_mcp=False：启动阶段只暴露内置工具，不做 MCP discovery。
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
        """懒加载 MCP 工具，然后刷新主 Agent 工具视图。

        当前实现是一旦需要 MCP，就加载所有 enabled=true 的 MCP server，
        不是按某一个具体 MCP server 精准加载。
        """
        self._runtime.subagent_registry = self.subagent_registry
        self._runtime.skill_registry = self.skill_registry
        self._runtime.refresh_builtin_tools()

        # include_mcp=True 会触发 SharedToolRuntime 加载/读取 MCP 工具缓存。
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
        """用当前工具列表重新编译 LangChain/LangGraph Agent。

        工具列表、系统提示词、MCP 状态变化后，需要重新 create_agent，
        这样模型下一轮才能看到最新工具 schema。
        """
        self.agent = create_agent(
            model=self._agent_model,
            tools=self.tools,
            system_prompt=self._build_system_prompt(),
            middleware=[
                RuntimeToolMiddleware(
                    # 这些 lambda 让中间件每次调用时读取“最新”的工具视图，
                    # 而不是绑定编译瞬间的旧列表。
                    tools=lambda: self.tools,
                    tool_map=lambda: self.tool_map,
                    system_message=self._build_system_prompt,

                    # 当模型调用 load_mcp_tools_tool 或未知 MCP 工具时，
                    # 中间件会通过这个回调触发 MCP 懒加载。
                    ensure_mcp_tools=self._refresh_tools_with_mcp,
                    is_mcp_loaded=lambda: self._runtime.mcp_loaded,
                    has_mcp_tools=lambda: bool(self._runtime._mcp_tools),

                    # task_tool 不走普通工具函数返回，而是在中间件里直接进入子 Agent 执行器。
                    subagent_executor=self.subagent_executor,
                ),
                *adapt_middlewares(self._middleware_pipeline.middlewares),
            ],
            # HuesaeThreadState 扩展了 LangChain AgentState，增加 current_subagent、
            # vision_context、artifacts 等项目自定义字段。
            state_schema=HuesaeThreadState,
            checkpointer=self._checkpointer,
            name="huesae_main_agent",
        )

    def register_sub_agent(self, agent: BaseSubAgent) -> None:
        """注册一个可被 task_tool 委派的子 Agent。"""
        description = None
        if agent.name == "image":
            description = "生图对话Agent，处理追问、推荐、扩写、确认、单图和组图生成。"
        elif agent.name == "general":
            description = "通用任务Agent，处理复杂通用任务、工具链执行、资料加工和结果汇总。"

        # 把共享 runtime 和 skill_registry 注入子 Agent，避免每个子 Agent 重复构造工具池。
        agent.runtime = self._runtime
        agent.skill_registry = self.skill_registry
        self.subagent_registry.register(agent, description=description)

        # 子 Agent 注册后，task_tool 的描述和可用子 Agent 列表会变化，所以刷新工具与图。
        self._refresh_tools()

    def invoke(self, user_input: str, *, thread_id: str = "default") -> dict:
        """运行一轮用户输入。

        Args:
            user_input: 用户本轮输入。
            thread_id: LangGraph checkpoint 线程 ID，用于隔离不同会话。

        Returns:
            返回给调用方展示的消息，以及可选的 ``artifacts`` / ``vision_context``。
            长流程子 Agent 状态保存在 checkpoint 的 ``current_subagent`` 里。
        """
        thread_id = str(thread_id or "default")
        config = self._graph_config(thread_id)

        # 安全检查优先级最高：命中后不进入模型、不调用工具。
        if self._check_safety(user_input):
            message = AIMessage(content=self._safety_response())
            self._append_graph_messages(config, [HumanMessage(content=user_input), message])
            return {"messages": [message], "safety_flag": True}

        # 如果上一轮已经进入子 Agent 流程，本轮直接续聊子 Agent，
        # 避免主 Agent 重新判断导致确认闭环被打断。
        current_subagent = self._checkpoint_value(config, "current_subagent")
        if current_subagent:
            return self._continue_subagent(user_input, thread_id, config, current_subagent)

        # 取出上一轮保存的视觉上下文，动态写入系统提示词。
        vision_context = self._checkpoint_value(config, "vision_context") or {}
        self._vision_context = dict(vision_context)

        # create_agent 只需要传入本轮新增 HumanMessage；历史由 LangGraph checkpointer 合并。
        graph_input = {
            "messages": [HumanMessage(content=user_input)],
            "user_input": user_input,
            "vision_context": vision_context,
        }

        try:
            graph_state = self.agent.invoke(graph_input, config=config)
        except Exception:
            # 兜底：模型、工具或 LangGraph 执行异常时不向用户抛栈。
            message = AIMessage(content=_FALLBACK_RESPONSE)
            self._append_graph_messages(config, [message])
            return {"messages": [message], "vision_context": vision_context}

        return self._format_graph_result(graph_state, user_input, thread_id, config, vision_context)

    def stream(self, user_input: str, *, thread_id: str = "default"):
        """提供一个“类流式”接口。

        当前实现并不是真正 token 级流式，而是先完整执行 invoke，
        再逐条 yield 最终消息，方便调用方保持统一消费方式。
        """
        result = self.invoke(user_input, thread_id=thread_id)
        for message in result.get("messages", []):
            yield message

    def get_state(self, thread_id: str = "default") -> dict:
        """读取指定 thread_id 的 LangGraph checkpoint 状态。"""
        return dict(self._snapshot_values(self._graph_config(thread_id)))

    def _format_graph_result(
        self,
        graph_state: dict,
        user_input: str,
        thread_id: str,
        config: dict,
        vision_context: dict,
    ) -> dict:
        """把 LangGraph 原始状态整理成对外返回格式。

        LangChain Agent 的 messages 里会包含 HumanMessage、AIMessage、ToolMessage。
        这里会抽取本轮新增消息，识别子 Agent 委派信号，更新视觉上下文，
        最后只返回需要展示给用户的 AIMessage 和 artifacts。
        """
        messages = graph_state.get("messages") or []
        new_messages = self._new_messages_for_turn(messages, user_input)
        latest_tool_result = self._latest_tool_content(new_messages)

        # 工具调用可能带来新的图片路径、反推提示词等，整理后写回 checkpoint。
        updated_vision_context = self._vision_context_from_messages(vision_context, new_messages)
        self._persist_runtime_state(config, vision_context=updated_vision_context)

        # 如果子 Agent 通过中间件已经写入 current_subagent/artifacts，
        # 这里直接把它们随最终消息返回给上层。
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

        # 普通主 Agent 回复路径：优先取最后一条 AIMessage；
        # 如果模型只调用了工具但没有自然语言回复，则把最后的工具结果格式化给用户。
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
        """继续一个未完成的子 Agent 会话。"""
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
        """提交子 Agent 执行结果到主 Agent 的 LangGraph 状态。"""
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
        """把运行时字段写入 LangGraph checkpoint。

        这里用 ``...`` 作为“未传入”的哨兵值，区分：
        - 不更新某字段
        - 明确把 current_subagent 清成 None
        - 明确把 artifacts 清成空列表
        """
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
        """追加消息到 LangGraph checkpoint。"""
        if messages:
            self.agent.update_state(config, {"messages": messages})

    def _checkpoint_value(self, config: dict, key: str):
        """从 checkpoint 快照中读取单个字段。"""
        return self._snapshot_values(config).get(key)

    def _snapshot_values(self, config: dict) -> dict:
        """读取 LangGraph 当前快照值，失败时返回空字典。"""
        try:
            snapshot = self.agent.get_state(config)
        except Exception:
            return {}
        return dict(getattr(snapshot, "values", {}) or {})

    @staticmethod
    def _graph_config(thread_id: str) -> dict:
        """构造 LangGraph 使用的 thread_id 配置结构。"""
        return {"configurable": {"thread_id": str(thread_id or "default")}}

    def _build_system_prompt(self, user_input: str | None = None) -> SystemMessage:
        """构建动态系统提示词。

        系统提示词不是固定字符串，而是每轮根据当前运行状态重新拼装：
        工具是否包含 MCP、有哪些子 Agent、有哪些 Skill、用户记忆和图像上下文都会影响它。
        """
        from .prompts import build_main_system_message

        # 当前可见工具列表与参数名，帮助模型正确选择 tool_call。
        tools_description = self._runtime.format_tools_for_prompt(
            include_mcp=self._runtime.mcp_loaded,
            include_task_tool=True,
            exclude_names=MAIN_AGENT_EXCLUDED_TOOL_NAMES,
        )

        # 工具使用约束：例如生图不要直接调用底层工具，而是委派 image 子 Agent。
        tool_constraints = self._runtime.format_tool_constraints(
            include_mcp=self._runtime.mcp_loaded,
            include_task_tool=True,
            exclude_names=MAIN_AGENT_EXCLUDED_TOOL_NAMES,
        )

        # MCP 未加载/已加载/无工具时，对模型的选择原则不同。
        mcp_tool_principles = self._runtime.format_mcp_tool_principles()

        # 子 Agent 描述会进入 task_tool 和系统提示词，帮助模型判断何时委派。
        subagents_description = self.subagent_registry.format_for_prompt()

        # Skill 只作为“工作指令”提示给模型；真正读取完整内容要调用 read_skill_tool。
        skills_section = (
            self.skill_registry.format_for_prompt()
            if self.skill_registry is not None
            else "暂无可用 Skills。"
        )

        # 长期记忆来自 Honcho。不可用时保持空文本，避免阻断主流程。
        memory_context_section = (
            self._get_memory_context(user_input)
            if self.memory_service is not None and self.memory_service.enabled
            else "暂无可用用户记忆。"
        )

        # 视觉上下文只放轻量文本引用，不放图片二进制。
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
        """读取长期记忆上下文，兼容不同 memory_service 方法签名。"""
        if self.memory_service is None:
            return "暂无可用用户记忆。"
        try:
            return self.memory_service.get_context(user_input=user_input)
        except TypeError:
            return self.memory_service.get_context()

    @staticmethod
    def _new_messages_for_turn(messages: list, user_input: str) -> list:
        """从完整 messages 中截取本轮用户输入之后产生的新消息。"""
        start_index = 0
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if getattr(message, "type", None) == "human" and str(message.content) == user_input:
                start_index = index + 1
                break
        return messages[start_index:]

    @staticmethod
    def _last_ai_message(messages: list) -> AIMessage | None:
        """找到消息列表中最后一条 AIMessage。"""
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None

    @staticmethod
    def _latest_tool_content(messages: list) -> str | None:
        """找到本轮最后一个工具返回文本。"""
        for message in reversed(messages):
            if getattr(message, "type", None) == "tool":
                return str(message.content)
        return None

    def _vision_context_from_messages(self, vision_context: dict, messages: list) -> dict:
        """根据本轮工具调用更新视觉上下文。

        LangChain 的 ToolMessage 通常只带 tool_call_id；具体工具名和参数在前面的 AIMessage.tool_calls。
        因此这里先建立 call_id -> tool_call 的映射，再把工具结果和参数合并分析。
        """
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
        """把工具结果转换成可展示文本。"""
        text = str(result).strip()
        if not text:
            return "工具已经执行完成，但没有返回可展示的内容。"
        return text

    def _get_vision_context(self) -> dict:
        """读取当前轮缓存的视觉上下文。"""
        return getattr(self, "_vision_context", {}) or {}

    def _update_vision_context(self, vision_context: dict, tool_name: str, tool_args: dict, result_text: str) -> dict:
        """根据某次工具调用更新 vision_context。

        reverse_image_prompt 会保存图片路径和反推提示词；
        其他带 image_path/image_paths 参数的工具则只保存最近图片引用。
        """
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
        """判断某次工具调用是否包含图片输入。"""
        if tool_name == "reverse_image_prompt":
            return True
        return any(key in tool_args for key in IMAGE_CONTEXT_PATH_KEYS)

    @staticmethod
    def _collect_image_paths(tool_args: dict) -> list[str]:
        """从工具参数中收集单图或多图路径。"""
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
        """把 vision_context 压缩成系统提示词可读的文本。"""
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
        """简单关键词安全检查。"""
        content = user_input.lower()
        return any(keyword in content for keyword in _SAFE_KEYWORDS)

    @staticmethod
    def _safety_response() -> str:
        """安全检查命中后的固定关怀回复。"""
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
