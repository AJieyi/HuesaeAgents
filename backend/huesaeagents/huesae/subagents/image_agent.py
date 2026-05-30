"""生图子Agent

主Agent的委派组件，负责处理生图相关的多轮对话。
会把阶段信息写入主Agent维护的子Agent状态中，确保确认闭环稳定。

支持流程：追问 → 推荐 → 扩写 → 确认描述 → 生图 → 确认图片 → 结束
"""
import asyncio
from typing import Literal, TypedDict

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .base import BaseSubAgent
from .image import expand_prompt
from .image.providers import ImageProvider, GenerationResult
from .image.prompts import IMAGE_CONVERSATION_SYSTEM_MESSAGE


# ============== LLM决策模型 ==============

class ImageDecision(BaseModel):
    """子Agent每回合的决策"""

    thought: str = Field(description="分析当前对话状态和用户需求")
    action: Literal[
        "ask_prompt",       # 追问：缺少提示词，请用户描述
        "recommend",        # 推荐：主动生成推荐提示词供选择
        "expand",           # 扩写：将简短描述扩写为详细提示词
        "ask_confirm",      # 确认：推荐/扩写后询问用户是否满意
        "generate",         # 生图：调用provider生成图片
        "show_image",       # 展示：图片已生成，展示给用户
        "finish",           # 结束：对话完成
    ] = Field(description="下一步动作")
    response: str = Field(description="给用户的回复消息，用温柔可爱的二次元语气")
    prompt: str | None = Field(default=None, description="当前确认的提示词")
    provider: str | None = Field(default=None, description="选择的生图工具（当前固定doubao）")
    size: str | None = Field(default="2K", description="图片尺寸，支持 1K, 2K, 3K, 4K 等")
    output_format: str | None = Field(default="jpeg", description="输出图片格式，支持 jpeg, png")
    is_batch: bool | None = Field(default=False, description="是否使用组图模式，用户明确表达需要多张图片时为true")


class ImageUserIntent(BaseModel):
    """用户在生图确认流程中的语义意图。"""

    thought: str = Field(description="结合当前阶段分析用户真实意图，不能用关键词机械匹配")
    intent: Literal[
        "confirm",              # 确认当前提示词或图片
        "reject",               # 否定当前提示词或图片，但尚未给出明确新需求
        "end",                  # 明确结束当前生图任务
        "regenerate",           # 保持当前提示词重新生成
        "expand_prompt",        # 扩写当前提示词
        "replace_prompt",       # 用用户给出的新提示词替换当前提示词
        "provide_prompt",       # 用户提供了可用于生图的描述
        "request_recommendation",  # 用户希望系统推荐主题或描述
        "clarify",              # 用户意图不明确，需要澄清
        "other",                # 与当前确认流程关系不明确
    ] = Field(description="用户语义意图")
    replacement_prompt: str | None = Field(
        default=None,
        description="当 intent=replace_prompt 或 provide_prompt 时，提取出的干净图片描述",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="意图判断置信度")
    clarification_question: str | None = Field(
        default=None,
        description="当 intent=clarify 或置信度较低时，给用户的澄清问题",
    )


# ============== 标准化返回格式 ==============

def _make_result(
    action: str,
    response: str,
    prompt: str | None = None,
    provider: str | None = None,
    **kwargs,
) -> dict:
    """构造标准化的子Agent返回结果"""
    return {
        "action": action,
        "response": response,
        "prompt": prompt,
        "provider": provider,
        "data": kwargs,
    }


class ImageWorkflowState(TypedDict, total=False):
    """LangGraph 生图对话状态。"""

    state: dict
    user_input: str
    decision: dict
    user_intent: dict
    result: dict


ImageWorkflowRoute = Literal[
    "finish_task",
    "clarify_user",
    "prompt_confirmation",
    "image_confirmation",
    "general_decision",
]


# ============== 生图子Agent ==============

class ImageSubAgent(BaseSubAgent):
    """生图子Agent

    无状态组件，每次调用接收完整对话历史，用 LLM 分析后输出决策。
    所有回复交给主Agent包装展示。

    典型调用：主Agent把完整对话历史和用户最新输入传入 process，
    子Agent返回统一结果，由主Agent决定是否继续追问或触发生图。
    """

    name = "image"

    def __init__(
        self,
        llm: BaseChatModel,
        providers: list[ImageProvider] | None = None,
        default_provider: str = "doubao",
    ):
        self.llm = llm
        self.providers: dict[str, ImageProvider] = {}
        if providers:
            for p in providers:
                self.register_provider(p)
        self.default_provider = default_provider
        self._checkpointer = InMemorySaver()
        self._transient_thread_index = 0
        self.workflow = self._build_workflow()

    def register_provider(self, provider: ImageProvider) -> None:
        """注册生图Provider（扩展点）"""
        self.providers[provider.name] = provider

    # ============== 主入口：标准化接口 ==============

    def invoke(self, user_input: str, *, thread_id: str, state: dict | None = None) -> dict:
        """Run the image graph with an isolated subagent thread state."""
        return self._run_workflow(
            dict(state or {}),
            user_input,
            thread_id=thread_id,
            execute_generation=True,
        )

    def _run_workflow(
        self,
        state: dict,
        user_input: str,
        *,
        execute_generation: bool,
        thread_id: str | None = None,
    ) -> dict:
        config = self._graph_config(thread_id)
        sub_state = self._merge_checkpoint_state(config, state)
        workflow_state = self.workflow.invoke({
            "state": sub_state,
            "user_input": user_input,
        }, config=config)
        result = workflow_state["result"]
        if execute_generation and result.get("action") == "generate":
            result = asyncio.run(self._execute_generation_result(result))
        self._persist_conversation_state(config, sub_state, user_input, result)
        return result

    def _graph_config(self, thread_id: str | None) -> dict:
        if not thread_id:
            self._transient_thread_index += 1
            thread_id = f"image-transient-{self._transient_thread_index}"
        return {"configurable": {"thread_id": str(thread_id)}}

    def _merge_checkpoint_state(self, config: dict, state: dict) -> dict:
        checkpoint_state = self.get_state(config["configurable"]["thread_id"])
        merged = dict(checkpoint_state)
        merged.update(state or {})
        return merged

    def get_state(self, thread_id: str) -> dict:
        """Return the image workflow's local persisted state."""
        try:
            snapshot = self.workflow.get_state({"configurable": {"thread_id": str(thread_id)}})
        except Exception:
            return {}
        values = getattr(snapshot, "values", {}) or {}
        return dict(values.get("state") or {})

    def _persist_conversation_state(self, config: dict, state: dict, user_input: str, result: dict) -> None:
        updated = dict(state or {})
        updated.update((result.get("data") or {}).get("state_update") or {})
        history = list(updated.get("messages") or [])
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=str(result.get("response") or "")))
        updated["messages"] = history
        self.workflow.update_state(config, {"state": updated})

    async def _execute_generation_result(self, result: dict) -> dict:
        """Execute image provider work inside the image subagent graph boundary."""
        prompt = result.get("prompt") or ""
        data = dict(result.get("data") or {})
        size = data.get("size", "2K")
        output_format = data.get("output_format", "jpeg")
        is_batch = data.get("is_batch", False)

        if is_batch:
            generations = await self.generate_images(
                prompt=prompt,
                provider_name=result.get("provider"),
                size=size,
                output_format=output_format,
            )
            image_urls = [generation.url for generation in generations]
        else:
            generation = await self.generate_image(
                prompt=prompt,
                provider_name=result.get("provider"),
                size=size,
                output_format=output_format,
            )
            image_urls = [generation.url]

        data.update(
            {
                "image_urls": image_urls,
                "artifacts": [{"type": "image", "url": url} for url in image_urls],
                "state_update": {
                    **(data.get("state_update") or {}),
                    "image_phase": "awaiting_image_confirm",
                    "last_image_urls": image_urls,
                    "last_generation_succeeded": True,
                },
            }
        )
        response = result.get("response") or "图片已经生成完成啦~"
        if image_urls:
            image_lines = "\n".join(f"[图片] {url}" for url in image_urls)
            confirm = (
                "这些图片可以吗？如果满意请回复“可以”，也可以说“换一组”或重新输入提示词~"
                if is_batch
                else "这张图片可以吗？如果满意请回复“可以”，也可以说“换一张”或重新输入提示词~"
            )
            response = f"{response}\n\n{image_lines}\n\n{confirm}"

        return {
            **result,
            "action": "show_image",
            "response": response,
            "data": data,
        }

    # ============== LangGraph 工作流 ==============

    def _build_workflow(self):
        """构建生图确认流程图。

        LLM 负责理解用户意图，LangGraph 节点负责把确认状态约束住：
        生成任务确认提示词后必须生图，确认图片后才结束。
        """
        workflow = StateGraph(ImageWorkflowState)
        workflow.add_node("llm_decide", self._workflow_decide)
        workflow.add_node("classify_user_intent", self._workflow_classify_user_intent)
        workflow.add_node("finish_task", self._workflow_finish_task)
        workflow.add_node("clarify_user", self._workflow_clarify_user)
        workflow.add_node("prompt_confirmation", self._workflow_prompt_confirmation)
        workflow.add_node("image_confirmation", self._workflow_image_confirmation)
        workflow.add_node("general_decision", self._workflow_general_decision)
        workflow.add_edge(START, "llm_decide")
        workflow.add_edge("llm_decide", "classify_user_intent")
        workflow.add_conditional_edges(
            "classify_user_intent",
            self._route_after_intent,
            {
                "finish_task": "finish_task",
                "clarify_user": "clarify_user",
                "prompt_confirmation": "prompt_confirmation",
                "image_confirmation": "image_confirmation",
                "general_decision": "general_decision",
            },
        )
        workflow.add_edge("finish_task", END)
        workflow.add_edge("clarify_user", END)
        workflow.add_edge("prompt_confirmation", END)
        workflow.add_edge("image_confirmation", END)
        workflow.add_edge("general_decision", END)
        return workflow.compile(checkpointer=self._checkpointer)

    def _workflow_decide(self, graph_state: ImageWorkflowState) -> dict:
        """调用 LLM 输出候选决策。"""
        state = graph_state.get("state", {})
        user_input = graph_state.get("user_input", "")
        return {"decision": self._decide(state, user_input).model_dump()}

    def _workflow_classify_user_intent(self, graph_state: ImageWorkflowState) -> dict:
        """使用 LLM 识别用户在确认流程中的语义意图。"""
        state = graph_state.get("state", {})
        user_input = graph_state.get("user_input", "")
        decision = self._decision_from_state(graph_state)
        return {"user_intent": self._classify_user_intent(state, user_input, decision).model_dump()}

    def _route_after_intent(self, graph_state: ImageWorkflowState) -> ImageWorkflowRoute:
        """根据 LLM 识别出的语义意图和当前阶段路由。"""
        state = graph_state.get("state", {})
        user_intent = self._user_intent_from_state(graph_state)
        phase = state.get("image_phase", "collecting_prompt")

        if user_intent.intent == "end":
            return "finish_task"

        if phase == "awaiting_image_confirm":
            return "image_confirmation"

        if phase == "awaiting_prompt_confirm":
            return "prompt_confirmation"

        return "general_decision"

    def _workflow_finish_task(self, graph_state: ImageWorkflowState) -> dict:
        """结束当前生图任务。"""
        return {"result": _make_result(
            action="finish",
            response="好的，那这次生图任务先结束啦~",
            state_update={"image_phase": "finished"},
        )}

    def _workflow_clarify_user(self, graph_state: ImageWorkflowState) -> dict:
        """向用户澄清当前确认流程的下一步。"""
        state = graph_state.get("state", {})
        user_intent = self._user_intent_from_state(graph_state)
        question = user_intent.clarification_question or self._default_clarification_question(state)
        return {"result": _make_result(
            action="ask_confirm",
            response=question,
            prompt=state.get("image_prompt") or state.get("last_prompt"),
            provider="doubao",
            state_update={"image_phase": state.get("image_phase", "collecting_prompt")},
        )}

    def _workflow_prompt_confirmation(self, graph_state: ImageWorkflowState) -> dict:
        """处理提示词确认阶段。"""
        state = graph_state.get("state", {})
        user_input = graph_state.get("user_input", "")
        decision = self._decision_from_state(graph_state)
        user_intent = self._user_intent_from_state(graph_state)
        return {"result": self._handle_prompt_confirmation_phase(
            state,
            user_input,
            decision,
            user_intent,
        )}

    def _workflow_image_confirmation(self, graph_state: ImageWorkflowState) -> dict:
        """处理图片确认阶段。"""
        state = graph_state.get("state", {})
        user_input = graph_state.get("user_input", "")
        decision = self._decision_from_state(graph_state)
        user_intent = self._user_intent_from_state(graph_state)
        return {"result": self._handle_image_confirmation_phase(
            state,
            user_input,
            decision,
            user_intent,
        )}

    def _workflow_general_decision(self, graph_state: ImageWorkflowState) -> dict:
        """处理尚未进入确认阶段的普通生图决策。"""
        state = graph_state.get("state", {})
        user_input = graph_state.get("user_input", "")
        decision = self._decision_from_state(graph_state)
        return {"result": self._result_from_decision(state, user_input, decision)}

    @staticmethod
    def _decision_from_state(graph_state: ImageWorkflowState) -> ImageDecision:
        decision = graph_state["decision"]
        if isinstance(decision, ImageDecision):
            return decision
        return ImageDecision.model_validate(decision)

    @staticmethod
    def _user_intent_from_state(graph_state: ImageWorkflowState) -> ImageUserIntent:
        user_intent = graph_state["user_intent"]
        if isinstance(user_intent, ImageUserIntent):
            return user_intent
        return ImageUserIntent.model_validate(user_intent)

    def _result_from_decision(
        self,
        state: dict,
        user_input: str,
        decision: ImageDecision,
    ) -> dict:
        """把 LLM 候选决策转换成带阶段更新的标准结果。"""
        if decision.action == "expand":
            return self._handle_expand(decision, user_input, state)

        if decision.action == "generate":
            prompt = decision.prompt or user_input
            return self._ask_prompt_confirm(
                prompt=prompt,
                response=(
                    f"我理解的图片描述是：\n{prompt}\n\n"
                    f"需要我帮您扩写得更细一点吗？如果这个描述可以，请回复“可以”，我再开始生图~"
                ),
                state=state,
                decision=decision,
            )

        if decision.action in ("ask_prompt", "recommend", "ask_confirm"):
            phase = "collecting_prompt" if decision.action == "ask_prompt" else "awaiting_prompt_confirm"
            prompt = decision.prompt or state.get("image_prompt")
            return _make_result(
                action=decision.action,
                response=decision.response,
                prompt=prompt,
                provider="doubao",
                state_update={
                    "image_phase": phase,
                    "image_prompt": prompt or "",
                    "confirmed_prompt": prompt or state.get("confirmed_prompt", ""),
                    "size": decision.size or state.get("size", "2K"),
                    "output_format": decision.output_format or state.get("output_format", "jpeg"),
                    "is_batch": decision.is_batch if decision.is_batch is not None else state.get("is_batch", False),
                },
            )

        if decision.action == "show_image":
            return _make_result(
                action="show_image",
                response=decision.response,
                prompt=decision.prompt,
                provider="doubao",
            )

        if decision.action == "finish":
            return _make_result(
                action="finish",
                response=decision.response,
                state_update={"image_phase": "finished"},
            )

        return _make_result(
            action="ask_prompt",
            response=decision.response or "请告诉我您想要生成什么样的图片？",
            state_update={"image_phase": "collecting_prompt"},
        )

    def _handle_prompt_confirmation_phase(
        self,
        state: dict,
        user_input: str,
        decision: ImageDecision,
        user_intent: ImageUserIntent,
    ) -> dict:
        """处理“等待用户确认提示词”阶段。"""
        if user_intent.intent == "expand_prompt" or decision.action == "expand":
            prompt = state.get("image_prompt") or decision.prompt or user_input
            return self._handle_expand(
                decision.model_copy(update={"prompt": prompt}),
                user_input,
                state,
            )

        replacement_prompt = user_intent.replacement_prompt
        if replacement_prompt:
            return self._ask_prompt_confirm(
                prompt=replacement_prompt,
                response=(
                    f"我会改用新的描述：\n{replacement_prompt}\n\n"
                    f"这个描述可以吗？如果可以，请回复“可以”，我再开始生图~"
                ),
                state=state,
                decision=decision,
            )

        if user_intent.intent == "confirm" or decision.action == "generate":
            image_task_type = state.get("image_task_type", "generate_image")
            if image_task_type == "generate_image":
                prompt = state.get("image_prompt") or decision.prompt or user_input
                return self._make_generate_result(prompt, state, decision)

            return _make_result(
                action="finish",
                response="好的，这次提示词处理完成啦~",
                prompt=state.get("image_prompt"),
                provider="doubao",
                state_update={"image_phase": "finished"},
            )

        if user_intent.intent == "reject":
            return self._make_clarification_result(
                state,
                "需要修改哪一部分呢？您可以直接告诉我新的描述，或者说需要扩写、重新推荐~",
            )

        return self._make_clarification_result(state, self._default_clarification_question(state))

    def _handle_image_confirmation_phase(
        self,
        state: dict,
        user_input: str,
        decision: ImageDecision,
        user_intent: ImageUserIntent,
    ) -> dict:
        """处理“等待用户确认图片”阶段。"""
        if user_intent.intent == "confirm" or decision.action == "finish":
            return _make_result(
                action="finish",
                response=decision.response or "太好啦，那这次生图任务就完成啦~",
                prompt=state.get("image_prompt"),
                provider="doubao",
                state_update={"image_phase": "finished"},
            )

        if user_intent.intent == "reject":
            return self._make_clarification_result(
                state,
                "没关系~ 可以告诉我哪里需要调整吗？您可以要求重新生成、扩写当前提示词，或直接给我新的提示词。",
            )

        if user_intent.intent == "expand_prompt" or decision.action == "expand":
            prompt = state.get("image_prompt") or state.get("last_prompt") or decision.prompt or user_input
            return self._handle_expand(
                decision.model_copy(update={"prompt": prompt}),
                user_input,
                state,
            )

        replacement_prompt = user_intent.replacement_prompt
        if replacement_prompt:
            return self._ask_new_prompt_confirm(replacement_prompt, state, decision)

        if user_intent.intent in ("replace_prompt", "provide_prompt") and decision.prompt:
            return self._ask_new_prompt_confirm(decision.prompt, state, decision)

        if user_intent.intent == "regenerate":
            prompt = state.get("last_prompt") or state.get("image_prompt") or decision.prompt or user_input
            return self._make_generate_result(prompt, state, decision)

        if decision.action == "generate" and decision.prompt:
            return self._ask_new_prompt_confirm(decision.prompt, state, decision)

        return self._make_clarification_result(state, self._default_clarification_question(state))

    # ============== LLM决策 ==============

    def _decide(self, state: dict, user_input: str) -> ImageDecision:
        """LLM决策：分析对话历史，决定下一步

        构建包含对话历史和当前状态的prompt，调用LLM结构化输出。
        """
        # 构建对话历史（取最近的消息，避免超出上下文）
        messages = state.get("messages", [])
        history_text = self._format_history(messages[-6:])  # 最近6条
        current_prompt = state.get("image_prompt", "")
        image_phase = state.get("image_phase", "collecting_prompt")
        confirmed_prompt = state.get("confirmed_prompt", "")
        last_prompt = state.get("last_prompt", "")
        image_task_type = state.get("image_task_type", "generate_image")
        available_tools = ", ".join(self.providers.keys()) or "doubao"

        user_prompt = f"""请分析当前对话状态，输出下一步决策。

当前状态：
- 用户最新输入：{user_input}
- 当前生图阶段：{image_phase}
- 当前已确认的提示词：{current_prompt or "（暂无）"}
- 最近一次确认用于生图的提示词：{confirmed_prompt or last_prompt or "（暂无）"}
- 图片子任务类型：{image_task_type}（generate_image=生成图片, expand_prompt=扩写提示词, convert_tags=转成Danbooru标签）
- 可用生图工具：{available_tools}

最近对话历史：
{history_text}

请严格遵循system prompt中的工作流程，特别注意图片子任务类型决定确认后的最终行为，输出JSON格式决策。"""

        # 调用LLM
        try:
            structured_llm = self.llm.with_structured_output(
                ImageDecision,
                method="json_mode",
            )
            result = structured_llm.invoke([
                IMAGE_CONVERSATION_SYSTEM_MESSAGE,
                HumanMessage(content=user_prompt),
            ])
            return result
        except Exception:
            # 降级处理：任何决策错误都转为追问，避免中断对话。
            return ImageDecision(
                thought="LLM决策失败，降级到追问",
                action="ask_prompt",
                response="抱歉，我刚才没理解清楚~ 请告诉我您想要生成什么样的图片？",
                prompt=None,
                provider=None,
            )

    def _format_history(self, messages: list) -> str:
        """格式化对话历史为文本"""
        lines = []
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                role = "用户" if getattr(msg, "type", "") == "human" else "AI"
                lines.append(f"{role}：{msg.content}")
        return "\n".join(lines) if lines else "（无历史对话）"

    # ============== Action处理 ==============

    def _handle_expand(
        self,
        decision: ImageDecision,
        user_input: str,
        state: dict | None = None,
    ) -> dict:
        """处理扩写：调用expand_prompt，返回确认状态"""
        state = state or {}
        prompt_to_expand = decision.prompt or user_input
        expanded = expand_prompt(prompt_to_expand, self.llm)

        return _make_result(
            action="ask_confirm",
            response=(
                f"扩写后的描述：\n"
                f"{expanded}\n\n"
                f"这个描述可以吗？需要修改哪里吗？"
            ),
            prompt=expanded,
            expanded_prompt=expanded,
            state_update={
                "image_phase": "awaiting_prompt_confirm",
                "image_prompt": expanded,
                "confirmed_prompt": expanded,
                "size": decision.size or state.get("size", "2K"),
                "output_format": decision.output_format or state.get("output_format", "jpeg"),
                "is_batch": decision.is_batch if decision.is_batch is not None else state.get("is_batch", False),
            },
        )

    def _make_generate_result(
        self,
        prompt: str,
        state: dict,
        decision: ImageDecision,
    ) -> dict:
        """构造生图动作结果，并标记为等待生图完成。"""
        size = decision.size or state.get("size", "2K")
        output_format = decision.output_format or state.get("output_format", "jpeg")
        is_batch = decision.is_batch if decision.is_batch is not None else state.get("is_batch", False)

        return _make_result(
            action="generate",
            response=decision.response or "图片正在生成中，请稍等~",
            prompt=prompt,
            provider="doubao",
            size=size,
            output_format=output_format,
            is_batch=is_batch,
            state_update={
                "image_phase": "awaiting_generation",
                "image_prompt": prompt,
                "confirmed_prompt": prompt,
                "last_prompt": prompt,
                "size": size,
                "output_format": output_format,
                "is_batch": is_batch,
            },
        )

    def _ask_prompt_confirm(
        self,
        prompt: str,
        response: str,
        state: dict,
        decision: ImageDecision,
    ) -> dict:
        """要求用户确认提示词，确认后才允许生图。"""
        size = decision.size or state.get("size", "2K")
        output_format = decision.output_format or state.get("output_format", "jpeg")
        is_batch = decision.is_batch if decision.is_batch is not None else state.get("is_batch", False)
        return _make_result(
            action="ask_confirm",
            response=response,
            prompt=prompt,
            provider="doubao",
            state_update={
                "image_phase": "awaiting_prompt_confirm",
                "image_prompt": prompt,
                "confirmed_prompt": prompt,
                "size": size,
                "output_format": output_format,
                "is_batch": is_batch,
            },
        )

    def _ask_new_prompt_confirm(
        self,
        prompt: str,
        state: dict,
        decision: ImageDecision,
    ) -> dict:
        """图片确认阶段收到新描述时，先进入提示词确认闭环。"""
        return self._ask_prompt_confirm(
            prompt=prompt,
            response=(
                f"看起来是个很可爱的场景呢！{prompt}\n\n"
                f"请问这个描述可以吗？或者想要我帮你丰富一下细节呢？"
            ),
            state=state,
            decision=decision,
        )

    def _classify_user_intent(
        self,
        state: dict,
        user_input: str,
        decision: ImageDecision,
    ) -> ImageUserIntent:
        """用 LLM 语义识别用户意图；不使用关键词规则。"""
        phase = state.get("image_phase", "collecting_prompt")
        current_prompt = state.get("image_prompt") or "（暂无）"
        last_prompt = state.get("last_prompt") or "（暂无）"
        last_images = state.get("last_image_urls") or []

        intent_prompt = f"""请基于语义判断用户在生图流程中的真实意图，不要做关键词匹配。

当前阶段：{phase}
当前提示词：{current_prompt}
最近一次生图提示词：{last_prompt}
最近生成图片：{last_images or "（暂无）"}
LLM候选动作：{decision.action}
LLM候选提示词：{decision.prompt or "（暂无）"}
用户最新输入：{user_input}

可选意图：
- confirm：确认当前提示词或图片
- reject：否定当前结果，但没有给出明确新动作
- end：明确结束当前生图任务
- regenerate：保持当前提示词重新生成
- expand_prompt：扩写当前提示词
- replace_prompt：用户给出了新提示词，要替换当前提示词
- provide_prompt：用户提供了可用于生图的描述
- request_recommendation：用户希望系统推荐主题或描述
- clarify：用户意图不明确，需要追问澄清
- other：与当前确认流程关系不明确

如果用户提供了新提示词，请把干净的图片描述放入 replacement_prompt。
如果无法确定用户想确认、修改、重生、结束还是提供新提示词，请选择 clarify，并给出一个简短中文澄清问题。"""

        try:
            structured_llm = self.llm.with_structured_output(
                ImageUserIntent,
                method="json_mode",
            )
            return structured_llm.invoke([HumanMessage(content=intent_prompt)])
        except Exception:
            return ImageUserIntent(
                thought="用户意图识别失败，进入澄清流程",
                intent="clarify",
                confidence=0.0,
                clarification_question=self._default_clarification_question(state),
            )

    def _make_clarification_result(self, state: dict, question: str) -> dict:
        """构造澄清结果，保持当前阶段不变。"""
        return _make_result(
            action="ask_confirm",
            response=question,
            prompt=state.get("image_prompt") or state.get("last_prompt"),
            provider="doubao",
            state_update={"image_phase": state.get("image_phase", "collecting_prompt")},
        )

    @staticmethod
    def _default_clarification_question(state: dict) -> str:
        """根据当前阶段生成默认澄清问题。"""
        phase = state.get("image_phase", "collecting_prompt")
        if phase == "awaiting_prompt_confirm":
            return "您是想确认这个描述开始生图，还是想继续修改或扩写提示词呢？"
        if phase == "awaiting_image_confirm":
            return "您是满意这张图想结束任务，还是想重新生成、扩写提示词，或换一组新的提示词呢？"
        return "我还没完全理解您的生图需求，可以再具体描述一下您想生成的画面吗？"

    # ============== 生图执行 ==============

    async def generate_image(
        self,
        prompt: str,
        provider_name: str | None = None,
        size: str = "2K",
        output_format: str = "jpeg",
    ) -> GenerationResult:
        """调用Provider生成图片

        Args:
            prompt: 提示词（自然语言）
            provider_name: Provider名称，默认doubao
            size: 图片尺寸，默认 2K
            output_format: 输出格式，默认 jpeg

        Returns:
            GenerationResult: 生成结果
        """
        provider_name = provider_name or self.default_provider

        if provider_name not in self.providers:
            available = list(self.providers.keys())
            raise ValueError(
                f"未知生图提供者：{provider_name}。"
                f"可用提供者：{available}"
            )

        provider = self.providers[provider_name]
        return await provider.generate(
            prompt=prompt,
            size=size,
            output_format=output_format,
        )

    # ============== 组图生成 ==============

    async def generate_images(
        self,
        prompt: str,
        provider_name: str | None = None,
        size: str = "2K",
        output_format: str = "jpeg",
    ) -> list[GenerationResult]:
        """调用Provider生成组图（非流式）

        Args:
            prompt: 提示词（自然语言）
            provider_name: Provider名称，默认doubao
            size: 图片尺寸，默认 2K
            output_format: 输出格式，默认 jpeg

        Returns:
            list[GenerationResult]: 生成结果列表
        """
        provider_name = provider_name or self.default_provider

        if provider_name not in self.providers:
            available = list(self.providers.keys())
            raise ValueError(
                f"未知生图提供者：{provider_name}。"
                f"可用提供者：{available}"
            )

        provider = self.providers[provider_name]
        generate_many = getattr(provider, "generate_images", None)
        if generate_many is not None:
            return await generate_many(
                prompt=prompt,
                size=size,
                output_format=output_format,
            )

        # 非组图 provider 仍可降级生成单张，保证接口返回类型稳定。
        generation = await provider.generate(
            prompt=prompt,
            size=size,
            output_format=output_format,
        )
        return [generation]

    # ============== 便捷方法 ==============

    def get_available_providers(self) -> list[str]:
        """获取可用Provider列表"""
        return list(self.providers.keys())


# ============== 工厂函数 ==============

def create_image_agent(
    llm: BaseChatModel | None = None,
    providers: list[ImageProvider] | None = None,
) -> ImageSubAgent:
    """创建生图Agent工厂函数

    Args:
        llm: 大语言模型，默认使用DeepSeek
        providers: Provider列表，默认包含Doubao

    Returns:
        ImageSubAgent: 生图子Agent实例
    """
    if llm is None:
        try:
            from huesae.models.models_factory import create_chat_model
        except ImportError:
            from huesaeagents.huesae.models.models_factory import create_chat_model
        llm = create_chat_model("deepseek")

    if providers is None:
        from .image.providers import DoubaoProvider
        providers = [DoubaoProvider()]

    return ImageSubAgent(llm=llm, providers=providers)
