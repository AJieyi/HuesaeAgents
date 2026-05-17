"""生图子Agent

主Agent的委派组件，负责处理生图相关的多轮对话。
会把阶段信息写入主Agent维护的子Agent状态中，确保确认闭环稳定。

支持流程：追问 → 推荐 → 扩写 → 确认描述 → 生图 → 确认图片 → 结束
"""
from typing import Literal, TypedDict

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain.messages import HumanMessage
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
    is_batch: bool | None = Field(default=False, description="是否使用组图模式，用户明确说明生成数量（如生成4张）时为true")


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
    decision: ImageDecision
    result: dict


ImageWorkflowRoute = Literal[
    "finish_task",
    "prompt_confirmation",
    "image_confirmation",
    "general_decision",
]


_CONFIRM_KEYWORDS = [
    "可以", "没问题", "就这个", "确认", "行", "好的", "好呀", "生成吧", "开始生成",
]
_NEGATIVE_CONFIRM_KEYWORDS = [
    "不可以", "不行", "不要", "不满意", "不对", "不太行",
]
_END_KEYWORDS = [
    "不用了", "结束吧", "不画了", "先这样", "谢谢不用", "取消",
]
_REGENERATE_KEYWORDS = [
    "换一张", "换一组", "重新生成", "再来一张", "再来一组", "重画",
]
_EXPAND_KEYWORDS = [
    "扩写", "扩展", "写详细", "丰富一下",
]
_REPLACE_PROMPT_KEYWORDS = [
    "重新输入", "新提示词", "换个提示词", "换一个主题", "修改描述",
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
        self.workflow = self._build_workflow()

    def register_provider(self, provider: ImageProvider) -> None:
        """注册生图Provider（扩展点）"""
        self.providers[provider.name] = provider

    # ============== 主入口：标准化接口 ==============

    def process(self, state: dict, user_input: str) -> dict:
        """处理用户输入，返回标准化结果

        Args:
            state: 当前状态（包含 messages 对话历史）
            user_input: 用户最新输入

        Returns:
            dict: 标准化结果 {action, response, prompt, provider, data}
        """
        workflow_state = self.workflow.invoke({
            "state": state,
            "user_input": user_input,
        })
        return workflow_state["result"]

    # ============== LangGraph 工作流 ==============

    def _build_workflow(self):
        """构建生图确认流程图。

        LLM 负责理解用户意图，LangGraph 节点负责把确认状态约束住：
        生成任务确认提示词后必须生图，确认图片后才结束。
        """
        workflow = StateGraph(ImageWorkflowState)
        workflow.add_node("llm_decide", self._workflow_decide)
        workflow.add_node("finish_task", self._workflow_finish_task)
        workflow.add_node("prompt_confirmation", self._workflow_prompt_confirmation)
        workflow.add_node("image_confirmation", self._workflow_image_confirmation)
        workflow.add_node("general_decision", self._workflow_general_decision)
        workflow.add_edge(START, "llm_decide")
        workflow.add_conditional_edges(
            "llm_decide",
            self._route_after_decision,
            {
                "finish_task": "finish_task",
                "prompt_confirmation": "prompt_confirmation",
                "image_confirmation": "image_confirmation",
                "general_decision": "general_decision",
            },
        )
        workflow.add_edge("finish_task", END)
        workflow.add_edge("prompt_confirmation", END)
        workflow.add_edge("image_confirmation", END)
        workflow.add_edge("general_decision", END)
        return workflow.compile()

    def _workflow_decide(self, graph_state: ImageWorkflowState) -> dict:
        """调用 LLM 输出候选决策。"""
        state = graph_state.get("state", {})
        user_input = graph_state.get("user_input", "")
        return {"decision": self._decide(state, user_input)}

    def _route_after_decision(self, graph_state: ImageWorkflowState) -> ImageWorkflowRoute:
        """按当前阶段路由到对应处理节点。"""
        state = graph_state.get("state", {})
        user_input = graph_state.get("user_input", "")
        phase = state.get("image_phase", "collecting_prompt")

        if self._is_end_request(user_input):
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

    def _workflow_prompt_confirmation(self, graph_state: ImageWorkflowState) -> dict:
        """处理提示词确认阶段。"""
        state = graph_state.get("state", {})
        user_input = graph_state.get("user_input", "")
        decision = graph_state["decision"]
        return {"result": self._handle_prompt_confirmation_phase(state, user_input, decision)}

    def _workflow_image_confirmation(self, graph_state: ImageWorkflowState) -> dict:
        """处理图片确认阶段。"""
        state = graph_state.get("state", {})
        user_input = graph_state.get("user_input", "")
        decision = graph_state["decision"]
        return {"result": self._handle_image_confirmation_phase(state, user_input, decision)}

    def _workflow_general_decision(self, graph_state: ImageWorkflowState) -> dict:
        """处理尚未进入确认阶段的普通生图决策。"""
        state = graph_state.get("state", {})
        user_input = graph_state.get("user_input", "")
        decision = graph_state["decision"]
        return {"result": self._result_from_decision(state, user_input, decision)}

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
    ) -> dict:
        """处理“等待用户确认提示词”阶段。"""
        if self._is_expand_request(user_input):
            prompt = state.get("image_prompt") or decision.prompt or user_input
            return self._handle_expand(
                decision.model_copy(update={"prompt": prompt}),
                user_input,
                state,
            )

        replacement_prompt = self._extract_replacement_prompt(user_input)
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

        if self._is_confirm(user_input):
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

        return self._result_from_decision(state, user_input, decision)

    def _handle_image_confirmation_phase(
        self,
        state: dict,
        user_input: str,
        decision: ImageDecision,
    ) -> dict:
        """处理“等待用户确认图片”阶段。"""
        if self._is_confirm(user_input):
            return _make_result(
                action="finish",
                response="太好啦，那这次生图任务就完成啦~",
                prompt=state.get("image_prompt"),
                provider="doubao",
                state_update={"image_phase": "finished"},
            )

        if self._is_negative_confirm(user_input):
            return _make_result(
                action="ask_prompt",
                response="没关系~ 可以告诉我哪里需要调整吗？也可以说“换一张”或重新输入提示词。",
                prompt=state.get("image_prompt"),
                provider="doubao",
                state_update={"image_phase": "awaiting_image_confirm"},
            )

        if self._is_regenerate_request(user_input):
            prompt = state.get("last_prompt") or state.get("image_prompt") or decision.prompt or user_input
            return self._make_generate_result(prompt, state, decision)

        if self._is_expand_request(user_input):
            prompt = state.get("image_prompt") or state.get("last_prompt") or decision.prompt or user_input
            return self._handle_expand(
                decision.model_copy(update={"prompt": prompt}),
                user_input,
                state,
            )

        replacement_prompt = self._extract_replacement_prompt(user_input)
        if replacement_prompt:
            return self._make_generate_result(replacement_prompt, state, decision)

        if decision.action == "generate" and decision.prompt:
            return self._make_generate_result(decision.prompt, state, decision)

        if decision.action == "ask_prompt":
            return _make_result(
                action="ask_prompt",
                response=decision.response,
                prompt=state.get("image_prompt"),
                provider="doubao",
                state_update={"image_phase": "collecting_prompt"},
            )

        return _make_result(
            action="ask_prompt",
            response="需要调整哪里呢？可以说“换一张”、发新的提示词，或者回复“可以”结束这次任务~",
            prompt=state.get("image_prompt"),
            provider="doubao",
            state_update={"image_phase": "awaiting_image_confirm"},
        )

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

    # ============== 风格处理 ==============

    @staticmethod
    def _ensure_anime_style(prompt: str) -> str:
        """确保提示词包含动漫风格前缀

        默认添加"二次元动漫风格"前缀，除非用户明确要求真人/写实风格。
        """
        if not prompt:
            return prompt
        lower = prompt.lower()
        # 用户明确要求非动漫风格
        if any(kw in lower for kw in ["真人", "写实", "照片", "photorealistic", "realistic", "real person"]):
            return prompt
        # 已经包含动漫关键词
        return f"图片风格为 二次元，{prompt}"

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
        expanded = self._ensure_anime_style(expanded)

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
        prompt = self._ensure_anime_style(prompt)
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

    @staticmethod
    def _is_confirm(user_input: str) -> bool:
        """判断用户是否在确认当前阶段。"""
        text = user_input.strip().lower()
        if ImageSubAgent._is_negative_confirm(user_input):
            return False
        return any(keyword in text for keyword in _CONFIRM_KEYWORDS)

    @staticmethod
    def _is_negative_confirm(user_input: str) -> bool:
        """判断用户是否明确否定当前结果。"""
        text = user_input.strip().lower()
        return any(keyword in text for keyword in _NEGATIVE_CONFIRM_KEYWORDS)

    @staticmethod
    def _is_end_request(user_input: str) -> bool:
        """判断用户是否明确要求结束任务。"""
        text = user_input.strip().lower()
        return any(keyword in text for keyword in _END_KEYWORDS)

    @staticmethod
    def _is_regenerate_request(user_input: str) -> bool:
        """判断用户是否要求重新生成图片。"""
        text = user_input.strip().lower()
        return any(keyword in text for keyword in _REGENERATE_KEYWORDS)

    @staticmethod
    def _is_expand_request(user_input: str) -> bool:
        """判断用户是否要求扩写提示词。"""
        text = user_input.strip().lower()
        return any(keyword in text for keyword in _EXPAND_KEYWORDS)

    @staticmethod
    def _extract_replacement_prompt(user_input: str) -> str:
        """从“重新输入提示词：...”这类表达中提取新提示词。"""
        if not any(keyword in user_input for keyword in _REPLACE_PROMPT_KEYWORDS):
            return ""

        for separator in ("：", ":"):
            if separator in user_input:
                candidate = user_input.split(separator, 1)[1].strip()
                if candidate:
                    return candidate

        return ""

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
