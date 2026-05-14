"""生图子Agent

主Agent的委派组件，负责处理生图相关的多轮对话。
自身无状态，每次根据完整对话历史做决策。

支持流程：追问 → 推荐 → 扩写 → 确认 → 生图 → 展示 → 结束
"""
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain.messages import HumanMessage

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
        # 1. LLM决策
        decision = self._decide(state, user_input)

        # 2. 根据action执行对应操作，返回标准化格式
        if decision.action == "expand":
            return self._handle_expand(decision, user_input)
        elif decision.action == "generate":
            return self._handle_generate(decision)
        elif decision.action in ("ask_prompt", "recommend", "ask_confirm"):
            return _make_result(
                action=decision.action,
                response=decision.response,
                prompt=decision.prompt,
                provider="doubao",
            )
        elif decision.action == "show_image":
            return _make_result(
                action="show_image",
                response=decision.response,
                prompt=decision.prompt,
                provider="doubao",
            )
        elif decision.action == "finish":
            return _make_result(
                action="finish",
                response=decision.response,
            )

        # 默认：追问
        return _make_result(
            action="ask_prompt",
            response=decision.response or "请告诉我您想要生成什么样的图片？",
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
        image_task_type = state.get("image_task_type", "generate_image")
        available_tools = ", ".join(self.providers.keys()) or "doubao"

        user_prompt = f"""请分析当前对话状态，输出下一步决策。

当前状态：
- 用户最新输入：{user_input}
- 当前已确认的提示词：{current_prompt or "（暂无）"}
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

    def _handle_expand(self, decision: ImageDecision, user_input: str) -> dict:
        """处理扩写：调用expand_prompt，返回确认状态"""
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
            prompt=prompt_to_expand,
            expanded_prompt=expanded,
        )

    def _handle_generate(self, decision: ImageDecision) -> dict:
        """处理生图决策

        不实际调用provider（异步操作由主Agent执行），
        只返回generate状态和相关参数。
        """
        prompt = self._ensure_anime_style(decision.prompt or "")
        return _make_result(
            action="generate",
            response=decision.response or "图片正在生成中，请稍等~",
            prompt=prompt,
            provider="doubao",
            size=decision.size or "2K",
            output_format=decision.output_format or "jpeg",
            is_batch=decision.is_batch or False,
        )

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
