"""生图对话管理器

用LLM驱动对话流程，替代硬编码状态机。
支持：追问、推荐、扩写、确认、生图、换图

核心设计：
- 每个回合，LLM分析完整对话历史，输出结构化决策（action/response/prompt）
- 根据action执行对应操作（追问、调用扩写、准备生图等）
- 实际生图由Graph节点调用异步方法完成
"""
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from .image import expand_prompt
from .image.providers import ImageProvider, GenerationResult
from .image.prompts import IMAGE_CONVERSATION_SYSTEM_MESSAGE


# ============== LLM决策模型 ==============

class ImageDecision(BaseModel):
    """子图Agent每回合的决策"""

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
    provider: str | None = Field(default=None, description="选择的生图工具（doubao/jimeng）")


# ============== 对话管理器 ==============

class ImageConversationManager:
    """生图对话管理器

    用LLM驱动对话流程，实现"思考-澄清-行动"闭环。

    Example:
        >>> from models.models_factory import create_chat_model
        >>> from agents.subagents.image.providers import DoubaoProvider, JimengProvider
        >>> manager = ImageConversationManager(
        ...     llm=create_chat_model("deepseek"),
        ...     providers=[DoubaoProvider(), JimengProvider()]
        ... )
        >>> result = manager.process({}, "我想生成图片")
        >>> print(result["messages"][0].content)
        '请告诉我您想要生成什么样的图片？...'
    """

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

    # ============== 主入口 ==============

    def process(self, state: dict, user_input: str) -> dict:
        """处理用户输入

        1. LLM决策（分析对话历史 → 决定action）
        2. 根据action执行对应操作

        Args:
            state: 当前Graph状态（包含messages等）
            user_input: 用户最新输入

        Returns:
            dict: 更新后的状态，包含messages、image_step等
        """
        # 1. LLM决策
        decision = self._decide(state, user_input)

        # 2. 根据action执行
        if decision.action == "expand":
            return self._handle_expand(decision, user_input)
        elif decision.action == "generate":
            return self._handle_generate(decision)
        elif decision.action in ("ask_prompt", "recommend", "ask_confirm"):
            return {
                "image_step": decision.action,
                "image_prompt": decision.prompt,
                "messages": [AIMessage(content=decision.response)],
            }
        elif decision.action == "show_image":
            return {
                "image_step": "show_image",
                "image_prompt": decision.prompt,
                "messages": [AIMessage(content=decision.response)],
            }
        elif decision.action == "finish":
            return {
                "image_step": "finish",
                "messages": [AIMessage(content=decision.response)],
            }

        # 默认：追问
        return {
            "image_step": "ask_prompt",
            "messages": [AIMessage(content=decision.response)],
        }

    # ============== LLM决策 ==============

    def _decide(self, state: dict, user_input: str) -> ImageDecision:
        """LLM决策：分析对话历史，决定下一步

        构建包含对话历史和当前状态的prompt，调用LLM结构化输出。
        """
        # 构建对话历史（取最近的消息，避免超出上下文）
        messages = state.get("messages", [])
        history_text = self._format_history(messages[-8:])  # 最近8条
        current_prompt = state.get("image_prompt", "")
        available_tools = ", ".join(self.providers.keys()) or "doubao, jimeng"

        user_prompt = f"""请分析当前对话状态，输出下一步决策。

当前状态：
- 用户最新输入：{user_input}
- 当前已确认的提示词：{current_prompt or "（暂无）"}
- 可用生图工具：{available_tools}

最近对话历史：
{history_text}

请严格遵循system prompt中的工作流程，输出JSON格式决策。"""

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
            # Fallback：任何错误都转为追问
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

    def _handle_expand(self, decision: ImageDecision, user_input: str) -> dict:
        """处理扩写：调用expand_prompt，返回确认状态"""
        prompt_to_expand = decision.prompt or user_input
        expanded = expand_prompt(prompt_to_expand, self.llm)

        return {
            "image_step": "ask_confirm",
            "image_prompt": prompt_to_expand,
            "expanded_prompt": expanded,
            "messages": [
                AIMessage(
                    content=(
                        f"扩写后的描述：\n"
                        f"{expanded}\n\n"
                        f"这个描述可以吗？需要修改哪里吗？"
                    )
                )
            ],
        }

    def _handle_generate(self, decision: ImageDecision) -> dict:
        """处理生图决策

        不实际调用provider（异步操作由Graph节点执行），
        只返回generate状态和相关参数。
        """
        return {
            "image_step": "generate",
            "image_prompt": decision.prompt,
            "selected_provider": decision.provider or self.default_provider,
            "messages": [
                AIMessage(
                    content=decision.response or "图片正在生成中，请稍等~"
                )
            ],
        }

    # ============== 生图执行 ==============

    async def generate_image(
        self,
        prompt: str,
        provider_name: str | None = None,
    ) -> GenerationResult:
        """调用Provider生成图片

        Args:
            prompt: 提示词（自然语言）
            provider_name: Provider名称，默认doubao

        Returns:
            GenerationResult: 生成结果
        """
        provider_name = provider_name or self.default_provider

        if provider_name not in self.providers:
            available = list(self.providers.keys())
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {available}"
            )

        provider = self.providers[provider_name]
        return await provider.generate(prompt=prompt)

    # ============== 便捷方法 ==============

    def get_available_providers(self) -> list[str]:
        """获取可用Provider列表"""
        return list(self.providers.keys())


# ============== 工厂函数 ==============

def create_image_agent(
    llm: BaseChatModel | None = None,
    providers: list[ImageProvider] | None = None,
) -> ImageConversationManager:
    """创建生图Agent工厂函数

    Args:
        llm: 大语言模型，默认使用DeepSeek
        providers: Provider列表，默认包含Doubao和Jimeng

    Returns:
        ImageConversationManager: 生图对话管理器实例
    """
    if llm is None:
        try:
            from huesae.models.models_factory import create_chat_model
        except ImportError:
            from huesaeagents.huesae.models.models_factory import create_chat_model
        llm = create_chat_model("deepseek")

    if providers is None:
        from .image.providers import DoubaoProvider, JimengProvider
        providers = [DoubaoProvider(), JimengProvider()]

    return ImageConversationManager(llm=llm, providers=providers)
