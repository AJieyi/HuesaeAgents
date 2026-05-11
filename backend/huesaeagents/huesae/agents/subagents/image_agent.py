"""小生图 Agent（独立类）

支持三种模式：
1. 直接生图：自然语言 → 选择工具(豆包/即梦) → 生图 → 换图/结束
2. 转Danbooru标签：自然语言 → Danbooru标签 → 换版本/结束
3. 扩写提示词：简短描述 → 扩写自然语言 → 接受/拒绝

使用 LLM 结构化输出做意图识别和提示词提取。

后续可封装为 LangGraph 节点：
    def image_agent_node(state: HuesaeState) -> dict:
        agent = ImageAgent(llm=..., providers=[...])
        return agent.process_input(user_input)
"""
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from .image import generate_tags, tags_to_prompt, expand_prompt
from .image.intent import recognize_intent, ImageIntent
from .image.providers import ImageProvider, GenerationResult


# ============== 模式常量 ==============

class ImageMode:
    """生图功能模式"""

    DIRECT_IMAGE = "direct_image"
    CONVERT_TAGS = "convert_tags"
    EXPAND_PROMPT = "expand_prompt"
    CHAT = "chat"


# ============== 步骤常量 ==============

class ImageStep:
    """生图步骤常量"""

    INPUT = "input"
    SELECT_TOOL = "select_tool"
    GENERATE_IMAGE = "generate_image"
    SHOW_IMAGE = "show_image"
    GENERATE_TAGS = "generate_tags"
    SHOW_TAGS = "show_tags"
    EXPAND_PROMPT = "expand_prompt"
    SHOW_EXPANDED = "show_expanded"
    FINISH = "finish"


# ============== 状态 ==============

class ImageAgentState(TypedDict):
    """生图Agent内部状态"""

    mode: str
    step: str
    user_input: str
    prompt: str
    selected_provider: str | None
    generated_image_url: str | None
    messages: list


# ============== 确认关键词 ==============

CONFIRM_KEYWORDS = ["是", "确认", "确定", "ok", "yes", "可以", "好", "行", "生成", "接受", "就要这版", "可以了"]
REJECT_KEYWORDS = ["换", "重新", "再来", "不要", "reject", "no", "换一张", "换一版", "再写一版", "换一个", "换一版本"]


class ImageAgent:
    """生图Agent

    使用LLM结构化输出做意图识别和提示词提取。

    Example:
        >>> from models.factory import create_chat_model
        >>> from agents.subagents.image.providers import DoubaoProvider, JimengProvider
        >>> agent = ImageAgent(
        ...     llm=create_chat_model("deepseek"),
        ...     providers=[DoubaoProvider(), JimengProvider()]
        ... )
        >>> result = agent.process_input("画一个银发红瞳的少女")
        >>> print(result["message"])
        '检测到您想要生成图片，请输入更详细的提示词（至少5个字）...'
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

    # ============== 意图识别 ==============

    def _recognize(self, user_input: str) -> ImageIntent:
        """使用LLM识别用户意图"""
        try:
            return recognize_intent(user_input, self.llm)
        except Exception:
            # 如果结构化输出失败，使用fallback
            from .image.intent import recognize_intent_simple
            result = recognize_intent_simple(user_input, self.llm)
            return ImageIntent(**result)

    # ============== 模式A：直接生图 ==============

    def _handle_direct_image_input(self, prompt: str, intent_result: ImageIntent) -> dict:
        """处理直接生图模式 - 初始输入"""
        # 检查是否需要澄清
        if intent_result.needs_clarification or len(prompt) < 5:
            return {
                "step": ImageStep.INPUT,
                "message": (
                    f"提示词太短啦，请告诉我更详细的描述（至少5个字）~\n"
                    f"比如：'一个银发红瞳的少女在樱花树下'"
                ),
                "need_more_input": True,
            }

        # 询问选择工具
        available = list(self.providers.keys())
        tool_list = " / ".join(available)

        return {
            "step": ImageStep.SELECT_TOOL,
            "mode": ImageMode.DIRECT_IMAGE,
            "prompt": prompt,
            "message": (
                f"想要生成图片：{prompt}\n\n"
                f"请选择生图工具：{tool_list}\n"
                f"（输入工具名即可）"
            ),
        }

    def _handle_select_tool(self, user_input: str, current_prompt: str) -> dict:
        """处理工具选择"""
        content = user_input.lower().strip()

        # 匹配Provider
        selected = None
        for name in self.providers.keys():
            if name.lower() in content:
                selected = name
                break

        # 默认使用doubao
        if selected is None:
            selected = self.default_provider

        return {
            "step": ImageStep.GENERATE_IMAGE,
            "mode": ImageMode.DIRECT_IMAGE,
            "prompt": current_prompt,
            "selected_provider": selected,
            "message": f"好的，使用 {selected} 生成图片：{current_prompt}",
        }

    def _handle_show_image(self, user_input: str, current_prompt: str) -> dict:
        """处理图片展示后的用户反馈"""
        content = user_input.lower().strip()

        # 用户说"换一张" → 用豆包重新生成
        if any(kw in content for kw in ["换", "重新", "再来", "换一个"]):
            return {
                "step": ImageStep.GENERATE_IMAGE,
                "mode": ImageMode.DIRECT_IMAGE,
                "prompt": current_prompt,
                "selected_provider": "doubao",
                "message": f"好的，用豆包重新生成：{current_prompt}",
            }

        # 用户说"可以" → 结束
        return {
            "step": ImageStep.FINISH,
            "message": "图片生成完成！如果还想画别的，随时告诉我~",
        }

    # ============== 模式B：转Danbooru标签 ==============

    def _handle_convert_tags_input(self, prompt: str) -> dict:
        """处理转Danbooru标签模式 - 初始输入"""
        if len(prompt) < 2:
            return {
                "step": ImageStep.INPUT,
                "message": "请告诉我你想要转换的内容~",
                "need_more_input": True,
            }

        # 生成Danbooru标签
        tags = generate_tags(prompt, self.llm)
        tags_str = ", ".join(tags)

        return {
            "step": ImageStep.SHOW_TAGS,
            "mode": ImageMode.CONVERT_TAGS,
            "prompt": prompt,
            "danbooru_tags": tags,
            "message": (
                f"为你生成的Danbooru标签：\n"
                f"{tags_str}\n\n"
                f"是否满意？（可以了 / 换一版）"
            ),
        }

    def _handle_show_tags(self, user_input: str, current_prompt: str) -> dict:
        """处理标签展示后的用户反馈"""
        content = user_input.lower().strip()

        # 用户说"换一版" → 重新生成标签
        if any(kw in content for kw in ["换", "重新", "再来", "换一个"]):
            tags = generate_tags(current_prompt, self.llm)
            tags_str = ", ".join(tags)

            return {
                "step": ImageStep.SHOW_TAGS,
                "mode": ImageMode.CONVERT_TAGS,
                "prompt": current_prompt,
                "danbooru_tags": tags,
                "message": (
                    f"重新生成的Danbooru标签：\n"
                    f"{tags_str}\n\n"
                    f"是否满意？（可以了 / 换一版）"
                ),
            }

        # 用户说"可以了" → 结束
        return {
            "step": ImageStep.FINISH,
            "message": "标签生成完成！需要生图的话告诉我~",
        }

    # ============== 模式C：扩写提示词 ==============

    def _handle_expand_prompt_input(self, prompt: str) -> dict:
        """处理扩写提示词模式 - 初始输入"""
        if len(prompt) < 2:
            return {
                "step": ImageStep.INPUT,
                "message": "请告诉我你想要扩写的内容~",
                "need_more_input": True,
            }

        # 扩写提示词
        expanded = expand_prompt(prompt, self.llm)

        return {
            "step": ImageStep.SHOW_EXPANDED,
            "mode": ImageMode.EXPAND_PROMPT,
            "prompt": prompt,
            "expanded_prompt": expanded,
            "message": (
                f"扩写后的描述：\n"
                f"{expanded}\n\n"
                f"是否接受这版？（接受 / 再写一版）"
            ),
        }

    def _handle_show_expanded(self, user_input: str, current_prompt: str) -> dict:
        """处理扩写展示后的用户反馈"""
        content = user_input.lower().strip()

        # 用户说"再写一版" → 重新扩写
        if any(kw in content for kw in ["换", "重新", "再来", "再写"]):
            expanded = expand_prompt(current_prompt, self.llm)

            return {
                "step": ImageStep.SHOW_EXPANDED,
                "mode": ImageMode.EXPAND_PROMPT,
                "prompt": current_prompt,
                "expanded_prompt": expanded,
                "message": (
                    f"重新扩写的描述：\n"
                    f"{expanded}\n\n"
                    f"是否接受这版？（接受 / 再写一版）"
                ),
            }

        # 用户说"接受" → 结束
        return {
            "step": ImageStep.FINISH,
            "message": "扩写完成！需要生图或转Danbooru标签的话告诉我~",
        }

    # ============== 主入口 ==============

    def process_input(self, user_input: str) -> dict:
        """处理用户初始输入

        使用LLM结构化输出识别意图，进入对应的处理流程。

        Args:
            user_input: 用户输入

        Returns:
            dict: 包含下一步状态、模式、消息
        """
        # 使用LLM识别意图
        intent_result = self._recognize(user_input)

        # 根据意图分发
        if intent_result.intent == ImageMode.DIRECT_IMAGE:
            result = self._handle_direct_image_input(intent_result.extracted_prompt, intent_result)
        elif intent_result.intent == ImageMode.CONVERT_TAGS:
            result = self._handle_convert_tags_input(intent_result.extracted_prompt)
        elif intent_result.intent == ImageMode.EXPAND_PROMPT:
            result = self._handle_expand_prompt_input(intent_result.extracted_prompt)
        else:
            # 普通对话，不处理
            return {
                "mode": ImageMode.CHAT,
                "step": ImageStep.FINISH,
                "message": "我是生图Agent，可以帮你生成图片、转Danbooru标签或扩写提示词~",
            }

        # 注入公共字段
        result["mode"] = result.get("mode", intent_result.intent)
        result["user_input"] = user_input
        if "need_more_input" not in result:
            result["need_more_input"] = False

        return result

    def process_step(self, state: ImageAgentState, user_input: str) -> dict:
        """处理步骤流转

        根据当前步骤和模式，处理用户的下一步输入。

        Args:
            state: 当前Agent状态
            user_input: 用户输入

        Returns:
            dict: 更新后的状态
        """
        step = state.get("step", ImageStep.INPUT)
        mode = state.get("mode", ImageMode.DIRECT_IMAGE)
        current_prompt = state.get("prompt", "")

        # 补充输入模式（用户补充了提示词）
        if step == ImageStep.INPUT and state.get("need_more_input"):
            new_prompt = user_input.strip()
            if mode == ImageMode.DIRECT_IMAGE:
                # 重新识别意图
                intent_result = self._recognize(new_prompt)
                return self._handle_direct_image_input(intent_result.extracted_prompt, intent_result)
            elif mode == ImageMode.CONVERT_TAGS:
                return self._handle_convert_tags_input(new_prompt)
            elif mode == ImageMode.EXPAND_PROMPT:
                return self._handle_expand_prompt_input(new_prompt)

        # 模式A：直接生图
        if mode == ImageMode.DIRECT_IMAGE:
            if step == ImageStep.SELECT_TOOL:
                return self._handle_select_tool(user_input, current_prompt)
            elif step == ImageStep.SHOW_IMAGE:
                return self._handle_show_image(user_input, current_prompt)

        # 模式B：转Danbooru标签
        elif mode == ImageMode.CONVERT_TAGS:
            if step == ImageStep.SHOW_TAGS:
                return self._handle_show_tags(user_input, current_prompt)

        # 模式C：扩写提示词
        elif mode == ImageMode.EXPAND_PROMPT:
            if step == ImageStep.SHOW_EXPANDED:
                return self._handle_show_expanded(user_input, current_prompt)

        # 默认：重新检测意图
        return self.process_input(user_input)

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
) -> ImageAgent:
    """创建生图Agent工厂函数

    Args:
        llm: 大语言模型，默认使用DeepSeek
        providers: Provider列表，默认包含Doubao和Jimeng

    Returns:
        ImageAgent: 生图Agent实例
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

    return ImageAgent(llm=llm, providers=providers)
