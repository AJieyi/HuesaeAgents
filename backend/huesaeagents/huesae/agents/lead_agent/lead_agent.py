"""主Agent（Lead Agent）

对话核心，负责：
1. 意图分类（LLM驱动，基于完整对话历史，返回 intent + image_intent）
2. 委派子Agent或直接聊天回复
3. 包装子Agent结果，保持角色语气

无状态设计，每次调用接收完整对话历史。
"""
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from ..subagents.base import BaseSubAgent


# ============== 意图常量 ==============

class Intent:
    """意图分类常量"""

    CHAT = "chat"
    IMAGE = "image"
    VOICE = "voice"
    MEMORY = "memory"
    SEARCH = "search"
    REMIND = "remind"
    SAFE = "safe"


# ============== 安全检查 ==============

_SAFE_KEYWORDS = [
    "自杀", "自残", "想死", "不想活", "结束生命", "活着没意思",
    "kill myself", "suicide", "self-harm",
]


# ============== LLM意图分类模型 ==============

class IntentResult(BaseModel):
    """LLM意图识别结果"""

    intent: Literal[
        "chat", "image", "voice", "memory", "search", "remind"
    ] = Field(
        description="用户意图：chat=普通聊天, image=生图/画画/图片相关(含扩写、标签), "
                    "voice=语音/声音, memory=记忆/日记/记录, search=搜索/查询, "
                    "remind=提醒/闹钟/定时"
    )
    image_intent: Literal[
        "generate_image", "expand_prompt", "convert_tags"
    ] | None = Field(
        default=None,
        description="当 intent=image 时的子分类："
                    "generate_image=生成图片(默认), "
                    "expand_prompt=扩写提示词, "
                    "convert_tags=转成Danbooru标签"
    )
    reason: str = Field(default="", description="判断理由")


# ============== 主Agent ==============

class HuesaeMainAgent:
    """主Agent：对话核心 + 子Agent委派

    每轮接收用户输入，用 LLM 做意图分类，然后：
    - 需要子Agent：调用子Agent获取结果，主Agent包装展示
    - 不需要子Agent：主Agent直接聊天回复

    Example:
        >>> from huesae.models.models_factory import create_chat_model
        >>> from huesae.agents.subagents.image_agent import create_image_agent
        >>> main = HuesaeMainAgent(llm=create_chat_model("deepseek"))
        >>> main.register_sub_agent(create_image_agent())
        >>> result = main.process({"messages": []}, "我想生成图片")
        >>> print(result["messages"][0].content)
        '请告诉我您想要生成什么样的图片？'
    """

    def __init__(
        self,
        llm: BaseChatModel,
        character_id: str = "gentle_sister",
    ):
        self.llm = llm
        self.character_id = character_id
        self.sub_agents: dict[str, BaseSubAgent] = {}

    def register_sub_agent(self, agent: BaseSubAgent) -> None:
        """注册子Agent"""
        self.sub_agents[agent.name] = agent

    # ============== 主入口 ==============

    def process(self, state: dict, user_input: str) -> dict:
        """处理用户输入

        Args:
            state: 当前状态，包含 messages 对话历史
            user_input: 用户最新输入

        Returns:
            dict: 包含 messages 列表，可选 pending_generation/prompt/image_intent/clear_image_intent
        """
        # 1. 安全检查（最高优先级）
        if self._check_safety(user_input):
            return {
                "messages": [AIMessage(content=self._safety_response())],
                "safety_flag": True,
                "high_risk_flag": True,
            }

        # 2. 意图分类
        # 如果当前已有 image_intent，说明在 image 流程中，直接复用意图
        # 避免每轮都调用 LLM 做意图分类，节省调用次数
        if state.get("image_intent"):
            intent = Intent.IMAGE
        else:
            intent_result = self._classify_intent(state, user_input)
            intent = intent_result.intent
            if intent == Intent.IMAGE:
                state = dict(state)
                state["image_intent"] = intent_result.image_intent or "generate_image"

        # 3. 如果意图匹配子Agent，调用子Agent
        if intent in self.sub_agents:
            state = dict(state)  # 复制，避免修改外部原始对象
            return self._handle_sub_agent(intent, state, user_input)

        # 4. 否则主Agent直接聊天回复
        chat_response = self._chat_reply(state, user_input)
        result = {"messages": [AIMessage(content=chat_response)]}
        # 如果之前有 image_intent，现在切回聊天，通知调用者清除
        if "image_intent" in state:
            result["clear_image_intent"] = True
        return result

    # ============== 意图分类 ==============

    def _classify_intent(self, state: dict, user_input: str) -> IntentResult:
        """用LLM做意图分类，传入完整对话历史

        让LLM同时判断主意图和 image 子意图。
        """
        messages = state.get("messages", [])

        # 构建对话历史摘要（最近6条）
        history_text = self._format_history(messages[-6:])

        prompt = f"""分析以下用户输入的意图，进行分类。

当前对话历史：
{history_text}

用户最新输入：{user_input}

分类规则：
- chat：普通聊天、问候、日常问答（如"你好""今天天气如何""真好看""谢谢"）
- image：与图片生成、画画、绘图、扩写、标签相关的需求
  - 如果用户输入包含"扩写""扩展""丰富"等词，intent=image，image_intent=expand_prompt
  - 如果用户输入包含"标签""danbooru""tag"等词，intent=image，image_intent=convert_tags
  - 如果对话历史中用户之前说过"我想生成图片"等，且当前输入是描述性的，intent=image，image_intent=generate_image
  - 但如果用户说"真好看""谢谢""换个话题"等，intent=chat
- voice：与语音、声音、朗读相关的需求
- memory：与记忆、日记、记录、回忆相关的需求
- search：与搜索、查询、查找信息相关的需求
- remind：与提醒、闹钟、定时、日程相关的需求

请以JSON格式输出结果。"""

        try:
            structured_llm = self.llm.with_structured_output(
                IntentResult,
                method="json_mode",
            )
            result = structured_llm.invoke([HumanMessage(content=prompt)])
            return result
        except Exception:
            # Fallback：简单规则判断
            return IntentResult(intent=Intent.CHAT, reason="Fallback默认聊天")

    def _format_history(self, messages: list) -> str:
        """格式化对话历史为文本"""
        lines = []
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                role = "用户" if getattr(msg, "type", "") == "human" else "AI"
                lines.append(f"{role}：{msg.content}")
        return "\n".join(lines) if lines else "（无历史对话）"

    # ============== 子Agent处理 ==============

    def _handle_sub_agent(self, intent: str, state: dict, user_input: str) -> dict:
        """调用子Agent并包装结果

        子Agent使用独立的 image_context 作为对话历史，
        避免生图中间过程污染主Agent的消息列表。
        """
        agent = self.sub_agents[intent]
        image_intent = state.get("image_intent", "generate_image")

        # 获取或创建子上下文（生图Agent独立对话历史）
        image_context = list(state.get("image_context", []))

        # 构建子Agent的 state，使用 image_context 作为对话历史
        sub_state = {
            "messages": image_context,
            "image_intent": image_intent,
        }

        sub_result = agent.process(sub_state, user_input)
        action = sub_result.get("action", "")

        # 子Agent返回的是"生图流程中的一步"（追问/推荐/确认）
        if action in ("ask_prompt", "recommend", "ask_confirm"):
            # 将本轮对话追加到子上下文
            image_context.append(HumanMessage(content=user_input))
            image_context.append(AIMessage(content=sub_result["response"]))
            return {
                "messages": [AIMessage(content=sub_result["response"])],
                "image_intent": image_intent,
                "image_context": image_context,
            }

        # 子Agent返回的是"执行生图"
        if action == "generate":
            # 使用子Agent返回的 prompt，如果没有则使用上一次保存的 prompt（换图场景）
            prompt = sub_result.get("prompt") or state.get("current_image_prompt", "")
            # 从子Agent结果中获取生图参数
            size = sub_result.get("size", "2K")
            output_format = sub_result.get("output_format", "jpeg")
            # 将本轮对话追加到子上下文（保留历史，用于后续换图/扩写）
            image_context.append(HumanMessage(content=user_input))
            image_context.append(
                AIMessage(content=sub_result.get("response", "图片正在生成中，请稍等~"))
            )
            return {
                "messages": [
                    AIMessage(content=sub_result.get("response", "图片正在生成中，请稍等~"))
                ],
                "pending_generation": True,
                "prompt": prompt,
                "size": size,
                "output_format": output_format,
                "image_intent": image_intent,
                "image_context": image_context,
            }

        # 子Agent返回结束
        if action == "finish":
            if image_intent == "generate_image":
                # 生图意图：用主Agent角色语气回复结束语
                chat_response = self._chat_reply(state, user_input)
                return {
                    "messages": [AIMessage(content=chat_response)],
                    "clear_image_intent": True,
                    "image_context": [],  # 清空子上下文
                }
            else:
                # 扩写/标签意图：直接返回子Agent的响应（已包含扩写/标签结果）
                return {
                    "messages": [AIMessage(content=sub_result["response"])],
                    "clear_image_intent": True,
                    "image_context": [],  # 清空子上下文
                }

        # 默认：直接展示子Agent的回复
        return {
            "messages": [AIMessage(content=sub_result["response"])],
            "image_intent": image_intent,
            "image_context": image_context,
        }

    async def execute_image_generation(
        self,
        prompt: str,
        size: str = "2K",
        output_format: str = "jpeg",
    ) -> dict:
        """执行生图并返回结果（供外部调用）

        Args:
            prompt: 生图提示词
            size: 图片尺寸，默认 2K
            output_format: 输出格式，默认 jpeg

        Returns:
            dict: {wrap_message, image_url} 或抛出异常
        """
        agent = self.sub_agents.get(Intent.IMAGE)
        if not agent:
            raise ValueError("Image agent not registered")

        generation = await agent.generate_image(
            prompt=prompt,
            size=size,
            output_format=output_format,
        )
        wrap_msg = self._create_wrap_message()

        return {
            "wrap_message": wrap_msg,
            "image_url": generation.url,
        }

    def _create_wrap_message(self) -> str:
        """用主Agent的角色语气生成图片展示语"""
        from ..subagents.image.prompts import get_character_system_message

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
        from ..subagents.image.prompts import get_character_system_message

        character_msg = get_character_system_message(self.character_id)
        messages = [character_msg] + state.get("messages", []) + [HumanMessage(content=user_input)]
        response = self.llm.invoke(messages)
        return response.content

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

    return HuesaeMainAgent(llm=llm, character_id=character_id)
