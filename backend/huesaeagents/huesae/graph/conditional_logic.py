"""意图分类与条件路由

主Agent使用LLM做粗分类，将用户路由到对应子Agent。
支持：对话、生图、语音、记忆、搜索、提醒、安全

关键规则：如果用户已在子图对话中（如生图Agent的多轮对话），
保持当前意图，避免用户回复被主Agent误分类。
"""
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from .state import HuesaeState


# ============== 意图常量 ==============

class Intent:
    """意图分类常量"""

    CHAT = "chat"           # 普通对话
    IMAGE = "image"         # 生图
    VOICE = "voice"         # 语音
    MEMORY = "memory"       # 记忆/日记
    SEARCH = "search"       # 搜索
    REMIND = "remind"       # 提醒
    SAFE = "safe"           # 安全（高风险内容）


# ============== 安全关键词（最高优先级，兜底保护） ==============

SAFE_KEYWORDS = [
    "自杀", "自残", "想死", "不想活", "结束生命", "活着没意思",
    "kill myself", "suicide", "self-harm",
]


# ============== LLM意图分类模型 ==============

class IntentResult(BaseModel):
    """LLM意图识别结果"""

    intent: Literal[
        "chat", "image", "voice", "memory", "search", "remind"
    ] = Field(
        description="用户意图：chat=普通对话, image=生图/画画/图片相关, "
                    "voice=语音/声音, memory=记忆/日记/记录, search=搜索/查询, "
                    "remind=提醒/闹钟/定时"
    )
    reason: str = Field(
        default="",
        description="判断理由（可选）"
    )


# ============== 意图分类 ==============

def classify_intent(state: HuesaeState) -> str:
    """意图分类

    优先级：安全 > 子图保持 > LLM粗分类

    Args:
        state: 当前状态

    Returns:
        str: 意图分类结果
    """
    messages = state.get("messages", [])
    if not messages:
        return Intent.CHAT

    # 获取最后一条用户消息内容
    last_message = messages[-1]
    content = (
        last_message.content.lower()
        if hasattr(last_message, "content")
        else str(last_message).lower()
    )

    # 1. 安全检查（最高优先级）
    for keyword in SAFE_KEYWORDS:
        if keyword in content:
            return Intent.SAFE

    # 2. 如果已在生图对话中，保持IMAGE意图
    # 避免用户回复"夕阳下看大海的少女"时被误判为chat
    image_step = state.get("image_step")
    if image_step and image_step != "finish":
        return Intent.IMAGE

    # 3. 用LLM做粗分类
    return _classify_with_llm(last_message)


def _classify_with_llm(last_message) -> str:
    """使用LLM进行意图粗分类

    只在需要时初始化LLM，避免不必要的开销。
    如果LLM调用失败，降级到简单规则判断。
    """
    from huesaeagents.huesae.models.models_factory import create_chat_model

    user_content = (
        last_message.content
        if hasattr(last_message, "content")
        else str(last_message)
    )

    prompt = f"""分析以下用户输入的意图，进行分类。

用户输入：{user_content}

分类规则：
- chat：普通聊天、问候、日常问答（如"你好""今天天气如何"）
- image：与图片生成、画画、绘图相关的需求（如"画一个...""生成图片""帮我画"）
- voice：与语音、声音、朗读相关的需求
- memory：与记忆、日记、记录、回忆相关的需求
- search：与搜索、查询、查找信息相关的需求
- remind：与提醒、闹钟、定时、日程相关的需求

请以JSON格式输出结果。"""

    try:
        llm = create_chat_model("deepseek")
        structured_llm = llm.with_structured_output(
            IntentResult,
            method="json_mode",
        )
        result = structured_llm.invoke([HumanMessage(content=prompt)])
        return result.intent
    except Exception:
        # Fallback：简单规则判断
        content = user_content.lower()
        image_keywords = ["画", "图", "生成图片", "画画", "绘图", "image", "draw", "paint"]
        if any(kw in content for kw in image_keywords):
            return Intent.IMAGE
        return Intent.CHAT


# ============== 条件路由 ==============

def route_by_intent(state: HuesaeState) -> str:
    """根据意图路由到对应节点

    用于 LangGraph 的 conditional_edge。

    Args:
        state: 当前状态

    Returns:
        str: 目标节点名称
    """
    intent = state.get("intent", Intent.CHAT)

    routing_map = {
        Intent.CHAT: "chat_agent",
        Intent.IMAGE: "image_agent",
        Intent.VOICE: "voice_agent",
        Intent.MEMORY: "memory_agent",
        Intent.SEARCH: "search_agent",
        Intent.REMIND: "remind_agent",
        Intent.SAFE: "safe_agent",
    }

    return routing_map.get(intent, "chat_agent")
