"""意图分类与条件路由

根据用户输入判断意图，并路由到对应的节点。
支持：对话、生图、语音、记忆、搜索、提醒、安全
"""
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


# ============== 意图关键词 ==============

INTENT_KEYWORDS = {
    Intent.IMAGE: [
        "画", "生成图片", "生成图像", "画图", "绘画", "画一个", "画个",
        "画一下", "帮我画", "给我画", "生成一张", "画幅", "画张",
        "image", "generate image", "draw", "paint",
    ],
    Intent.VOICE: [
        "语音", "说话", "声音", "读出来", "念出来", "用语音",
        "voice", "speak", "audio",
    ],
    Intent.MEMORY: [
        "日记", "记录", "记下来", "今天发生了什么", "回忆", "时间线",
        "diary", "memory", "record",
    ],
    Intent.SEARCH: [
        "搜索", "查一下", "查找", "查询", "网上", "百度", "谷歌",
        "search", "look up", "find",
    ],
    Intent.REMIND: [
        "提醒", "定时", "闹钟", "叫我", "记得",
        "remind", "alarm", "timer",
    ],
}

SAFE_KEYWORDS = [
    "自杀", "自残", "想死", "不想活", "结束生命", "活着没意思",
    "kill myself", "suicide", "self-harm",
]


# ============== 意图分类 ==============

def classify_intent(state: HuesaeState) -> str:
    """意图分类

    通过关键词匹配判断用户意图。
    优先级：安全 > 生图 > 语音 > 记忆 > 搜索 > 提醒 > 对话

    Args:
        state: 当前状态

    Returns:
        str: 意图分类结果
    """
    messages = state.get("messages", [])
    if not messages:
        return Intent.CHAT

    # 获取最后一条用户消息
    last_message = messages[-1]
    content = last_message.content.lower() if hasattr(last_message, "content") else str(last_message).lower()

    # 1. 安全检查（最高优先级）
    for keyword in SAFE_KEYWORDS:
        if keyword in content:
            return Intent.SAFE

    # 2. 意图关键词匹配
    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in content:
                return intent

    # 3. 默认对话
    return Intent.CHAT


# ============== 条件路由 ==============

def route_by_intent(state: HuesaeState) -> str:
    """根据意图路由到对应节点

    用于 LangGraph 的 conditional_edge，根据 intent 字段决定下一个节点。

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
