"""LangGraph 工作流定义"""
from typing import Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from .state import ThreadState


# ============ 情绪关键词映射 ============

EMOTION_KEYWORDS = {
    "开心": ["开心", "高兴", "快乐", "太好了", "棒", "耶", "欢呼"],
    "难过": ["难过", "伤心", "失落", "沮丧", "郁闷", "委屈", "想哭"],
    "害羞": ["害羞", "脸红", "不好意思", "羞", "脸红红"],
    "寂寞": ["寂寞", "孤独", "无聊", "一个人", "没人陪"],
    "愤怒": ["生气", "愤怒", "讨厌", "烦", "气死了", "讨厌"],
    "害怕": ["害怕", "担心", "紧张", "焦虑", "不安", "怕"],
}

# ============ 颜文字映射 ============

EMOTION_KAOMOJI = {
    "开心": ["(≧▽≦)/", "(*^▽^*)", "ヽ(○´∀`)ﾉ♪", "✧٩(ˊωˋ*)و✧"],
    "难过": ["(´;ω;`)", "(╥﹏╥)", "QAQ", "(´_`)｡o"],
    "害羞": ["(*/ω＼*)", "(〃'▽'〃)", "(≧﹏≦)", "(*/ω・ﾟ)"],
    "寂寞": ["(´・ω・`)", "（委屈巴巴）", "(´;ω;`)｡oO"],
    "愤怒": ["(╬▔皿▔)╯", "(｀Д´)", "(╯°□°）╯︵ ┻━┻"],
    "害怕": ["(°△°|||)", "(ﾟДﾟ≡ﾟДﾟ)", "Σ(°△°|||)"],
}

# ============ 角色动作映射 ============

CHARACTER_ACTIONS = {
    "gentle_sister": {
        "开心": ["轻轻微笑", "温柔地抚摸你的头", "欣慰地点头"],
        "难过": ["轻轻拥抱你", "温柔地拍拍你的背", "担忧地看着你"],
        "害羞": ["别过脸去", "小声说话", "偷偷看你"],
        "寂寞": ["靠近你坐下", "轻轻握住你的手", "安静地陪着你"],
        "愤怒": ["深呼吸", "冷静地看着你", "叹气"],
        "害怕": ["担忧地皱眉", "紧张地看着你", "紧紧抓住你的衣角"],
    },
    "tsundere": {
        "开心": ["别过脸去偷笑", "嘴硬地哼了一声", "偷偷看你"],
        "难过": ["别过脸去", "小声嘟囔", "偷偷擦眼泪"],
        "害羞": ["脸红", "把你推开", "大喊我才没有"],
        "寂寞": ["才不是在等你", "傲娇地别过头", "尾巴却垂下来"],
        "愤怒": ["才不是关心你", "生气地跺脚", "哼"],
        "害怕": ["紧握裙角", "发抖", "偷偷抓住你袖子"],
    },
    "furry_fox": {
        "开心": ["甩尾巴", "耳朵抖动", "蹭蹭你的手"],
        "难过": ["尾巴垂下", "耳朵耷拉", "蹭蹭你的膝盖"],
        "害羞": ["尾巴卷起来", "耳朵压低", "脸红"],
        "寂寞": ["蜷缩成一团", "抱着尾巴", "轻轻呜咽"],
        "愤怒": ["炸毛", "尾巴蓬松起来", "龇牙"],
        "害怕": ["缩进你怀里", "尾巴夹紧", "耳朵后压"],
    },
}


# ============ 节点定义 ============

def input_node(state: ThreadState) -> ThreadState:
    """
    输入处理节点

    解析用户输入，可扩展为：
    - 消息格式化
    - 敏感词过滤
    - 意图识别

    Args:
        state: 当前线程状态

    Returns:
        更新后的线程状态
    """
    return state


def emotion_detect_node(state: ThreadState) -> ThreadState:
    """
    情绪检测节点

    通过关键词匹配检测用户情绪

    Args:
        state: 当前线程状态，包含 messages

    Returns:
        更新后的线程状态，包含 emotion_state 和 emotion_score
    """
    messages = state.get("messages", [])
    if not messages:
        return state

    # 获取用户最后一条消息
    last_message = messages[-1]
    if hasattr(last_message, "content"):
        content = last_message.content.lower()
    else:
        content = str(last_message).lower()

    # 关键词匹配
    detected_emotion = "平静"
    max_matches = 0

    for emotion, keywords in EMOTION_KEYWORDS.items():
        matches = sum(1 for keyword in keywords if keyword in content)
        if matches > max_matches:
            max_matches = matches
            detected_emotion = emotion

    # 计算情绪强度（简单实现：匹配越多强度越高）
    emotion_score = min(max_matches * 0.3, 1.0) if max_matches > 0 else 0.0

    return {
        "emotion_state": detected_emotion,
        "emotion_score": emotion_score,
    }


def reasoner_node(state: ThreadState, model=None) -> ThreadState:
    """
    LLM 推理节点（核心）

    调用大模型生成回复

    Args:
        state: 当前线程状态，包含 messages
        model: 聊天模型实例（可选）

    Returns:
        更新后的线程状态，messages 中包含 AI 回复
    """
    # 获取消息列表
    messages = state.get("messages", [])

    # 如果没有传入模型，则创建
    if model is None:
        from ..models.factory import create_chat_model
        model = create_chat_model("deepseek")

    response = model.invoke(messages)

    # 将 AI 回复添加到消息列表
    return {"messages": [response]}


def output_node(state: ThreadState) -> ThreadState:
    """
    输出格式化节点

    根据情绪状态添加颜文字和动作描述

    Args:
        state: 当前线程状态，包含 emotion_state, character_id

    Returns:
        更新后的线程状态
    """
    emotion = state.get("emotion_state", "平静")
    character_id = state.get("character_id", "gentle_sister")
    messages = state.get("messages", [])

    if not messages:
        return state

    # 获取AI最后一条回复
    last_ai_message = messages[-1]
    if not isinstance(last_ai_message, AIMessage):
        return state

    content = last_ai_message.content

    # 获取颜文字
    kaomojis = EMOTION_KAOMOJI.get(emotion, [""])
    kaomojis = kaomojis if kaomojis else [""]
    kaomojis_str = "".join(kaomojis[:2])  # 取前2个颜文字

    # 获取角色动作
    character_actions = CHARACTER_ACTIONS.get(character_id, {})
    actions = character_actions.get(emotion, [])
    action_str = actions[0] if actions else ""

    # 构建带颜文字和动作的回复
    if kaomojis_str or action_str:
        if action_str:
            formatted_content = f"*{action_str}*\n\n{content}"
        else:
            formatted_content = content
        if kaomojis_str:
            formatted_content = f"{formatted_content}\n\n{kaomojis_str}"

        # 更新消息内容
        new_message = AIMessage(content=formatted_content)
        # 替换最后一条消息
        new_messages = messages[:-1] + [new_message]
        return {"messages": new_messages}

    return state


# ============ 工作流构建 ============

def create_workflow(checkpointer: Any = None) -> StateGraph:
    """
    创建 LangGraph 工作流

    工作流结构（5节点）：
    input -> emotion_detect -> reasoner -> output -> END

    Args:
        checkpointer: 持久化检查点，默认使用内存存储

    Returns:
        编译后的 StateGraph
    """
    # 默认使用内存存储
    if checkpointer is None:
        checkpointer = MemorySaver()

    # 创建状态图
    workflow = StateGraph(ThreadState)

    # 添加节点
    workflow.add_node("input", input_node)
    workflow.add_node("emotion_detect", emotion_detect_node)
    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node("output", output_node)

    # 设置入口点
    workflow.set_entry_point("input")

    # 定义边
    workflow.add_edge("input", "emotion_detect")
    workflow.add_edge("emotion_detect", "reasoner")
    workflow.add_edge("reasoner", "output")
    workflow.add_edge("output", END)

    # 编译工作流（带持久化支持）
    return workflow.compile(checkpointer=checkpointer)
