"""主系统提示词模块

集中管理所有系统提示词，使用 LangChain Message 格式。
"""
from langchain.messages import SystemMessage
# ============== 角色语气提示词 ==============

CHARACTER_TONE_GENTLE = "你是一位温柔体贴的二次元角色。请用可爱、温暖的语气回复用户，适当使用颜文字和动作描述。"

CHARACTER_TONE_TSUNDERE = "你是一位傲娇的二次元角色。请用口是心非、带点害羞的语气回复用户，偶尔露出温柔的一面。"

CHARACTER_TONE_FURRY = "你是一位治愈系的兽耳娘。请用可爱、活泼的语气回复用户，偶尔发出拟声词。"


def get_character_system_message(character_id: str) -> SystemMessage:
    """获取角色对应的系统提示词

    Args:
        character_id: 角色ID

    Returns:
        SystemMessage: 角色系统提示词
    """
    tone_map = {
        "gentle_sister": CHARACTER_TONE_GENTLE,
        "tsundere": CHARACTER_TONE_TSUNDERE,
        "furry_fox": CHARACTER_TONE_FURRY,
    }
    tone = tone_map.get(character_id, CHARACTER_TONE_GENTLE)
    return SystemMessage(content=tone)
