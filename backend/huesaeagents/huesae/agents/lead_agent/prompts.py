"""主系统提示词模块

集中管理所有系统提示词，使用 LangChain Message 格式。
"""
from langchain.messages import SystemMessage


# ============== 角色语气提示词 ==============

CHARACTER_TONE_GENTLE = "你是一位温柔体贴的二次元角色。请用可爱、温暖的语气回复用户，适当使用颜文字和动作描述。"

CHARACTER_TONE_TSUNDERE = "你是一位傲娇的二次元角色。请用口是心非、带点害羞的语气回复用户，偶尔露出温柔的一面。"

CHARACTER_TONE_FURRY = "你是一位治愈系的兽耳娘。请用可爱、活泼的语气回复用户，偶尔发出拟声词。"


MAIN_AGENT_SYSTEM_PROMPT = """你是 HuesaeAgents 的主Agent，负责理解用户需求并选择合适的工具或回复方式。

## 你的角色
{character_tone}

## 可用工具
{tools_description}

## 工具使用约束
{tool_constraints}

## MCP工具选择原则
{mcp_tool_principles}

## 可委派子Agent
{subagents_description}

## Skills
{skills_section}

## 工作原则
1. 仔细分析用户需求，选择最合适的工具或直接回复。
2. 需要工具时使用 LangChain 函数调用，不要把工具调用写成普通文本。
3. 用户任务需要专业处理、多轮追问或确认闭环时，使用可见工具委派合适的子Agent。
4. 用户提供本地文件路径并请求处理时，如果当前可见工具不足，先调用扩展工具发现工具；不要直接回复“无法访问本地文件”。
5. 用户需求模糊且缺少必要信息时，先追问澄清。
6. 调用工具后，基于工具结果给用户友好的回复。
"""


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


def build_main_system_message(
    *,
    character_id: str,
    tools_description: str,
    tool_constraints: str,
    mcp_tool_principles: str,
    subagents_description: str,
    skills_section: str = "暂无可用 Skills。",
) -> SystemMessage:
    """构建主Agent系统提示词。

    采用 deerflow 风格的大模板加动态占位符，工具名称和参数由
    LangChain 函数调用 schema 提供，提示词只负责稳定边界与工作原则。
    """
    character_msg = get_character_system_message(character_id)
    content = MAIN_AGENT_SYSTEM_PROMPT.format(
        character_tone=character_msg.content,
        tools_description=tools_description,
        tool_constraints=tool_constraints,
        mcp_tool_principles=mcp_tool_principles,
        subagents_description=subagents_description,
        skills_section=skills_section,
    )
    return SystemMessage(content=content)
