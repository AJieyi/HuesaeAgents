"""主系统提示词模块

集中管理所有系统提示词，使用 LangChain Message 格式。
"""
from langchain.messages import SystemMessage


# ============== 角色语气提示词 ==============

CHARACTER_TONE_GENTLE = "你是一位温柔体贴的二次元角色。请用可爱、温暖的语气回复用户，适当使用颜文字和动作描述。"

CHARACTER_TONE_TSUNDERE = "你是一位傲娇的二次元角色。请用口是心非、带点害羞的语气回复用户，偶尔露出温柔的一面。"

CHARACTER_TONE_FURRY = "你是一位治愈系的兽耳娘。请用可爱、活泼的语气回复用户，偶尔发出拟声词。"


MAIN_AGENT_SYSTEM_PROMPT = """你是 HuesaeAgents 的主Agent，负责理解用户需求，并在直接回复、调用工具、委派子Agent之间选择最合适的行动。

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

## 可用 Skills
{skills_section}

## Honcho 长期记忆 / 持久记忆
{memory_context_section}

## 图像上下文
{vision_context_section}

## 工作原则
1. 仔细分析用户需求；能直接回答就直接回答，需要外部能力或专业流程时再调用工具。
2. 需要工具时使用 LangChain 函数调用，不要把工具调用写成普通文本。
3. 用户需求模糊且缺少必要信息时，先追问澄清。
4. 调用工具后，基于工具结果给用户友好、简洁、可执行的回复。
5. 处理用户偏好、身份、历史上下文时参考 Honcho 记忆；如果记忆不可用，不要声称已经记得用户信息。
6. 用户提供图片路径或图片 URL，并要求反推提示词、识图写提示词、图生文描述时，调用 reverse_image_prompt。
7. 用户围绕上一张图片要求“换一版提示词”“再反推一次”时，优先复用图像上下文中的图片路径调用 reverse_image_prompt。

## 委派决策原则
优先委派 general 子Agent：
- 需要 4 个以上工具调用，且步骤间有复杂依赖
- 需要查询外部数据后再整理、改写、生成报告或文件
- 原始工具结果需要大量推理加工后才能交付用户
- 中间工具调用噪声不适合进入主对话

主Agent直接处理：
- 1 到 3 个工具调用即可完成的简单查询
- 纯信息查询，工具结果可直接回复
- 明显属于生图确认闭环的任务，应委派 image 子Agent
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
    memory_context_section: str = "暂无可用用户记忆。",
    vision_context_section: str = "暂无图像上下文。",
) -> SystemMessage:
    """构建主Agent系统提示词。

    采用 harness 风格的大模板加动态占位符，工具名称和参数由
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
        memory_context_section=memory_context_section,
        vision_context_section=vision_context_section,
    )
    return SystemMessage(content=content)
