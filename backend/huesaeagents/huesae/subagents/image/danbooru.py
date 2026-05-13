"""Danbooru 标签生成器

使用 LLM 将自然语言描述转换为 Danbooru 格式标签
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage


DEFAULT_SYSTEM_PROMPT = """你是一个专业的 Danbooru 标签生成专家。

你的任务是将用户的中文图片描述转换为高质量的 Danbooru 标签。

规则：
1. 标签使用英文，用逗号分隔
2. 包含以下维度：
   - 角色特征（发色、瞳色、发型、年龄）
   - 表情和动作
   - 服装
   - 场景和环境
   - 光线和氛围
   - 画风（anime, illustration, masterpiece 等）
3. 标签按优先级排序，最重要的在前面
4. 不要输出解释，只输出标签列表
5. 标签之间用英文逗号分隔

示例输入：一个银发红瞳的少女在樱花树下
示例输出：1girl, silver hair, red eyes, cherry blossoms, tree, petals, school uniform, smile, looking at viewer, spring, soft lighting, anime style, masterpiece"""


def generate_tags(
    user_input: str,
    llm: BaseChatModel,
    character_id: str = "",
) -> list[str]:
    """自然语言 → Danbooru标签

    Args:
        user_input: 用户输入的中文描述
        llm: 大语言模型实例
        character_id: 当前角色ID（可选，用于调整标签风格）

    Returns:
        list[str]: Danbooru标签列表
    """
    # 构建提示词
    messages = [
        HumanMessage(content=DEFAULT_SYSTEM_PROMPT),
        HumanMessage(content=f"请为以下描述生成Danbooru标签：\n{user_input}"),
    ]

    # 调用LLM
    response = llm.invoke(messages)
    content = response.content.strip()

    # 解析标签
    tags = _parse_tags(content)
    return tags


def _parse_tags(content: str) -> list[str]:
    """解析LLM返回的标签字符串

    Args:
        content: LLM返回的原始文本

    Returns:
        list[str]: 清洗后的标签列表
    """
    # 去除可能的说明文字
    lines = content.split("\n")
    tag_lines = [line for line in lines if "," in line or not line.strip().startswith(("-", "*", "1.", "2."))]
    tag_text = ",".join(tag_lines)

    # 分割并清洗
    raw_tags = tag_text.split(",")
    tags = []
    for tag in raw_tags:
        tag = tag.strip().lower()
        # 过滤空标签和说明性文字
        if tag and len(tag) > 1 and not tag.startswith(("示例", "标签", "输出")):
            tags.append(tag)

    # 去重并保持顺序
    seen = set()
    unique_tags = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)

    return unique_tags


def tags_to_prompt(tags: list[str]) -> str:
    """将标签列表拼接为Provider可用的提示词

    Args:
        tags: Danbooru标签列表

    Returns:
        str: 逗号分隔的标签字符串
    """
    return ", ".join(tags)
