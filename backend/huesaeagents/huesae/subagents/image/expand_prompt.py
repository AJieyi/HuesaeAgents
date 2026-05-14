"""提示词扩写器

将用户的简短提示词扩写为更丰富的自然语言描述。
与Danbooru标签生成不同，扩写的是自然语言。
"""
from langchain_core.language_models import BaseChatModel
from langchain.messages import HumanMessage


DEFAULT_EXPAND_PROMPT = """你是一个专业的AI绘画提示词扩写专家。

你的任务是将用户简短的图片描述扩写为更丰富、更详细的描述。

扩写原则：
1. 保留用户原始描述的核心元素（角色、场景、动作等）
2. 增加细节：光线、氛围、视角、情绪、材质等
3. 使用生动、具体的形容词
4. 输出为自然语言，不是标签
5. 长度控制在100字以内
6. 不要输出解释，只输出扩写后的描述

示例输入：一个银发红瞳的少女在樱花树下
示例输出：一位银发如瀑布般倾泻的少女，红宝石般的眼眸中映着春日的暖阳。她静静地站在盛开的樱花树下，粉色花瓣随风飘落，点缀在她洁白的连衣裙上。柔和的光线从侧面洒下，为整个场景增添了一层梦幻的滤镜。少女微微抬头，嘴角带着淡淡的微笑，仿佛在享受着这美好的春日时光。"""


def expand_prompt(user_input: str, llm: BaseChatModel) -> str:
    """扩写用户的自然语言提示词

    Args:
        user_input: 用户输入的简短描述
        llm: 大语言模型实例

    Returns:
        str: 扩写后的自然语言描述
    """
    messages = [
        HumanMessage(content=DEFAULT_EXPAND_PROMPT),
        HumanMessage(content=f"请扩写以下描述：\n{user_input}"),
    ]

    response = llm.invoke(messages)
    return response.content.strip()
