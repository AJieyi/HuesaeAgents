"""Agent 工具入口（DeerFlow Harness Engineering 模式）。

工具选择由 LLM 自主决定，系统只提供工具列表和描述。
主Agent通过 ReAct 循环让 LLM 自主决策调用哪个工具。
"""
from dataclasses import dataclass
from typing import Literal

from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from ..subagents.registry import SubAgentRegistry


# ============== LLM 决策模型 ==============

class Action(BaseModel):
    """LLM 决策：选择直接回复或调用工具。"""

    thought: str = Field(description="分析用户需求和当前状态，思考应该采取什么行动")
    type: Literal["reply", "tool_call"] = Field(
        description="行动类型：reply=直接回复用户，tool_call=调用工具"
    )
    tool_name: str | None = Field(
        default=None,
        description="当 type=tool_call 时，要调用的工具名称（generate_image_tool/generate_images_tool/expand_prompt_tool/convert_tags_tool/task_tool）"
    )
    tool_args: dict | None = Field(
        default=None,
        description="当 type=tool_call 时，以JSON对象形式传递工具参数"
    )
    response: str | None = Field(
        default=None,
        description="当 type=reply 时，给用户的直接回复内容"
    )


SUBAGENT_TASK_PREFIX = "__SUBAGENT_TASK__"


def encode_subagent_task(subagent_type: str, description: str) -> str:
    """编码子Agent委派结果，供主Agent识别。"""
    return f"{SUBAGENT_TASK_PREFIX}:{subagent_type}:{description}"


def parse_subagent_task(result: str) -> tuple[str, str] | None:
    """解析子Agent委派结果。"""
    if not result.startswith(SUBAGENT_TASK_PREFIX):
        return None
    parts = result.split(":", 2)
    if len(parts) < 3:
        return None
    return parts[1], parts[2]


@dataclass
class ToolRegistry:
    """运行时工具注册表。"""

    tools: list[BaseTool]

    def __init__(self):
        self.tools = []
        self._tool_map: dict[str, BaseTool] = {}

    def register(self, tool_obj: BaseTool) -> None:
        """注册单个工具。"""
        if tool_obj.name in self._tool_map:
            return
        self.tools.append(tool_obj)
        self._tool_map[tool_obj.name] = tool_obj

    def extend(self, tools: list[BaseTool]) -> None:
        """批量注册工具。"""
        for tool_obj in tools:
            self.register(tool_obj)

    def get(self, name: str) -> BaseTool | None:
        """按名称获取工具。"""
        return self._tool_map.get(name)

    def names(self) -> list[str]:
        """返回所有工具名称。"""
        return list(self._tool_map.keys())


# ============== 工具创建工厂 ==============

def create_tools(
    llm: BaseChatModel,
    subagent_registry: SubAgentRegistry | None = None,
) -> list[BaseTool]:
    """创建主Agent可用工具列表。"""

    # 延迟导入避免循环依赖。
    try:
        from huesaeagents.huesae.tools.doubao import create_doubao_client
        from huesaeagents.huesae.subagents.image import expand_prompt
        from huesaeagents.huesae.subagents.image.prompts import DANBOORU_SYSTEM_MESSAGE
    except ImportError:
        from .doubao import create_doubao_client
        from ..subagents.image import expand_prompt
        from ..subagents.image.prompts import DANBOORU_SYSTEM_MESSAGE

    @tool
    def generate_image_tool(prompt: str, size: str = "2K", output_format: str = "jpeg") -> str:
        """生成单张图片。当用户明确要求生成1张图片、画画、绘图时使用此工具。

        用户输入示例："生成一张夕阳下的大海"、"画一只猫"等。

        Args:
            prompt: 图片描述提示词，如"一个银发红瞳的少女在樱花树下"
            size: 图片尺寸，支持 1K, 2K, 3K, 4K，默认 2K
            output_format: 输出格式，jpeg 或 png，默认 jpeg
        """
        client = create_doubao_client()
        url = client.generate_image(
            prompt=prompt,
            size=size,
            output_format=output_format,
        )
        return f"图片已生成，URL: {url}"

    @tool
    def generate_images_tool(prompt: str, size: str = "2K", output_format: str = "jpeg") -> str:
        """生成一组连贯图片（组图）。当用户明确要求生成多张图片（如"生成4张""来3张图"）时使用此工具。

        用户输入示例："生成4张四季插画"、"来一组头像"等。

        Args:
            prompt: 图片描述提示词，需保留数量描述如"生成4张四季插画"
            size: 图片尺寸，支持 1K, 2K, 3K, 4K，默认 2K
            output_format: 输出格式，jpeg 或 png，默认 jpeg
        """
        client = create_doubao_client()
        images = client.generate_images(
            prompt=prompt,
            size=size,
            max_images=12,
            output_format=output_format,
        )
        urls = [img["url"] for img in images if img.get("url")]
        return "图片已生成，URL列表:\n" + "\n".join(urls)

    @tool
    def expand_prompt_tool(prompt: str) -> str:
        """扩写图片提示词。当用户要求扩写、丰富、扩展图片描述时使用此工具。

        用户输入示例："扩写：一个少女在樱花树下"、"帮我写详细点"等。

        Args:
            prompt: 用户提供的简短图片描述
        """
        expanded = expand_prompt(prompt, llm)
        return expanded

    @tool
    def convert_tags_tool(description: str) -> str:
        """将自然语言描述转换为Danbooru标签。当用户要求生成标签、转成Danbooru标签时使用此工具。

        用户输入示例："转成Danbooru标签：一个猫娘在咖啡馆"等。

        Args:
            description: 自然语言描述
        """
        messages = [DANBOORU_SYSTEM_MESSAGE, HumanMessage(content=description)]
        response = llm.invoke(messages)
        return response.content

    @tool
    def task_tool(description: str, subagent_type: str = "image") -> str:
        """委托子Agent处理复杂任务。当任务需要多步骤、专业处理、多轮对话时使用此工具。

        使用场景：
        - 用户说"我想生成图片"但没有提供具体描述（需要追问）
        - 用户要求推荐图片主题
        - 需要多轮确认的生图流程

        支持的子Agent：
        - image: 生图对话Agent，处理追问、推荐、扩写、确认、生图完整流程

        Args:
            description: 任务描述，即用户原始输入
            subagent_type: 子Agent类型，当前支持 "image"
        """
        if subagent_registry is not None and subagent_type not in subagent_registry.names():
            available = ", ".join(subagent_registry.names()) or "无"
            return f"错误：未知子Agent {subagent_type}。可用子Agent：{available}"

        # task_tool 不实际执行子Agent；它向主Agent返回一个结构化标记，
        # 由主Agent统一管理子Agent上下文和多轮状态。
        return encode_subagent_task(subagent_type, description)

    if subagent_registry is not None:
        task_tool.description = (
            task_tool.description
            + "\n\n当前可用子Agent：\n"
            + subagent_registry.format_for_prompt()
        )

    registry = ToolRegistry()
    registry.extend([
        generate_image_tool,
        generate_images_tool,
        expand_prompt_tool,
        convert_tags_tool,
        task_tool,
    ])
    return registry.tools
