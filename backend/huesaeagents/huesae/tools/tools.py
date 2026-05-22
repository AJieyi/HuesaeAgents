"""Agent 工具入口（DeerFlow Harness Engineering 模式）。

工具选择由 LLM 自主决定，系统只提供工具列表和描述。
主Agent通过 ReAct 循环让 LLM 自主决策调用哪个工具。
"""
import subprocess
from typing import Literal

from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain.messages import HumanMessage

from ..skills.registry import SkillRegistry
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
        description="当 type=tool_call 时，要调用的当前可见工具名称"
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
LOAD_MCP_TOOLS_SIGNAL = "__LOAD_MCP_TOOLS__"


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


def is_load_mcp_tools_signal(result: str) -> bool:
    """判断工具结果是否要求主Agent加载 MCP 工具。"""
    return result == LOAD_MCP_TOOLS_SIGNAL


class ToolRegistry:
    """运行时工具注册表。"""

    def __init__(self):
        self.tools: list[BaseTool] = []
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

def get_builtin_tools(
    llm: BaseChatModel,
    subagent_registry: SubAgentRegistry | None = None,
    skill_registry: SkillRegistry | None = None,
) -> list[BaseTool]:
    """创建内置工具列表。"""

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
    def reverse_image_prompt(
        image_path: str,
        style: Literal["default", "alternative"] = "default",
        previous_prompt: str = "",
    ) -> str:
        """根据图片反推 AI 绘画提示词。用户提供本地图片路径或图片 URL，并要求反推提示词、识图写提示词、图生文描述时调用。

        如果用户基于上一张图要求“换一版提示词”或“再反推一版”，继续调用此工具，
        style 使用 alternative，并把上一版提示词传入 previous_prompt。

        Args:
            image_path: 图片本地绝对路径或图片 URL。
            style: default 表示标准反推，alternative 表示基于同一张图换一版描述。
            previous_prompt: 上一次反推出的提示词，仅在 style=alternative 时传入。
        """
        from ..services.vision import VisionService

        service = VisionService()
        return service.reverse_prompt(
            image_path=image_path,
            style=style,
            previous_prompt=previous_prompt,
        )

    @tool
    def load_mcp_tools_tool() -> str:
        """加载 MCP 扩展工具。当用户需要外部 MCP 能力，但当前工具列表还没有具体 MCP 工具时调用。

        使用场景：
        - 用户需要当前内置工具无法完成的外部扩展能力
        - 用户提供线上平台链接，希望解析、下载或提取内容
        - 用户提供本地文件路径，希望读取、分析或生成脚本
        - 当前工具列表中还没有具体的 MCP 工具名称
        - 需要先发现线上 MCP server 暴露的工具，再继续选择具体工具
        """
        return LOAD_MCP_TOOLS_SIGNAL

    @tool
    def read_skill_tool(skill_name: str) -> str:
        """读取指定 Skill 的完整指令。当用户需求匹配某个 Skill 时，先调用此工具获取详细工作流程。

        Args:
            skill_name: Skill 名称或别名，例如 weather、polecomic、manga-animation。
        """
        if skill_registry is None:
            return "当前未配置 Skill 注册表，无法读取 Skill。"
        return skill_registry.get_content(skill_name)

    @tool
    def bash_tool(command: str, timeout_seconds: int = 30) -> str:
        """执行 shell 命令并返回输出。仅在 Skill 指令明确要求运行命令时使用。

        Args:
            command: 要执行的 shell 命令。
            timeout_seconds: 命令超时时间，范围 1 到 120 秒。
        """
        timeout = max(1, min(int(timeout_seconds or 30), 120))
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return f"命令执行超时：超过 {timeout} 秒。"

        output = (result.stdout or "") + (result.stderr or "")
        if not output.strip():
            output = f"命令已执行完成，退出码：{result.returncode}"
        if result.returncode != 0:
            return f"命令执行失败，退出码：{result.returncode}\n{output.strip()}"
        return output.strip()

    @tool
    def task_tool(description: str, subagent_type: str = "image") -> str:
        """委托子Agent处理复杂任务。当任务需要多步骤、专业处理、多轮对话时使用此工具。

        使用场景：
        - 用户表达生图需求但没有提供具体描述（需要追问）
        - 用户要求推荐图片主题
        - 需要多轮确认的生图流程
        - 需要4步以上工具调用、信息整合、资料加工、报告生成等复杂通用任务
        - 需要外部数据查询并进行整理、改写或总结
        - 最终产物是文件、代码、报告或复杂结果文本

        支持的子Agent：
        - image: 生图对话Agent，处理追问、推荐、扩写、确认、生图完整流程
        - general: 通用任务Agent，处理复杂通用任务、工具链执行和结果汇总

        Args:
            description: 任务描述，即用户原始输入
            subagent_type: 子Agent类型，当前支持 "image" 或 "general"
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
        reverse_image_prompt,
        load_mcp_tools_tool,
        read_skill_tool,
        bash_tool,
        task_tool,
    ])
    return registry.tools


def get_available_tools(
    llm: BaseChatModel,
    subagent_registry: SubAgentRegistry | None = None,
    *,
    include_mcp: bool = True,
    include_task_tool: bool = True,
    runtime=None,
    skill_registry: SkillRegistry | None = None,
) -> list[BaseTool]:
    """获取调用方可见的工具列表。

    主Agent使用 include_task_tool=True；子Agent使用 include_task_tool=False，
    从架构上禁止子Agent继续委派其他子Agent。
    """
    if runtime is not None:
        return runtime.get_tools(
            include_mcp=include_mcp,
            include_task_tool=include_task_tool,
        )

    from .runtime import build_shared_runtime

    shared_runtime = build_shared_runtime(
        llm,
        subagent_registry,
        skill_registry=skill_registry,
    )
    return shared_runtime.get_tools(
        include_mcp=include_mcp,
        include_task_tool=include_task_tool,
    )


def create_tools(
    llm: BaseChatModel,
    subagent_registry: SubAgentRegistry | None = None,
    skill_registry: SkillRegistry | None = None,
) -> list[BaseTool]:
    """创建主Agent可用工具列表。

    兼容旧入口；新代码优先通过 SharedToolRuntime 获取工具。
    """
    return get_available_tools(
        llm,
        subagent_registry,
        include_mcp=False,
        include_task_tool=True,
        skill_registry=skill_registry,
    )
