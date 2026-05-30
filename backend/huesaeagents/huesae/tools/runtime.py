"""共享工具运行时。

主Agent和未来的通用子Agent都从这里获取工具视图。
子Agent视图会过滤掉 task_tool，避免子Agent嵌套委派。
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ..mcp.cache import initialize_mcp_tools
from ..skills.registry import SkillRegistry
from ..subagents.registry import SubAgentRegistry


MAIN_AGENT_EXCLUDED_TOOL_NAMES = frozenset({"generate_image_tool", "generate_images_tool"})
GENERAL_AGENT_EXCLUDED_TOOL_NAMES = frozenset({"generate_image_tool", "generate_images_tool"})


class SharedToolRuntime:
    """合并内置工具与 MCP 工具的共享池。"""

    def __init__(
        self,
        llm: BaseChatModel,
        subagent_registry: SubAgentRegistry | None = None,
        *,
        mcp_tools_loader=initialize_mcp_tools,
        skill_registry: SkillRegistry | None = None,
    ):
        self.llm = llm
        self.subagent_registry = subagent_registry
        self.skill_registry = skill_registry
        self._mcp_tools_loader = mcp_tools_loader
        self._builtin_tools: list[BaseTool] = []
        self._mcp_tools: list[BaseTool] | None = None

    @property
    def mcp_loaded(self) -> bool:
        """MCP 工具是否已经完成 discovery。"""
        return self._mcp_tools is not None

    def refresh_builtin_tools(self) -> None:
        """刷新内置工具，通常在子Agent注册变化后调用。"""
        from .tools import get_builtin_tools

        self._builtin_tools = get_builtin_tools(
            self.llm,
            self.subagent_registry,
            skill_registry=self.skill_registry,
        )

    def refresh_mcp_tools(self, force: bool = False) -> None:
        """刷新 MCP 工具缓存。"""
        self._mcp_tools = self._mcp_tools_loader(force=force)

    def get_tools(
        self,
        *,
        include_mcp: bool = True,
        include_task_tool: bool = True,
        exclude_names: set[str] | frozenset[str] | None = None,
    ) -> list[BaseTool]:
        """返回调用方可见的工具列表。"""
        if not self._builtin_tools:
            self.refresh_builtin_tools()

        tools = list(self._builtin_tools)
        if include_mcp:
            if self._mcp_tools is None:
                self._mcp_tools = self._mcp_tools_loader()
            tools.extend(self._mcp_tools)

        if not include_task_tool:
            tools = [tool for tool in tools if tool.name != "task_tool"]

        if exclude_names:
            tools = [tool for tool in tools if tool.name not in exclude_names]

        seen: set[str] = set()
        unique_tools: list[BaseTool] = []
        for tool in tools:
            if tool.name in seen:
                continue
            seen.add(tool.name)
            unique_tools.append(tool)
        return unique_tools

    def get_tool_map(
        self,
        *,
        include_mcp: bool = True,
        include_task_tool: bool = True,
        exclude_names: set[str] | frozenset[str] | None = None,
    ) -> dict[str, BaseTool]:
        """返回工具名到工具对象的映射。"""
        return {
            tool.name: tool
            for tool in self.get_tools(
                include_mcp=include_mcp,
                include_task_tool=include_task_tool,
                exclude_names=exclude_names,
            )
        }

    def format_tools_for_prompt(
        self,
        *,
        include_mcp: bool = True,
        include_task_tool: bool = True,
        exclude_names: set[str] | frozenset[str] | None = None,
    ) -> str:
        """格式化工具描述，供系统提示词注入。"""
        lines = []
        for tool in self.get_tools(
            include_mcp=include_mcp,
            include_task_tool=include_task_tool,
            exclude_names=exclude_names,
        ):
            args_desc = self._format_tool_args(tool)
            lines.append(f"- {tool.name}: {tool.description}{args_desc}")
        return "\n".join(lines)

    def format_tool_constraints(
        self,
        *,
        include_mcp: bool | None = None,
        include_task_tool: bool = True,
        exclude_names: set[str] | frozenset[str] | None = None,
    ) -> str:
        """生成当前工具视图的使用约束，供系统提示词动态注入。"""
        if include_mcp is None:
            include_mcp = self.mcp_loaded

        visible_tools = self.get_tools(
            include_mcp=include_mcp,
            include_task_tool=include_task_tool,
            exclude_names=exclude_names,
        )
        visible_names = {tool.name for tool in visible_tools}
        hidden_builtin_names = self._hidden_builtin_tool_names(exclude_names)
        lines = [
            "- 工具名称、描述和参数 schema 是选择工具的主要依据；只调用当前可见工具，不要编造工具名或参数名。",
            "- 每轮只选择一个行动：直接回复，或调用一个最合适的工具。",
        ]

        if hidden_builtin_names:
            target = "通过可见的委派工具交给合适的子Agent处理"
            if "task_tool" in visible_names and self._has_subagent("image"):
                target = "通过 task_tool 委派 image 子Agent处理完整确认闭环"
            hidden_text = "、".join(hidden_builtin_names)
            lines.append(f"- 当前Agent不可直接调用以下底层工具：{hidden_text}；相关任务请{target}。")

        if "task_tool" in visible_names:
            lines.append("- 需要专业子Agent、多轮追问或用户确认闭环时，调用 task_tool 委派对应子Agent。")

        if "load_mcp_tools_tool" in visible_names:
            if self.mcp_loaded:
                lines.append("- MCP 工具已经加载，后续优先根据具体 MCP 工具自身描述和参数 schema 选择。")
            else:
                lines.append("- 当用户需要当前可见工具之外的扩展能力时，先调用 load_mcp_tools_tool 完成 MCP 工具发现。")

        if "read_skill_tool" in visible_names:
            lines.append("- 当用户需求匹配某个 Skill 时，先调用 read_skill_tool 读取完整指令；Skill 不是工具本身。")
        if "bash_tool" in visible_names:
            lines.append("- bash_tool 仅用于执行已读取 Skill 中明确需要的命令，调用前确认命令和参数来自当前任务。")

        return "\n".join(lines)

    def format_mcp_tool_principles(self) -> str:
        """根据当前 MCP discovery 状态动态生成选择原则。"""
        if self._mcp_tools is None:
            return (
                "- 当前还未加载 MCP 工具；只有当用户任务需要外部扩展能力且当前可见工具不足时，才调用 load_mcp_tools_tool。\n"
                "- MCP 工具加载后，根据工具自身名称、描述和参数 schema 决定是否调用。"
            )

        if not self._mcp_tools:
            return "- 已尝试加载 MCP 工具，但当前没有可用 MCP 工具；请直接说明无法使用扩展工具，必要时追问配置。"

        lines = [
            "- MCP 工具来自扩展服务的实时 discovery，选择时以工具自身描述和参数 schema 为准。",
            "- 如果用户提供的本地路径、文件类型或任务目标不清楚，先追问澄清，不要盲目调用工具。",
        ]
        for tool in self._mcp_tools:
            description = (tool.description or "").strip()
            args_desc = self._format_tool_args(tool)
            lines.append(f"- {tool.name}: {description}{args_desc}")
        return "\n".join(lines)

    def _hidden_builtin_tool_names(
        self,
        exclude_names: set[str] | frozenset[str] | None,
    ) -> list[str]:
        """返回当前调用方主动隐藏的内置工具名称。"""
        if not exclude_names:
            return []
        if not self._builtin_tools:
            self.refresh_builtin_tools()
        builtin_names = {tool.name for tool in self._builtin_tools}
        return sorted(name for name in exclude_names if name in builtin_names)

    def _has_subagent(self, name: str) -> bool:
        """判断指定子Agent是否已注册。"""
        return self.subagent_registry is not None and name in self.subagent_registry.names()

    @staticmethod
    def _format_tool_args(tool: BaseTool) -> str:
        """把工具参数 schema 压缩成适合提示词展示的字段列表。"""
        visible_args = getattr(tool, "args", None)
        if isinstance(visible_args, dict) and visible_args:
            return f" 参数：{list(visible_args.keys())}"

        args_schema = getattr(tool, "args_schema", None)
        if args_schema is None or not hasattr(args_schema, "model_json_schema"):
            return ""
        try:
            schema = args_schema.model_json_schema()
        except Exception:
            return ""
        properties = schema.get("properties") or {}
        if not properties:
            return ""
        return f" 参数：{list(properties.keys())}"


def build_shared_runtime(
    llm: BaseChatModel,
    subagent_registry: SubAgentRegistry | None = None,
    *,
    mcp_tools_loader=initialize_mcp_tools,
    skill_registry: SkillRegistry | None = None,
) -> SharedToolRuntime:
    """创建共享工具运行时。"""
    runtime = SharedToolRuntime(
        llm=llm,
        subagent_registry=subagent_registry,
        mcp_tools_loader=mcp_tools_loader,
        skill_registry=skill_registry,
    )
    runtime.refresh_builtin_tools()
    return runtime
