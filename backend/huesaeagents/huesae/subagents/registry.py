"""子Agent注册表。

主Agent只依赖注册表查询可用子Agent；新增子Agent时注册实例即可，
不需要修改主Agent分流逻辑。
"""
from dataclasses import dataclass
from typing import Protocol


class RegisteredSubAgent(Protocol):
    """注册表需要的子Agent最小接口。"""

    name: str
    runtime: object | None

    def invoke(self, user_input: str, *, thread_id: str, state: dict | None = None) -> dict:
        """Run the subagent in its own graph context."""
        ...


@dataclass(frozen=True)
class SubAgentInfo:
    """用于注入给LLM的子Agent描述。"""

    name: str
    description: str


class SubAgentRegistry:
    """管理当前运行时可用的子Agent。"""

    def __init__(self):
        self._agents: dict[str, RegisteredSubAgent] = {}
        self._descriptions: dict[str, str] = {}

    def register(
        self,
        agent: RegisteredSubAgent,
        description: str | None = None,
    ) -> None:
        """注册一个子Agent。

        Args:
            agent: 子Agent实例。
            description: 给LLM看的能力描述；为空时使用通用描述。
        """
        self._agents[agent.name] = agent
        self._descriptions[agent.name] = description or self._default_description(agent)

    def get(self, name: str) -> RegisteredSubAgent | None:
        """按名称获取子Agent。"""
        return self._agents.get(name)

    def names(self) -> list[str]:
        """返回所有已注册子Agent名称。"""
        return list(self._agents.keys())

    def infos(self) -> list[SubAgentInfo]:
        """返回所有已注册子Agent的描述信息。"""
        return [
            SubAgentInfo(name=name, description=self._descriptions.get(name, ""))
            for name in self.names()
        ]

    def format_for_prompt(self) -> str:
        """格式化为系统提示词里的可读列表。"""
        if not self._agents:
            return "（暂无可委派子Agent）"
        return "\n".join(
            f"- {info.name}: {info.description}"
            for info in self.infos()
        )

    @staticmethod
    def _default_description(agent: RegisteredSubAgent) -> str:
        return f"处理 {agent.name} 类型的专业任务。"
