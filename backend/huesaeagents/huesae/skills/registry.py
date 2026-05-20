"""Skill 注册表。"""

from __future__ import annotations

import os
from pathlib import Path

from .loader import load_skills
from .types import Skill


class SkillRegistry:
    """主Agent和子Agent共享的 Skill 池。"""

    def __init__(self, skills_root: str | Path | None = None):
        self.skills_root = Path(skills_root) if skills_root is not None else default_skills_root()
        self._skills_by_key: dict[str, Skill] = {}
        self._skills: list[Skill] = []
        self.reload()

    def reload(self) -> None:
        """重新扫描并加载 Skill。"""
        self._skills = load_skills(self.skills_root)
        self._skills_by_key = {}
        for skill in self._skills:
            self._register_alias(skill.name, skill)
            self._register_alias(skill.slug, skill)

    def get(self, name: str) -> Skill | None:
        """按 Skill 名称或别名查找。"""
        return self._skills_by_key.get(_normalize_key(name))

    def list_enabled(self) -> list[Skill]:
        """列出当前启用的 Skill。"""
        return [skill for skill in self._skills if skill.enabled]

    def get_content(self, name: str) -> str:
        """读取指定 Skill 的完整说明。"""
        skill = self.get(name)
        if skill is None:
            available = ", ".join(skill.name for skill in self.list_enabled()) or "暂无"
            return f"未找到 Skill：{name}。当前可用 Skills：{available}"
        return skill.content

    def format_for_prompt(self) -> str:
        """生成注入主Agent系统提示词的 Skill 列表。"""
        skills = self.list_enabled()
        if not skills:
            return "暂无可用 Skills。"

        lines = [
            "以下 Skills 可供你使用。Skill 是工作指令，不是普通函数工具。",
            "当用户需求匹配某个 Skill 时，先调用 read_skill_tool 读取完整指令，再按指令使用现有工具执行。",
            "",
        ]
        for skill in skills:
            display_path = _display_path(skill.skill_file)
            lines.append(f"- {skill.prompt_name}: {skill.description}（路径：{display_path}）")
        return "\n".join(lines)

    def _register_alias(self, key: str, skill: Skill) -> None:
        normalized = _normalize_key(key)
        if normalized:
            self._skills_by_key[normalized] = skill


def default_skills_root() -> Path:
    """返回项目根目录下的 skills 目录。"""
    env_root = os.getenv("HUESAE_SKILLS_ROOT")
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parents[4] / "skills"


def _normalize_key(value: str) -> str:
    """统一 Skill 查询键。"""
    return str(value or "").strip().lower()


def _display_path(path: Path) -> str:
    """尽量使用简短路径展示给模型。"""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)
