"""Skill 系统入口。

Skill 是注入到 Agent 提示词中的知识与工作指令，不等同于函数工具。
"""

from .loader import load_skills
from .registry import SkillRegistry
from .types import Skill

__all__ = [
    "Skill",
    "SkillRegistry",
    "load_skills",
]
