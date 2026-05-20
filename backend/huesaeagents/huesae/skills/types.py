"""Skill 数据模型。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    """单个 Skill 的元数据与完整指令内容。"""

    name: str
    description: str
    slug: str
    skill_dir: Path
    skill_file: Path
    content: str
    enabled: bool = True

    @property
    def prompt_name(self) -> str:
        """返回适合注入提示词的名称。"""
        if self.name == self.slug:
            return self.name
        return f"{self.name}（别名：{self.slug}）"
