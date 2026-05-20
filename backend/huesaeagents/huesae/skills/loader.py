"""Skill 发现与加载。"""

from __future__ import annotations

import json
from pathlib import Path

from .types import Skill


def load_skills(skills_root: Path) -> list[Skill]:
    """扫描 skills 根目录，加载所有启用的 SKILL.md。"""
    root = Path(skills_root)
    if not root.exists() or not root.is_dir():
        return []

    skills: list[Skill] = []
    for skill_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue

        content = skill_file.read_text(encoding="utf-8")
        frontmatter = _parse_frontmatter(content)
        meta = _read_meta(skill_dir / "_meta.json")
        enabled = bool(meta.get("enabled", True))
        if not enabled:
            continue

        slug = str(meta.get("slug") or skill_dir.name).strip() or skill_dir.name
        name = str(frontmatter.get("name") or slug).strip() or slug
        description = str(frontmatter.get("description") or meta.get("description") or "").strip()
        if not description:
            description = f"{name} Skill 指令"

        skills.append(Skill(
            name=name,
            description=description,
            slug=slug,
            skill_dir=skill_dir,
            skill_file=skill_file,
            content=content,
            enabled=enabled,
        ))
    return skills


def _read_meta(meta_file: Path) -> dict:
    """读取 _meta.json；文件不存在或格式错误时返回空字典。"""
    if not meta_file.exists():
        return {}
    try:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _parse_frontmatter(content: str) -> dict[str, str]:
    """解析 SKILL.md 开头的简单 YAML frontmatter。"""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
    if end_index is None:
        return {}

    return _parse_simple_yaml(lines[1:end_index])


def _parse_simple_yaml(lines: list[str]) -> dict[str, str]:
    """解析本项目 Skill 元信息需要的 YAML 子集。"""
    values: dict[str, str] = {}
    current_key: str | None = None
    block_lines: list[str] = []

    def flush_block() -> None:
        nonlocal current_key, block_lines
        if current_key is not None:
            values[current_key] = " ".join(line.strip() for line in block_lines).strip()
        current_key = None
        block_lines = []

    for raw_line in lines:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        if current_key is not None and raw_line.startswith((" ", "\t")):
            block_lines.append(raw_line)
            continue

        flush_block()

        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value in (">", "|"):
            current_key = key
            block_lines = []
            continue
        values[key] = _strip_quotes(value)

    flush_block()
    return values


def _strip_quotes(value: str) -> str:
    """去掉最外层的简单引号。"""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
