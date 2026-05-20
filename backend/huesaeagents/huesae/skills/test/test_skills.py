"""Skill 系统测试。"""

from pathlib import Path

from huesaeagents.huesae.skills import SkillRegistry, load_skills


def _write_skill(root: Path, slug: str, skill_md: str, meta_json: str = "{}") -> Path:
    """写入测试用 Skill。"""
    skill_dir = root / slug
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    (skill_dir / "_meta.json").write_text(meta_json, encoding="utf-8")
    return skill_dir


def test_load_skills_reads_frontmatter_and_meta(tmp_path):
    """加载器应读取 frontmatter、slug 和完整内容。"""
    _write_skill(
        tmp_path,
        "weather",
        """---
name: weather
description: Get current weather.
---

# Weather
Use curl.
""",
        '{"slug": "weather"}',
    )

    skills = load_skills(tmp_path)

    assert len(skills) == 1
    assert skills[0].name == "weather"
    assert skills[0].slug == "weather"
    assert "Use curl" in skills[0].content


def test_load_skills_supports_multiline_description(tmp_path):
    """加载器应支持 frontmatter 中的多行 description。"""
    _write_skill(
        tmp_path,
        "polecomic",
        """---
name: manga-animation
description: >
  完整的漫剧生成工作流 skill。
  每个阶段都需要用户确认。
---

# 漫剧生成 Skill
""",
        '{"slug": "polecomic"}',
    )

    skill = load_skills(tmp_path)[0]

    assert skill.name == "manga-animation"
    assert skill.slug == "polecomic"
    assert "漫剧生成工作流" in skill.description
    assert "用户确认" in skill.description


def test_skill_registry_formats_prompt_and_reads_content(tmp_path):
    """注册表应格式化 Skill 列表，并能按名称或 slug 读取全文。"""
    _write_skill(
        tmp_path,
        "polecomic",
        """---
name: manga-animation
description: 漫剧生成工作流。
---

# 指令正文
""",
        '{"slug": "polecomic"}',
    )

    registry = SkillRegistry(tmp_path)
    prompt = registry.format_for_prompt()

    assert "manga-animation" in prompt
    assert "polecomic" in prompt
    assert "read_skill_tool" in prompt
    assert "# 指令正文" in registry.get_content("manga-animation")
    assert "# 指令正文" in registry.get_content("polecomic")


def test_skill_registry_ignores_disabled_skill(tmp_path):
    """禁用的 Skill 不应注入提示词。"""
    _write_skill(
        tmp_path,
        "disabled",
        """---
name: disabled
description: disabled skill
---
""",
        '{"slug": "disabled", "enabled": false}',
    )

    registry = SkillRegistry(tmp_path)

    assert registry.list_enabled() == []
    assert "暂无可用 Skills" in registry.format_for_prompt()
