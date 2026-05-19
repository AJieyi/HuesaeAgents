"""手动验证 video-capture-script-mcp。

运行方式：
    conda activate HuesaeAgents
    python backend/huesaeagents/huesae/mcp/test/manual_video_mcp_smoke.py

说明：
    - 会读取项目根目录 `.env` 中的 TENCENT_SECRET_ID、TENCENT_SECRET_KEY、TENCENT_REGION。
    - 会通过 extensions_config.json 启动线上 npm stdio MCP server。
    - 默认优先调用视频信息工具；如果工具名变化，会打印已发现工具，方便手动调整。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def _prepare_import_path() -> Path:
    """把 backend 目录加入 sys.path，支持直接运行本文件。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    backend_dir = Path(__file__).resolve().parents[4]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    return backend_dir


def _mask_env_status() -> None:
    """打印环境变量读取状态，不泄露密钥。"""
    for key in ("TENCENT_SECRET_ID", "TENCENT_SECRET_KEY", "TENCENT_REGION", "FFMPEG_PATH", "FFPROBE_PATH"):
        value = os.getenv(key, "")
        if key in ("TENCENT_REGION", "FFMPEG_PATH", "FFPROBE_PATH"):
            status = value or "未设置"
        else:
            status = "已设置" if value else "未设置"
        print(f"{key}: {status}")


def _prepend_path(path: Path) -> None:
    """把目录补到当前进程 PATH 前面。"""
    current_path = os.environ.get("PATH", "")
    path_text = str(path)
    if path_text.lower() not in [part.lower() for part in current_path.split(os.pathsep) if part]:
        os.environ["PATH"] = path_text + os.pathsep + current_path


def _normalize_ffmpeg_env() -> None:
    """根据 .env 中的 FFmpeg 配置修正当前进程 PATH。"""
    ffmpeg_path = os.getenv("FFMPEG_PATH", "").strip().strip('"')
    ffprobe_path = os.getenv("FFPROBE_PATH", "").strip().strip('"')

    for value in (ffmpeg_path, ffprobe_path):
        if not value:
            continue
        path = Path(value)
        directory = path if path.is_dir() else path.parent
        if directory.exists():
            _prepend_path(directory)

    if ffmpeg_path and not ffprobe_path:
        path = Path(ffmpeg_path)
        directory = path if path.is_dir() else path.parent
        candidate = directory / "ffprobe.exe"
        if candidate.exists():
            os.environ["FFPROBE_PATH"] = str(candidate)
            _prepend_path(directory)

    if ffprobe_path and not ffmpeg_path:
        path = Path(ffprobe_path)
        directory = path if path.is_dir() else path.parent
        candidate = directory / "ffmpeg.exe"
        if candidate.exists():
            os.environ["FFMPEG_PATH"] = str(candidate)
            _prepend_path(directory)


def _print_ffmpeg_status() -> None:
    """打印当前进程能否找到 ffmpeg/ffprobe。"""
    print(f"PATH中ffmpeg: {shutil.which('ffmpeg') or '未找到'}")
    print(f"PATH中ffprobe: {shutil.which('ffprobe') or '未找到'}")


def _find_tool(tools: list[Any], candidates: tuple[str, ...]):
    """按名称片段查找工具。"""
    lowered_candidates = tuple(candidate.lower() for candidate in candidates)
    for tool in tools:
        name = getattr(tool, "name", "").lower()
        if any(candidate in name for candidate in lowered_candidates):
            return tool
    return None


def _invoke_tool(tool, video_path: Path):
    """尽量兼容不同 MCP 工具的参数名。"""
    schema = getattr(tool, "args_schema", None)
    properties = {}
    if schema is not None and hasattr(schema, "model_json_schema"):
        properties = schema.model_json_schema().get("properties") or {}

    args: dict[str, Any] = {}
    for candidate in ("videoPath", "video_path", "filePath", "file_path", "path", "input"):
        if candidate in properties:
            args[candidate] = str(video_path)
            break
    if not args:
        args["videoPath"] = str(video_path)

    print(f"\n调用工具：{tool.name}")
    print(f"参数：{json.dumps(args, ensure_ascii=False)}")
    return tool.invoke(args)


def main() -> int:
    """发现 MCP 工具并调用测试视频。"""
    _prepare_import_path()

    from huesaeagents.huesae.config.extensions_config import ExtensionsConfig, load_project_env
    from huesaeagents.huesae.mcp.cache import initialize_mcp_tools, reset_mcp_tools_cache

    load_project_env()
    _normalize_ffmpeg_env()
    _mask_env_status()
    _print_ffmpeg_status()

    video_path = Path(__file__).resolve().parent / "mvp_test_video.mp4"
    # video_path = "https://www.bilibili.com/video/BV1kf9UBCEtv/?share_source=copy_web&vd_source=b0499d8e7099a7ec35cdb0381235aa7b"
    if not video_path.exists():
        print(f"测试视频不存在：{video_path}")
        return 1

    config = ExtensionsConfig.from_file()
    print(f"\n扩展配置：{config.source_path or '未找到'}")
    print(f"启用 MCP server：{list(config.get_enabled_mcp_servers())}")

    reset_mcp_tools_cache()
    tools = initialize_mcp_tools(force=True)
    print(f"\n发现 MCP 工具数：{len(tools)}")
    for tool in tools:
        print(f"- {tool.name}: {getattr(tool, 'description', '')}")

    if not tools:
        print("\n没有发现 MCP 工具，请检查 npm、网络、extensions_config.json。")
        return 1

    tool = _find_tool(tools, ("get_video_info", "video_info", "info"))
    if tool is None:
        tool = _find_tool(tools, ("analyze_video_content", "analyze", "video"))
    if tool is None:
        print("\n未找到视频信息或视频分析工具，请根据上方工具名手动调整候选名称。")
        return 1

    result = _invoke_tool(tool, video_path)
    print("\n工具返回：")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
