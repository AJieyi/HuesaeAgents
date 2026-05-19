"""共享工具运行时测试。"""

from langchain.tools import tool

from huesaeagents.huesae.mcp import cache
from huesaeagents.huesae.subagents.registry import SubAgentRegistry
from huesaeagents.huesae.tools.runtime import build_shared_runtime


@tool
def fake_mcp_video_tool(video_path: str) -> str:
    """读取视频信息的假 MCP 工具。"""
    return f"视频路径：{video_path}"


@tool
def fake_douyin_download_tool(share_link: str) -> str:
    """获取抖音视频下载链接的假 MCP 工具。"""
    return f"下载链接：{share_link}"


@tool
def fake_bilibili_info_tool(url: str) -> str:
    """获取B站视频信息的假 MCP 工具。"""
    return f"B站视频信息：{url}"


@tool
def fake_fysh_bilibili_parse_tool(url: str) -> str:
    """解析B站视频链接的假 MCP 工具。"""
    return f"fysh B站解析结果：{url}"


def test_mcp_cache_reuses_and_refreshes(monkeypatch, tmp_path):
    """MCP cache 首次加载、复用缓存、mtime 变化刷新。"""
    config_path = tmp_path / "extensions_config.json"
    config_path.write_text('{"mcpServers": {}}', encoding="utf-8")
    calls = []

    async def fake_get_mcp_tools(config_path_arg=None):
        calls.append(config_path_arg)
        return [fake_mcp_video_tool]

    monkeypatch.setattr(cache, "get_mcp_tools", fake_get_mcp_tools)
    cache.reset_mcp_tools_cache()

    first = cache.get_cached_mcp_tools(str(config_path))
    second = cache.get_cached_mcp_tools(str(config_path))
    config_path.write_text('{"mcpServers": {"video": {"enabled": false}}}', encoding="utf-8")
    third = cache.get_cached_mcp_tools(str(config_path))

    assert first == [fake_mcp_video_tool]
    assert second is first
    assert third == [fake_mcp_video_tool]
    assert len(calls) == 2


def test_mcp_cache_failure_returns_empty(monkeypatch, tmp_path):
    """MCP 加载异常时返回空列表并记录错误。"""
    config_path = tmp_path / "extensions_config.json"
    config_path.write_text('{"mcpServers": {}}', encoding="utf-8")

    async def fake_get_mcp_tools(config_path_arg=None):
        raise RuntimeError("mcp down")

    monkeypatch.setattr(cache, "get_mcp_tools", fake_get_mcp_tools)
    cache.reset_mcp_tools_cache()

    tools = cache.get_cached_mcp_tools(str(config_path))

    assert tools == []
    assert cache.get_last_mcp_error() == "mcp down"


def test_shared_runtime_merges_builtin_and_mcp(llm):
    """共享运行时合并内置工具与 MCP 工具，并支持子Agent过滤 task_tool。"""
    registry = SubAgentRegistry()
    runtime = build_shared_runtime(
        llm,
        registry,
        mcp_tools_loader=lambda *args, **kwargs: [fake_mcp_video_tool],
    )

    main_names = {tool.name for tool in runtime.get_tools(include_mcp=True, include_task_tool=True)}
    child_tools = runtime.get_tools(include_mcp=True, include_task_tool=False)
    child_names = {tool.name for tool in child_tools}

    assert "generate_image_tool" in main_names
    assert "task_tool" in main_names
    assert "fake_mcp_video_tool" in main_names
    assert "fake_mcp_video_tool" in child_names
    assert "task_tool" not in child_names
    assert runtime.get_tool_map()["fake_mcp_video_tool"] is fake_mcp_video_tool


def test_shared_runtime_merges_multiple_mcp_servers(llm):
    """共享运行时应能合并多个 MCP server 暴露的工具。"""
    runtime = build_shared_runtime(
        llm,
        SubAgentRegistry(),
        mcp_tools_loader=lambda *args, **kwargs: [
            fake_mcp_video_tool,
            fake_douyin_download_tool,
        ],
    )

    tool_names = {tool.name for tool in runtime.get_tools(include_mcp=True)}

    assert "fake_mcp_video_tool" in tool_names
    assert "fake_douyin_download_tool" in tool_names
    assert "task_tool" in tool_names


def test_shared_runtime_merges_three_mcp_servers(llm):
    """共享运行时应能合并 video、douyin、bilibili 三个 MCP server 工具。"""
    runtime = build_shared_runtime(
        llm,
        SubAgentRegistry(),
        mcp_tools_loader=lambda *args, **kwargs: [
            fake_mcp_video_tool,
            fake_douyin_download_tool,
            fake_bilibili_info_tool,
        ],
    )

    tool_names = {tool.name for tool in runtime.get_tools(include_mcp=True)}

    assert "fake_mcp_video_tool" in tool_names
    assert "fake_douyin_download_tool" in tool_names
    assert "fake_bilibili_info_tool" in tool_names
    assert "task_tool" in tool_names


def test_shared_runtime_merges_fysh_bilibili_mcp(llm):
    """共享运行时应能合并 fysh1010/bilibili-mcp 暴露的工具。"""
    runtime = build_shared_runtime(
        llm,
        SubAgentRegistry(),
        mcp_tools_loader=lambda *args, **kwargs: [
            fake_mcp_video_tool,
            fake_douyin_download_tool,
            fake_bilibili_info_tool,
            fake_fysh_bilibili_parse_tool,
        ],
    )

    tool_names = {tool.name for tool in runtime.get_tools(include_mcp=True)}

    assert "fake_fysh_bilibili_parse_tool" in tool_names
    assert "fake_bilibili_info_tool" in tool_names
    assert "task_tool" in tool_names


def test_shared_runtime_does_not_load_mcp_until_requested(llm):
    """共享运行时在 include_mcp=False 时不触发 MCP discovery。"""
    calls = []
    runtime = build_shared_runtime(
        llm,
        SubAgentRegistry(),
        mcp_tools_loader=lambda *args, **kwargs: calls.append(kwargs) or [fake_mcp_video_tool],
    )

    builtin_names = {tool.name for tool in runtime.get_tools(include_mcp=False)}

    assert calls == []
    assert runtime.mcp_loaded is False
    assert "load_mcp_tools_tool" in builtin_names
    assert "fake_mcp_video_tool" not in builtin_names

    mcp_names = {tool.name for tool in runtime.get_tools(include_mcp=True)}

    assert len(calls) == 1
    assert runtime.mcp_loaded is True
    assert "fake_mcp_video_tool" in mcp_names
