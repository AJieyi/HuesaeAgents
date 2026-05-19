"""扩展配置测试。"""

import json

from huesaeagents.huesae.config.extensions_config import ExtensionsConfig
from huesaeagents.huesae.mcp.client import build_server_params, build_servers_config


def test_missing_config_returns_empty(monkeypatch, tmp_path):
    """配置不存在时返回空配置，不阻断启动。"""
    missing_path = tmp_path / "missing_extensions_config.json"
    monkeypatch.setenv("HUESAE_EXTENSIONS_CONFIG_PATH", str(missing_path))

    config = ExtensionsConfig.from_file(str(tmp_path / "missing.json"))

    assert config.get_enabled_mcp_servers() == {}


def test_env_variables_are_expanded(monkeypatch, tmp_path):
    """支持 `$VAR` 与 `${VAR}` 两种环境变量写法。"""
    config_path = tmp_path / "extensions_config.json"
    config_path.write_text(
        json.dumps({
            "mcpServers": {
                "video": {
                    "command": "npx",
                    "args": ["-y", "${MCP_PACKAGE}"],
                    "env": {
                        "TENCENT_SECRET_ID": "$TENCENT_SECRET_ID",
                        "TENCENT_REGION": "${TENCENT_REGION}",
                    },
                }
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("MCP_PACKAGE", "@pickstar-2002/video-mcp@latest")
    monkeypatch.setenv("TENCENT_SECRET_ID", "sid")
    monkeypatch.setenv("TENCENT_REGION", "ap-beijing")

    config = ExtensionsConfig.from_file(str(config_path))
    server = config.mcp_servers["video"]

    assert server.args == ["-y", "@pickstar-2002/video-mcp@latest"]
    assert server.env["TENCENT_SECRET_ID"] == "sid"
    assert server.env["TENCENT_REGION"] == "ap-beijing"


def test_video_mcp_stdio_config_builds_client_params(tmp_path):
    """视频 MCP stdio 配置能转换成 MultiServerMCPClient 参数。"""
    config_path = tmp_path / "extensions_config.json"
    config_path.write_text(
        json.dumps({
            "mcpServers": {
                "video-capture-script-mcp": {
                    "enabled": True,
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "@pickstar-2002/video-mcp@latest"],
                    "env": {"TENCENT_REGION": "ap-beijing"},
                }
            }
        }),
        encoding="utf-8",
    )

    config = ExtensionsConfig.from_file(str(config_path))
    params = build_server_params(
        "video-capture-script-mcp",
        config.mcp_servers["video-capture-script-mcp"],
    )
    servers_config = build_servers_config(config)

    assert params["transport"] == "stdio"
    assert params["command"] == "npx"
    assert params["args"] == ["-y", "@pickstar-2002/video-mcp@latest"]
    assert servers_config["video-capture-script-mcp"] == params


def test_douyin_mcp_stdio_config_builds_client_params(monkeypatch, tmp_path):
    """抖音 MCP stdio 配置能转换成 uvx 启动参数，并映射 API_KEY。"""
    config_path = tmp_path / "extensions_config.json"
    config_path.write_text(
        json.dumps({
            "mcpServers": {
                "video-capture-script-mcp": {
                    "enabled": True,
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "@pickstar-2002/video-mcp@latest"],
                },
                "douyin-mcp-server": {
                    "enabled": True,
                    "type": "stdio",
                    "command": "uvx",
                    "args": ["douyin-mcp-server"],
                    "env": {"API_KEY": "${DOUYIN_MCP_API_KEY}"},
                },
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("DOUYIN_MCP_API_KEY", "douyin-key")

    config = ExtensionsConfig.from_file(str(config_path))
    servers_config = build_servers_config(config)
    douyin_params = servers_config["douyin-mcp-server"]

    assert set(servers_config) == {"video-capture-script-mcp", "douyin-mcp-server"}
    assert douyin_params["transport"] == "stdio"
    assert douyin_params["command"] == "uvx"
    assert douyin_params["args"] == ["douyin-mcp-server"]
    assert douyin_params["env"]["API_KEY"] == "douyin-key"


def test_douyin_mcp_api_key_can_fallback_to_dashscope(monkeypatch, tmp_path):
    """用户只配置 DASHSCOPE_API_KEY 时，抖音 MCP 也能拿到 API_KEY。"""
    monkeypatch.delenv("DOUYIN_MCP_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")
    config_path = tmp_path / "extensions_config.json"
    config_path.write_text(
        json.dumps({
            "mcpServers": {
                "douyin-mcp-server": {
                    "enabled": True,
                    "type": "stdio",
                    "command": "uvx",
                    "args": ["douyin-mcp-server"],
                    "env": {"API_KEY": "${DOUYIN_MCP_API_KEY}"},
                }
            }
        }),
        encoding="utf-8",
    )

    config = ExtensionsConfig.from_file(str(config_path))

    assert config.mcp_servers["douyin-mcp-server"].env["API_KEY"] == "dashscope-key"


def test_bilibili_mcp_stdio_config_builds_client_params(tmp_path):
    """B站 MCP stdio 配置能转换成 uvx 启动参数。"""
    config_path = tmp_path / "extensions_config.json"
    config_path.write_text(
        json.dumps({
            "mcpServers": {
                "bilibili-video-download-mcp": {
                    "enabled": True,
                    "type": "stdio",
                    "command": "uvx",
                    "args": ["bilibili-video-download-mcp"],
                    "env": {
                        "FFMPEG_PATH": "/usr/bin/ffmpeg",
                        "FFPROBE_PATH": "/usr/bin/ffprobe",
                    },
                }
            }
        }),
        encoding="utf-8",
    )

    config = ExtensionsConfig.from_file(str(config_path))
    servers_config = build_servers_config(config)
    bili_params = servers_config["bilibili-video-download-mcp"]

    assert bili_params["transport"] == "stdio"
    assert bili_params["command"] == "uvx"
    assert bili_params["args"] == ["bilibili-video-download-mcp"]
    assert bili_params["env"]["FFMPEG_PATH"] == "/usr/bin/ffmpeg"
    assert bili_params["env"]["FFPROBE_PATH"] == "/usr/bin/ffprobe"


def test_fysh_bilibili_mcp_stdio_config_builds_client_params(tmp_path):
    """fysh1010/bilibili-mcp 配置能转换成 npx 启动参数。"""
    config_path = tmp_path / "extensions_config.json"
    config_path.write_text(
        json.dumps({
            "mcpServers": {
                "bilibili-mcp": {
                    "enabled": True,
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "mcp-server-bilibili"],
                }
            }
        }),
        encoding="utf-8",
    )

    config = ExtensionsConfig.from_file(str(config_path))
    servers_config = build_servers_config(config)
    bili_params = servers_config["bilibili-mcp"]

    assert bili_params["transport"] == "stdio"
    assert bili_params["command"] == "npx"
    assert bili_params["args"] == ["-y", "mcp-server-bilibili"]
