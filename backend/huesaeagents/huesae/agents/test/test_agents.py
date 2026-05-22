"""Agent 行为测试。

测试范围：
1. 主Agent工具注册和子Agent委派
2. 生图子Agent的标准化接口
3. 标签生成、提示词扩写、Provider 注册等独立模块

本文件使用本地假模型，不访问真实 LLM 或生图 API。
"""
import sys
from pathlib import Path

import pytest
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool

backend_dir = Path(__file__).resolve().parents[4]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from huesaeagents.huesae.agents.lead_agent import HuesaeMainAgent
from huesaeagents.huesae.agents.middlewares import (
    AgentMiddleware,
    MiddlewarePipeline,
    TokenUsageMiddleware,
    build_middlewares,
)
from huesaeagents.huesae.config import (
    MiddlewareConfig,
    TokenUsageConfig,
    reset_middleware_config,
    set_middleware_config,
)
from huesaeagents.huesae.skills import SkillRegistry
from huesaeagents.huesae.services.memory import HonchoMemoryService
from huesaeagents.huesae.subagents.image_agent import (
    ImageDecision,
    ImageSubAgent,
    ImageUserIntent,
    create_image_agent,
)
from huesaeagents.huesae.subagents.image import (
    DoubaoProvider,
    expand_prompt,
    generate_tags,
)
VIDEO_PATH = "F:/videos/demo.mp4"
IMAGE_PATH_1 = "F:/images/a.png"
IMAGE_PATH_2 = "F:/images/b.png"
IMAGE_PATH_3 = "F:/images/c.png"
DOUYIN_URL = "https://www.douyin.com/video/1234567890"
BILIBILI_URL = "https://www.bilibili.com/video/BV1xx411c7mu"


class FakeStructuredLLM:
    """按 Pydantic Schema 返回固定结构化结果的假模型。"""

    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        user_input, original_user_input = self._latest_user_context(messages)

        if self.schema is ImageDecision:
            return self._decide_image_action(user_input)

        if self.schema is ImageUserIntent:
            return self._decide_image_user_intent(user_input)

        raise AssertionError(f"测试假模型不支持的结构化输出：{self.schema}")

    @staticmethod
    def _latest_user_context(messages) -> tuple[str, str]:
        tool_result = ""
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                content = str(message.content)
                if content.startswith("工具执行结果："):
                    tool_result = content

        user_text = ""
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                content = str(message.content)
                for line in content.splitlines():
                    if line.startswith("- 用户最新输入："):
                        user_text = line.removeprefix("- 用户最新输入：").strip()
                        break
                    if line.startswith("用户最新输入："):
                        user_text = line.removeprefix("用户最新输入：").strip()
                        break
                if not user_text:
                    user_text = content
                break

        if tool_result:
            return tool_result, user_text
        return user_text, user_text

    @staticmethod
    def _decide_image_action(user_input: str) -> ImageDecision:
        if user_input == "你帮我推荐一些吧":
            return ImageDecision(
                thought="用户需要推荐图片主题。",
                action="recommend",
                response="可以试试：樱花树下的少女、雨夜霓虹街道、星空下的魔法书。",
                prompt=None,
                provider="doubao",
            )

        if user_input in ("你帮我扩展一下吧", "扩写一下我的提示词", "帮我扩写提示词"):
            return ImageDecision(
                thought="用户要求扩写提示词。",
                action="expand",
                response="我来帮你扩写一下~",
                prompt="夏天的图",
                provider="doubao",
            )

        if user_input in ("就这个描述", "直接生图"):
            return ImageDecision(
                thought="用户确认当前描述，可以开始生图。",
                action="generate",
                response="图片正在生成中，请稍等~",
                prompt=None,
                provider="doubao",
                size="2K",
                output_format="jpeg",
                is_batch=False,
            )

        if user_input in ("真好看，谢谢", "可以了，谢谢"):
            return ImageDecision(
                thought="用户表示满意，结束子Agent对话。",
                action="finish",
                response="喜欢就好，下次还可以继续来找我画图~",
            )

        if user_input in ("我想生成图片", "我要生图"):
            return ImageDecision(
                thought="缺少图片描述，需要追问。",
                action="ask_prompt",
                response="请告诉我您想要生成什么样的图片？",
                prompt=None,
                provider="doubao",
            )

        return ImageDecision(
            thought="用户已经给出可生成的描述。",
            action="generate",
            response="图片正在生成中，请稍等~",
            prompt=user_input,
            provider="doubao",
            size="2K",
            output_format="jpeg",
            is_batch=False,
        )

    @staticmethod
    def _decide_image_user_intent(user_input: str) -> ImageUserIntent:
        intent_map = {
            "可以": ImageUserIntent(
                thought="用户确认当前阶段。",
                intent="confirm",
                confidence=0.95,
            ),
            "不可以": ImageUserIntent(
                thought="用户否定当前结果。",
                intent="reject",
                confidence=0.95,
            ),
            "换一张": ImageUserIntent(
                thought="用户希望保持当前提示词重新生成。",
                intent="regenerate",
                confidence=0.95,
            ),
            "扩写一下我的提示词": ImageUserIntent(
                thought="用户希望扩写当前提示词。",
                intent="expand_prompt",
                confidence=0.95,
            ),
            "我重新输入一组提示词：雨夜霓虹街道": ImageUserIntent(
                thought="用户提供新的生图提示词。",
                intent="replace_prompt",
                replacement_prompt="雨夜霓虹街道",
                confidence=0.95,
            ),
            "二次元风格，生成一张在草坪躺着的jk少女": ImageUserIntent(
                thought="用户在图片确认阶段提供了新的生图描述。",
                intent="provide_prompt",
                replacement_prompt="二次元风格，生成一张在草坪躺着的jk少女",
                confidence=0.95,
            ),
            "我要生图": ImageUserIntent(
                thought="用户表达生图意愿，但没有给出具体画面描述。",
                intent="clarify",
                confidence=0.4,
                clarification_question="我还没完全理解您的生图需求，可以再具体描述一下您想生成的画面吗？",
            ),
            "就这个描述": ImageUserIntent(
                thought="保守分类器暂时不确定用户是确认还是修改。",
                intent="clarify",
                confidence=0.4,
                clarification_question="您是想确认这个描述开始生图，还是想继续修改或扩写提示词呢？",
            ),
            "帮我扩写提示词": ImageUserIntent(
                thought="保守分类器暂时不确定用户要扩写还是继续修改。",
                intent="clarify",
                confidence=0.4,
                clarification_question="您是想确认这个描述开始生图，还是想继续修改或扩写提示词呢？",
            ),
            "可以了，谢谢": ImageUserIntent(
                thought="保守分类器暂时不确定用户是在确认图片还是普通致谢。",
                intent="clarify",
                confidence=0.4,
                clarification_question="您是满意这张图想结束任务，还是想继续调整呢？",
            ),
        }
        return intent_map.get(
            user_input,
            ImageUserIntent(
                thought="普通生图输入，不处于确认阶段时由 ImageDecision 处理。",
                intent="other",
                confidence=0.8,
            ),
        )


class FakeToolCallingLLM:
    """按当前绑定工具返回 tool_calls 的假模型。"""

    def __init__(self, tools):
        self.tools = tools
        self.tool_names = {tool.name for tool in tools}

    def invoke(self, messages):
        user_input = self._latest_user_text(messages)
        latest_tool_result = self._latest_tool_result(messages)
        if latest_tool_result is not None:
            return AIMessage(content=latest_tool_result)

        if user_input == "北京今天天气怎么样":
            if "read_skill_tool" in self.tool_names:
                return self._tool_call("read_skill_tool", {"skill_name": "weather"})
            return AIMessage(content="我需要先读取天气 Skill。")

        if user_input == "加载视频MCP":
            if "load_mcp_tools_tool" in self.tool_names:
                return self._tool_call("load_mcp_tools_tool", {})
            return AIMessage(content="MCP扩展工具已加载，可以继续处理视频任务。")

        if user_input == "MCP未加载时直接猜测视频信息工具":
            if "video-capture-script-mcp_get_video_info" not in self.tool_names:
                return self._tool_call("video-capture-script-mcp_get_video_info", {})
            return self._tool_call(
                "video-capture-script-mcp_get_video_info",
                {"videoPath": VIDEO_PATH},
            )

        if user_input == f"这是本地视频{VIDEO_PATH}，请帮分析视频内容":
            return self._mcp_or_load(
                "video-capture-script-mcp_analyze_video_content",
                {"videoPath": VIDEO_PATH},
            )

        if user_input == f"这张图片{IMAGE_PATH_1}，反推提示词":
            return self._tool_call(
                "reverse_image_prompt",
                {"image_path": IMAGE_PATH_1},
            )

        if user_input == f"换一版这张图的提示词：{IMAGE_PATH_1}":
            return self._tool_call(
                "reverse_image_prompt",
                {
                    "image_path": IMAGE_PATH_1,
                    "style": "alternative",
                    "previous_prompt": "一位银发少女站在樱花树下",
                },
            )

        if user_input == f"这是本地图片{IMAGE_PATH_1}，请帮我基于图片分析图片内容":
            return self._mcp_or_load(
                "video-capture-script-mcp_analyze_image_batch",
                {"imagePaths": [IMAGE_PATH_1]},
            )

        if user_input == f"这是本地图片1{IMAGE_PATH_1}，本地图片2{IMAGE_PATH_2}，本地图片3{IMAGE_PATH_3}，请帮我基于图片分析图片内容":
            return self._mcp_or_load(
                "video-capture-script-mcp_analyze_image_batch",
                {"imagePaths": [IMAGE_PATH_1, IMAGE_PATH_2, IMAGE_PATH_3]},
            )

        if user_input == f"这是本地图片1{IMAGE_PATH_1}，本地图片2{IMAGE_PATH_2}，本地图片3{IMAGE_PATH_3}，请帮我基于图片生成专业拍摄脚本":
            return self._mcp_or_load(
                "video-capture-script-mcp_generate_image_script",
                {"imagePaths": [IMAGE_PATH_1, IMAGE_PATH_2, IMAGE_PATH_3]},
            )

        if user_input == f"这是本地视频{VIDEO_PATH}，请帮我基于视频内容生成专业拍摄脚本":
            return self._mcp_or_load(
                "video-capture-script-mcp_generate_video_script",
                {"videoPath": VIDEO_PATH},
            )

        if user_input == f"这是视频地址{VIDEO_PATH}，获取视频文件基本信息":
            return self._mcp_or_load(
                "video-capture-script-mcp_get_video_info",
                {"videoPath": VIDEO_PATH},
            )

        if user_input == f"{DOUYIN_URL}，下载视频":
            return self._mcp_or_load(
                "douyin-mcp-server_get_douyin_download_link",
                {"share_link": DOUYIN_URL},
            )

        if user_input == f"获取这个B站视频的信息：{BILIBILI_URL}":
            return self._mcp_or_load(
                "bilibili-video-download-mcp_get_video_info",
                {"url": BILIBILI_URL},
            )

        if user_input == f"下载这个B站视频：{BILIBILI_URL}":
            return self._mcp_or_load(
                "bilibili-video-download-mcp_download_video",
                {"url": BILIBILI_URL, "output_dir": "./downloads", "merge": True},
            )

        if user_input == f"解析这个B站视频：{BILIBILI_URL}":
            return self._mcp_or_load(
                "bilibili-mcp_parse_bilibili_video",
                {"url": BILIBILI_URL},
            )

        if user_input.startswith("读取视频信息"):
            return self._tool_call(
                "fake_mcp_video_info",
                {"video_path": user_input.removeprefix("读取视频信息").strip()},
            )

        if user_input == "画一只猫":
            return self._tool_call(
                "task_tool",
                {"description": user_input, "subagent_type": "image"},
            )

        if user_input in ("我想生成图片", "我要生图"):
            return self._tool_call(
                "task_tool",
                {"description": user_input, "subagent_type": "image"},
            )

        return AIMessage(content="你好呀，我在这里~")

    def _mcp_or_load(self, tool_name: str, args: dict) -> AIMessage:
        if tool_name in self.tool_names:
            return self._tool_call(tool_name, args)
        return self._tool_call("load_mcp_tools_tool", {})

    @staticmethod
    def _tool_call(name: str, args: dict) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}],
        )

    @staticmethod
    def _latest_user_text(messages) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return str(message.content)
        return ""

    @staticmethod
    def _latest_tool_result(messages) -> str | None:
        for message in reversed(messages):
            if getattr(message, "type", None) == "tool":
                content = str(message.content)
                if content.startswith("MCP扩展工具已加载"):
                    return None
                return content
        return None


class FakeLLM:
    """覆盖测试所需的 LLM 接口，避免真实网络调用。"""

    def with_structured_output(self, schema, method=None):
        return FakeStructuredLLM(schema)

    def bind_tools(self, tools):
        return FakeToolCallingLLM(tools)

    def invoke(self, messages):
        text = "\n".join(str(getattr(message, "content", "")) for message in messages)

        if "Danbooru标签" in text:
            return AIMessage(content="1girl, silver hair, red eyes, cherry blossoms, anime style")

        if "请扩写以下描述" in text:
            return AIMessage(content="夕阳下的战舰停泊在金色海面上，云层被晚霞染亮，画面具有二次元插画质感。")

        if "用户请求的图片已经生成完成了" in text:
            return AIMessage(content="这是生成好的图片哦~")

        return AIMessage(content="你好呀，我在这里~")


@tool("video-capture-script-mcp_analyze_video_content")
def fake_analyze_video_content(videoPath: str) -> str:
    """使用腾讯混元多模态API分析视频内容。"""
    return f"视频内容分析结果：{videoPath}"


@tool("video-capture-script-mcp_analyze_image_batch")
def fake_analyze_image_batch(imagePaths: list[str]) -> str:
    """批量分析图片内容。"""
    return "图片内容分析结果：" + ",".join(imagePaths)


@tool("video-capture-script-mcp_generate_image_script")
def fake_generate_image_script(imagePaths: list[str]) -> str:
    """基于批量图片内容生成专业拍摄脚本。"""
    return "图片脚本生成结果：" + ",".join(imagePaths)


@tool("video-capture-script-mcp_generate_video_script")
def fake_generate_video_script(videoPath: str) -> str:
    """基于视频内容生成专业拍摄脚本。"""
    return f"视频脚本生成结果：{videoPath}"


@tool("video-capture-script-mcp_get_video_info")
def fake_get_video_info(videoPath: str) -> str:
    """获取视频文件基本信息。"""
    return f"视频信息：{videoPath}"


def _fake_video_mcp_tools():
    """模拟 video MCP 暴露的可用工具，不包含抽帧工具。"""
    return [
        fake_analyze_video_content,
        fake_analyze_image_batch,
        fake_generate_image_script,
        fake_generate_video_script,
        fake_get_video_info,
    ]


@tool("douyin-mcp-server_parse_douyin_video_info")
def fake_parse_douyin_video_info(share_link: str) -> str:
    """解析抖音视频分享链接并返回视频基础信息。"""
    return f"抖音视频信息：{share_link}"


@tool("douyin-mcp-server_get_douyin_download_link")
def fake_get_douyin_download_link(share_link: str) -> str:
    """获取抖音视频下载链接。"""
    return f"抖音视频下载链接：https://download.example.com/video.mp4?src={share_link}"


@tool("douyin-mcp-server_extract_douyin_text")
def fake_extract_douyin_text(share_link: str) -> str:
    """提取抖音视频语音文本或文案。"""
    return f"抖音视频文本：{share_link}"


def _fake_douyin_mcp_tools():
    """模拟 douyin MCP 暴露的可用工具。"""
    return [
        fake_parse_douyin_video_info,
        fake_get_douyin_download_link,
        fake_extract_douyin_text,
    ]


def _fake_all_mcp_tools():
    """模拟 video MCP 与 douyin MCP 同时启用。"""
    return _fake_video_mcp_tools() + _fake_douyin_mcp_tools()


@tool("bilibili-video-download-mcp_get_video_info")
def fake_bilibili_get_video_info(url: str) -> str:
    """获取B站视频信息。"""
    return f"B站视频信息：{url}"


@tool("bilibili-video-download-mcp_download_video")
def fake_bilibili_download_video(
    url: str,
    output_dir: str = "./downloads",
    format: str = "mp4",
    cookies_path: str = "",
    merge: bool = True,
) -> str:
    """下载B站视频到指定目录。"""
    return f"B站视频已下载：{url} -> {output_dir}，格式：{format}，合并：{merge}"


def _fake_bilibili_mcp_tools():
    """模拟 bilibili MCP 暴露的可用工具。"""
    return [
        fake_bilibili_get_video_info,
        fake_bilibili_download_video,
    ]


def _fake_all_platform_mcp_tools():
    """模拟 video、douyin、bilibili MCP 同时启用。"""
    return _fake_video_mcp_tools() + _fake_douyin_mcp_tools() + _fake_bilibili_mcp_tools()


@tool("bilibili-mcp_parse_bilibili_video")
def fake_fysh_parse_bilibili_video(url: str) -> str:
    """解析B站视频链接。"""
    return f"fysh B站视频解析结果：{url}"


@tool("bilibili-mcp_get_bilibili_video_info")
def fake_fysh_get_bilibili_video_info(url: str) -> str:
    """获取B站视频详细信息。"""
    return f"fysh B站视频详细信息：{url}"


@tool("bilibili-mcp_get_bilibili_download_urls")
def fake_fysh_get_bilibili_download_urls(url: str) -> str:
    """获取B站视频下载链接。"""
    return f"fysh B站下载链接：https://download.example.com/bili.mp4?src={url}"


def _fake_fysh_bilibili_mcp_tools():
    """模拟 fysh1010/bilibili-mcp 暴露的可用工具。"""
    return [
        fake_fysh_parse_bilibili_video,
        fake_fysh_get_bilibili_video_info,
        fake_fysh_get_bilibili_download_urls,
    ]


def _fake_all_bilibili_mcp_tools():
    """模拟两个 B站 MCP 同时启用。"""
    return _fake_bilibili_mcp_tools() + _fake_fysh_bilibili_mcp_tools()


@pytest.fixture(scope="module")
def llm():
    """共享的本地假模型。"""
    return FakeLLM()


@pytest.fixture(scope="module")
def main_agent(llm):
    """共享的主Agent实例。"""
    agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=lambda *args, **kwargs: [])
    agent.register_sub_agent(create_image_agent(llm=llm, providers=[]))
    return agent


@pytest.fixture(scope="module")
def image_agent(llm):
    """共享的生图子Agent实例。"""
    return ImageSubAgent(llm=llm, providers=[])


class TestMainAgentHarness:
    """测试主Agent的工具注册和子Agent委派能力。"""

    def test_tools_are_available(self, main_agent):
        """主Agent应只暴露当前视图允许调用的工具。"""
        tool_names = {tool.name for tool in main_agent.tools}
        assert "generate_image_tool" not in tool_names
        assert "generate_images_tool" not in tool_names
        assert "reverse_image_prompt" in tool_names
        assert "load_mcp_tools_tool" in tool_names
        assert "read_skill_tool" in tool_names
        assert "bash_tool" in tool_names
        assert "task_tool" in tool_names

    def test_image_subagent_registered(self, main_agent):
        """生图子Agent应注册到子Agent注册表。"""
        assert main_agent.subagent_registry.get("image") is not None

    def test_child_tool_view_has_no_task_tool(self, main_agent):
        """子Agent工具视图不应包含 task_tool，避免子Agent继续委派子Agent。"""
        tools = main_agent._runtime.get_tools(include_mcp=False, include_task_tool=False)
        tool_names = {tool.name for tool in tools}

        assert "task_tool" not in tool_names
        assert "generate_image_tool" in tool_names

    def test_main_prompt_uses_dynamic_tool_sections(self, main_agent):
        """主Agent系统提示词应使用 runtime 注入的工具约束和 MCP 原则。"""
        prompt = main_agent._build_system_prompt().content

        assert "## 工具使用约束" in prompt
        assert "## MCP工具选择原则" in prompt
        assert "当前Agent不可直接调用以下底层工具" in prompt
        assert "LangChain 函数调用" in prompt
        assert "请以 JSON 格式输出" not in prompt
        assert "generate_image_tool" not in {tool.name for tool in main_agent.tools}
        assert "video-capture-script-mcp_analyze_video_content" not in prompt

    def test_main_prompt_injects_skill_list(self, llm, tmp_path):
        """主Agent提示词应注入可用 Skill 列表。"""
        skill_dir = tmp_path / "weather"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: weather
description: Get current weather.
---

# Weather
""",
            encoding="utf-8",
        )

        agent = HuesaeMainAgent(
            llm=llm,
            mcp_tools_loader=lambda *args, **kwargs: [],
            skill_registry=SkillRegistry(tmp_path),
        )
        prompt = agent._build_system_prompt().content

        assert "## 可用 Skills" in prompt
        assert "weather" in prompt
        assert "read_skill_tool" in prompt

    def test_main_agent_reads_skill_before_execution(self, llm, tmp_path):
        """匹配 Skill 的任务应能调用 read_skill_tool 获取完整指令。"""
        skill_dir = tmp_path / "weather"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: weather
description: Get current weather.
---

# Weather
使用 wttr.in 查询天气。
""",
            encoding="utf-8",
        )

        agent = HuesaeMainAgent(
            llm=llm,
            mcp_tools_loader=lambda *args, **kwargs: [],
            skill_registry=SkillRegistry(tmp_path),
        )
        result = agent.process({"messages": []}, "北京今天天气怎么样")

        assert "wttr.in" in result["messages"][0].content


class TestImageSubAgent:
    """测试生图子Agent标准化接口。"""

    def test_ask_prompt_when_no_description(self, image_agent):
        """用户只说想生成图片时，子Agent应追问具体描述。"""
        result = image_agent.process({}, "我想生成图片")

        assert result["action"] == "ask_prompt"
        assert "请告诉我" in result["response"]

    def test_initial_image_request_uses_agent_question_not_generic_clarification(self, image_agent):
        """初始生图请求应使用生图Agent追问，不应被澄清节点覆盖。"""
        result = image_agent.process({}, "我要生图")

        assert result["action"] == "ask_prompt"
        assert "请告诉我您想要生成什么样的图片" in result["response"]

    def test_generate_when_has_prompt(self, image_agent):
        """用户提供明确提示词时，子Agent应先请求确认。"""
        state = {
            "messages": [
                HumanMessage(content="我想生成图片"),
                AIMessage(content="请告诉我您想要生成什么样的图片？"),
            ],
        }
        result = image_agent.process(state, "夕阳下看大海的少女，穿着水手服")

        assert result["action"] == "ask_confirm"
        assert "如果这个描述可以" in result["response"]
        assert result["data"]["state_update"]["image_phase"] == "awaiting_prompt_confirm"

    def test_recommend_when_asked(self, image_agent):
        """用户要求推荐时，子Agent应返回推荐内容。"""
        result = image_agent.process({}, "你帮我推荐一些吧")

        assert result["action"] == "recommend"
        assert "樱花树下" in result["response"]

    def test_expand_when_asked(self, image_agent):
        """用户要求扩写时，子Agent应扩写并进入确认状态。"""
        state = {
            "messages": [HumanMessage(content="夏天的图")],
            "image_prompt": "夏天的图",
        }
        result = image_agent.process(state, "你帮我扩展一下吧")

        assert result["action"] == "ask_confirm"
        assert "扩写后的描述" in result["response"]
        assert result["data"]["expanded_prompt"] == "夕阳下的战舰停泊在金色海面上，云层被晚霞染亮，画面具有二次元插画质感。"

    def test_finish_when_satisfied(self, image_agent):
        """用户满意后，子Agent应返回结束动作。"""
        result = image_agent.process({}, "真好看，谢谢")

        assert result["action"] == "finish"
        assert "喜欢就好" in result["response"]

    def test_confirm_prompt_then_generate(self, image_agent):
        """生图任务中，用户确认提示词后必须进入生图，不能直接结束。"""
        state = {
            "image_task_type": "generate_image",
            "image_phase": "awaiting_prompt_confirm",
            "image_prompt": "夕阳下看大海的少女",
            "confirmed_prompt": "夕阳下看大海的少女",
        }
        result = image_agent.process(state, "可以")

        assert result["action"] == "generate"
        assert result["prompt"] == "夕阳下看大海的少女"
        assert result["data"]["state_update"]["image_phase"] == "awaiting_generation"

    def test_confirm_description_uses_existing_prompt_when_intent_is_uncertain(self, image_agent):
        """意图分类保守时，如果动作决策为生图，应使用已确认描述生图。"""
        state = {
            "image_task_type": "generate_image",
            "image_phase": "awaiting_prompt_confirm",
            "image_prompt": "二次元风格，高中女生打着伞",
            "confirmed_prompt": "二次元风格，高中女生打着伞",
        }
        result = image_agent.process(state, "就这个描述")

        assert result["action"] == "generate"
        assert result["prompt"] == "二次元风格，高中女生打着伞"
        assert result["data"]["state_update"]["image_phase"] == "awaiting_generation"

    def test_expand_prompt_when_intent_is_uncertain(self, image_agent):
        """意图分类保守时，如果动作决策为扩写，应扩写当前描述。"""
        state = {
            "image_task_type": "generate_image",
            "image_phase": "awaiting_prompt_confirm",
            "image_prompt": "二次元风格，高中女生打着伞",
            "confirmed_prompt": "二次元风格，高中女生打着伞",
        }
        result = image_agent.process(state, "帮我扩写提示词")

        assert result["action"] == "ask_confirm"
        assert "扩写后的描述" in result["response"]
        assert result["data"]["state_update"]["image_phase"] == "awaiting_prompt_confirm"

    def test_confirm_image_then_finish(self, image_agent):
        """图片生成后，用户确认图片满意时任务才结束。"""
        state = {
            "image_task_type": "generate_image",
            "image_phase": "awaiting_image_confirm",
            "image_prompt": "夕阳下看大海的少女",
            "last_prompt": "夕阳下看大海的少女",
            "last_image_urls": ["https://example.com/a.jpeg"],
        }
        result = image_agent.process(state, "可以")

        assert result["action"] == "finish"
        assert result["data"]["state_update"]["image_phase"] == "finished"

    def test_negative_confirm_does_not_finish_image_task(self, image_agent):
        """用户否定当前图片时，不能被误判为确认完成。"""
        state = {
            "image_task_type": "generate_image",
            "image_phase": "awaiting_image_confirm",
            "image_prompt": "夕阳下看大海的少女",
            "last_prompt": "夕阳下看大海的少女",
            "last_image_urls": ["https://example.com/a.jpeg"],
        }
        result = image_agent.process(state, "不可以")

        assert result["action"] != "finish"
        assert result["data"]["state_update"]["image_phase"] == "awaiting_image_confirm"

    def test_expand_after_image_does_not_finish_before_generation(self, image_agent):
        """图片确认阶段扩写后，再确认应继续生图，而不是结束任务。"""
        state = {
            "image_task_type": "generate_image",
            "image_phase": "awaiting_image_confirm",
            "image_prompt": "夏天的图",
            "last_prompt": "夏天的图",
            "last_image_urls": ["https://example.com/a.jpeg"],
        }
        expanded = image_agent.process(state, "扩写一下我的提示词")

        assert expanded["action"] == "ask_confirm"
        assert expanded["data"]["state_update"]["image_phase"] == "awaiting_prompt_confirm"

        state.update(expanded["data"]["state_update"])
        generated = image_agent.process(state, "可以")

        assert generated["action"] == "generate"
        assert generated["data"]["state_update"]["image_phase"] == "awaiting_generation"

    def test_regenerate_after_image_keeps_task_active(self, image_agent):
        """图片确认阶段要求换一张时，应继续用上次提示词生图。"""
        state = {
            "image_task_type": "generate_image",
            "image_phase": "awaiting_image_confirm",
            "image_prompt": "星空下的少女",
            "last_prompt": "星空下的少女",
            "last_image_urls": ["https://example.com/a.jpeg"],
        }
        result = image_agent.process(state, "换一张")

        assert result["action"] == "generate"
        assert "星空下的少女" in result["prompt"]

    def test_replace_prompt_after_image_asks_for_prompt_confirmation(self, image_agent):
        """图片确认阶段重新输入提示词时，应先确认新描述，再继续生图。"""
        state = {
            "image_task_type": "generate_image",
            "image_phase": "awaiting_image_confirm",
            "image_prompt": "星空下的少女",
            "last_prompt": "星空下的少女",
            "last_image_urls": ["https://example.com/a.jpeg"],
        }
        result = image_agent.process(state, "我重新输入一组提示词：雨夜霓虹街道")

        assert result["action"] == "ask_confirm"
        assert "雨夜霓虹街道" in result["prompt"]
        assert result["data"]["state_update"]["image_phase"] == "awaiting_prompt_confirm"

    def test_new_prompt_after_image_confirmation_enters_prompt_confirm_flow(self, image_agent):
        """图片确认阶段直接给出新描述时，应把它作为新提示词进入确认闭环。"""
        state = {
            "image_task_type": "generate_image",
            "image_phase": "awaiting_image_confirm",
            "image_prompt": "二次元风格，高中女生打着伞",
            "last_prompt": "二次元风格，高中女生打着伞",
            "last_image_urls": ["https://example.com/a.jpeg"],
        }
        result = image_agent.process(state, "二次元风格，生成一张在草坪躺着的jk少女")

        assert result["action"] == "ask_confirm"
        assert "草坪躺着的jk少女" in result["prompt"]
        assert "请问这个描述可以吗" in result["response"]
        assert result["data"]["state_update"]["image_phase"] == "awaiting_prompt_confirm"

        state.update(result["data"]["state_update"])
        generated = image_agent.process(state, "可以")

        assert generated["action"] == "generate"
        assert "草坪躺着的jk少女" in generated["prompt"]

    def test_decide_structured_output(self, image_agent):
        """LLM 决策结果应符合 ImageDecision 结构。"""
        decision = image_agent._decide({}, "画一个猫娘")

        assert isinstance(decision, ImageDecision)
        assert decision.action == "generate"
        assert decision.response is not None

    def test_standardized_result_format(self, image_agent):
        """子Agent返回结果应符合统一格式。"""
        result = image_agent.process({}, "我想生成图片")

        assert set(result) == {"action", "response", "prompt", "provider", "data"}


class TestMainAgentIntegration:
    """测试主Agent与生图子Agent集成。"""

    def test_chat_directly(self, main_agent):
        """普通聊天应由主Agent直接回复。"""
        result = main_agent.process({"messages": []}, "你好")

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "你好呀" in result["messages"][0].content

    def test_delegate_to_image_agent(self, main_agent):
        """模糊生图需求应委派给生图子Agent追问。"""
        result = main_agent.process({"messages": []}, "我想生成图片")

        assert len(result["messages"]) == 1
        assert "请告诉我" in result["messages"][0].content
        assert result["active_subagent"]["agent_type"] == "image"

    def test_delegate_initial_image_request(self, main_agent):
        """用户说要生图时，应委派生图Agent并追问画面描述。"""
        result = main_agent.process({"messages": []}, "我要生图")

        assert len(result["messages"]) == 1
        assert "请告诉我您想要生成什么样的图片" in result["messages"][0].content
        assert result["active_subagent"]["agent_type"] == "image"

    def test_image_description_is_delegated_to_subagent(self, main_agent):
        """主Agent处理生图描述时，应通过 task_tool 转入子Agent确认流程。"""
        result = main_agent.process({"messages": []}, "画一只猫")

        assert result.get("pending_generation") is None
        assert result["active_subagent"]["agent_type"] == "image"
        assert result["active_subagent"]["state"]["image_phase"] == "awaiting_prompt_confirm"
        assert "如果这个描述可以" in result["messages"][0].content

    def test_chat_after_image_generation(self, main_agent):
        """没有 active_subagent 时，普通反馈不应再次进入生图流程。"""
        messages = [
            HumanMessage(content="我想生成图片"),
            AIMessage(content="请告诉我您想要生成什么样的图片？"),
            HumanMessage(content="夕阳下看大海的少女"),
            AIMessage(content="图片已生成完成"),
        ]
        result = main_agent.process({"messages": messages}, "真好看")

        assert "image_url" not in result
        assert len(result["messages"]) == 1
        assert "你好呀" in result["messages"][0].content

    def test_clear_subagent_after_user_accepts_generated_image(self, main_agent):
        """图片确认阶段用户礼貌收尾时，应结束生图子Agent并清空上下文。"""
        image_agent = main_agent.subagent_registry.get("image")
        active_subagent = {
            "agent_type": "image",
            "agent": image_agent,
            "state": {
                "image_task_type": "generate_image",
                "image_phase": "awaiting_image_confirm",
                "image_prompt": "二次元风格，高中女生打着伞",
                "last_prompt": "二次元风格，高中女生打着伞",
                "last_image_urls": ["https://example.com/a.jpeg"],
            },
            "history": [
                HumanMessage(content="二次元风格，高中女生打着伞"),
                AIMessage(content="[图片] https://example.com/a.jpeg\n\n这张图片可以吗？"),
            ],
        }

        result = main_agent.process(
            {
                "messages": [],
                "active_subagent": active_subagent,
            },
            "可以了，谢谢",
        )

        assert result.get("clear_subagent") is True
        assert "喜欢就好" in result["messages"][0].content

    def test_subagent_receives_shared_runtime(self, main_agent):
        """注册子Agent时，子Agent应拿到与主Agent相同的共享工具运行时。"""
        image_agent = main_agent.subagent_registry.get("image")

        assert image_agent.runtime is main_agent._runtime

    def test_main_agent_executes_mcp_tool_from_shared_runtime(self, main_agent):
        """主Agent应能执行共享池中的 MCP 工具，并把结果交回 ReAct 下一轮。"""

        @tool("fake_mcp_video_info")
        def fake_mcp_video_info(video_path: str) -> str:
            """读取视频信息。"""
            return f"视频信息：{video_path}"

        main_agent._runtime._mcp_tools = [fake_mcp_video_info]
        main_agent._refresh_tools_with_mcp()

        result = main_agent.process({"messages": []}, "读取视频信息 F:/videos/a.mp4")

        assert "视频信息" in result["messages"][0].content
        assert "F:/videos/a.mp4" in result["messages"][0].content

    def test_main_agent_does_not_discover_mcp_on_startup(self, llm):
        """主Agent启动时只加载内置工具，不触发 MCP discovery。"""
        calls = []

        def fake_loader(*args, **kwargs):
            calls.append(kwargs)
            return []

        agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=fake_loader)

        assert calls == []
        assert agent._runtime.mcp_loaded is False
        assert "load_mcp_tools_tool" in agent.tool_map

    def test_main_agent_loads_mcp_only_when_tool_requests_it(self, llm):
        """只有调用 MCP 加载工具时，才初始化 MCP 工具缓存。"""
        calls = []

        def fake_loader(*args, **kwargs):
            calls.append(kwargs)
            return []

        agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=fake_loader)
        result = agent.process({"messages": []}, "加载视频MCP")

        assert len(calls) == 1
        assert agent._runtime.mcp_loaded is True
        assert "MCP扩展工具已加载" in result["messages"][0].content

    def test_mcp_prompt_principles_are_dynamic_after_loading(self, llm):
        """MCP 加载后，主Agent提示词应动态展示已发现工具而不是写死工具规则。"""
        agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=lambda *args, **kwargs: _fake_video_mcp_tools())

        agent._refresh_tools_with_mcp()
        prompt = agent._build_system_prompt().content

        assert "video-capture-script-mcp_analyze_video_content" in prompt
        assert "video-capture-script-mcp_generate_video_script" in prompt
        assert "video-capture-script-mcp_extract_video_frames" not in prompt
        assert "generate_image_tool" not in {tool.name for tool in agent.tools}

    def test_video_mcp_analyze_video_content(self, llm):
        """主Agent应选择 MCP 视频内容分析工具处理本地视频分析。"""
        agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=lambda *args, **kwargs: _fake_video_mcp_tools())
        result = agent.process({"messages": []}, f"这是本地视频{VIDEO_PATH}，请帮分析视频内容")

        assert "视频内容分析结果" in result["messages"][0].content
        assert VIDEO_PATH in result["messages"][0].content

    def test_video_mcp_analyze_single_image(self, llm):
        """主Agent应选择图片批量分析工具处理单张本地图片分析。"""
        agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=lambda *args, **kwargs: _fake_video_mcp_tools())
        result = agent.process({"messages": []}, f"这是本地图片{IMAGE_PATH_1}，请帮我基于图片分析图片内容")

        assert "图片内容分析结果" in result["messages"][0].content
        assert IMAGE_PATH_1 in result["messages"][0].content

    def test_video_mcp_analyze_multiple_images(self, llm):
        """主Agent应选择图片批量分析工具处理多张本地图片分析。"""
        agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=lambda *args, **kwargs: _fake_video_mcp_tools())
        user_input = f"这是本地图片1{IMAGE_PATH_1}，本地图片2{IMAGE_PATH_2}，本地图片3{IMAGE_PATH_3}，请帮我基于图片分析图片内容"
        result = agent.process({"messages": []}, user_input)

        assert "图片内容分析结果" in result["messages"][0].content
        assert IMAGE_PATH_1 in result["messages"][0].content
        assert IMAGE_PATH_3 in result["messages"][0].content

    def test_video_mcp_generate_image_script(self, llm):
        """主Agent应选择图片脚本工具处理基于图片生成拍摄脚本。"""
        agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=lambda *args, **kwargs: _fake_video_mcp_tools())
        user_input = f"这是本地图片1{IMAGE_PATH_1}，本地图片2{IMAGE_PATH_2}，本地图片3{IMAGE_PATH_3}，请帮我基于图片生成专业拍摄脚本"
        result = agent.process({"messages": []}, user_input)

        assert "图片脚本生成结果" in result["messages"][0].content
        assert IMAGE_PATH_2 in result["messages"][0].content

    def test_video_mcp_generate_video_script(self, llm):
        """主Agent应选择视频脚本工具处理基于视频生成拍摄脚本。"""
        agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=lambda *args, **kwargs: _fake_video_mcp_tools())
        result = agent.process({"messages": []}, f"这是本地视频{VIDEO_PATH}，请帮我基于视频内容生成专业拍摄脚本")

        assert "视频脚本生成结果" in result["messages"][0].content
        assert VIDEO_PATH in result["messages"][0].content

    def test_video_mcp_get_video_info(self, llm):
        """主Agent应使用 videoPath 参数调用视频基本信息工具。"""
        agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=lambda *args, **kwargs: _fake_video_mcp_tools())
        result = agent.process({"messages": []}, f"这是视频地址{VIDEO_PATH}，获取视频文件基本信息")

        assert "视频信息" in result["messages"][0].content
        assert VIDEO_PATH in result["messages"][0].content

    def test_video_mcp_get_video_info_loads_mcp_then_returns_result(self, llm):
        """MCP 未加载时的视频信息请求，应先加载 MCP，再调用视频信息工具并返回结果。"""
        calls = []

        def fake_loader(*args, **kwargs):
            calls.append(kwargs)
            return _fake_video_mcp_tools()

        agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=fake_loader)
        result = agent.process({"messages": []}, f"这是视频地址{VIDEO_PATH}，获取视频文件基本信息")

        assert len(calls) == 1
        assert "视频信息" in result["messages"][0].content
        assert VIDEO_PATH in result["messages"][0].content

    def test_reverse_image_prompt_tool(self, llm):
        """主Agent应调用识图工具反推图片提示词。"""
        agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=lambda *args, **kwargs: [])

        @tool("reverse_image_prompt")
        def fake_reverse_image_prompt(
            image_path: str,
            style: str = "default",
            previous_prompt: str = "",
        ) -> str:
            """根据图片反推 AI 绘画提示词。"""
            return f"反推提示词：{image_path}"

        agent.tool_map["reverse_image_prompt"] = fake_reverse_image_prompt
        agent.tools.append(fake_reverse_image_prompt)

        result = agent.process({"messages": []}, f"这张图片{IMAGE_PATH_1}，反推提示词")

        assert "反推提示词" in result["messages"][0].content
        assert IMAGE_PATH_1 in result["messages"][0].content

    def test_reverse_image_prompt_updates_vision_context(self, llm):
        """识图工具执行后，主Agent应保存轻量图像上下文。"""
        agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=lambda *args, **kwargs: [])

        @tool("reverse_image_prompt")
        def reverse_tool(
            image_path: str,
            style: str = "default",
            previous_prompt: str = "",
        ) -> str:
            """根据图片反推 AI 绘画提示词。"""
            return f"反推提示词：{image_path}"

        agent.tool_map["reverse_image_prompt"] = reverse_tool
        agent.tools.append(reverse_tool)

        result = agent.process({"messages": []}, f"这张图片{IMAGE_PATH_1}，反推提示词")

        assert result.get("vision_context", {}).get("image_path") == IMAGE_PATH_1
        assert result.get("vision_context", {}).get("last_reverse_prompt") == f"反推提示词：{IMAGE_PATH_1}"

    def test_douyin_mcp_download_video(self, llm):
        """主Agent应动态调用抖音 MCP 的下载链接工具。"""
        agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=lambda *args, **kwargs: _fake_all_mcp_tools())
        result = agent.process({"messages": []}, f"{DOUYIN_URL}，下载视频")

        assert "抖音视频下载链接" in result["messages"][0].content
        assert DOUYIN_URL in result["messages"][0].content
        assert "generate_image_tool" not in {tool.name for tool in agent.tools}

    def test_bilibili_mcp_get_video_info(self, llm):
        """主Agent应动态调用 B站 MCP 的视频信息工具。"""
        agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=lambda *args, **kwargs: _fake_all_platform_mcp_tools())
        result = agent.process({"messages": []}, f"获取这个B站视频的信息：{BILIBILI_URL}")

        assert "B站视频信息" in result["messages"][0].content
        assert BILIBILI_URL in result["messages"][0].content

    def test_bilibili_mcp_download_video(self, llm):
        """主Agent应动态调用 B站 MCP 的下载视频工具。"""
        agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=lambda *args, **kwargs: _fake_all_platform_mcp_tools())
        result = agent.process({"messages": []}, f"下载这个B站视频：{BILIBILI_URL}")

        assert "B站视频已下载" in result["messages"][0].content
        assert BILIBILI_URL in result["messages"][0].content

    def test_fysh_bilibili_mcp_parse_video(self, llm):
        """主Agent应动态调用 fysh1010/bilibili-mcp 的解析工具。"""
        agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=lambda *args, **kwargs: _fake_all_bilibili_mcp_tools())
        result = agent.process({"messages": []}, f"解析这个B站视频：{BILIBILI_URL}")

        assert "fysh B站视频解析结果" in result["messages"][0].content
        assert BILIBILI_URL in result["messages"][0].content

    def test_unknown_mcp_tool_loads_then_replans_with_schema(self, llm):
        """MCP 未加载时猜到工具名，不应立即用旧参数执行，应加载后重新规划。"""
        calls = []

        def fake_loader(*args, **kwargs):
            calls.append(kwargs)
            return _fake_video_mcp_tools()

        agent = HuesaeMainAgent(llm=llm, mcp_tools_loader=fake_loader)
        result = agent.process({"messages": []}, "MCP未加载时直接猜测视频信息工具")

        assert len(calls) == 1
        assert "视频信息" in result["messages"][0].content
        assert VIDEO_PATH in result["messages"][0].content

    def test_main_agent_injects_memory_into_prompt(self, llm):
        """主Agent应把 Honcho 记忆注入系统提示词。"""

        class _FakeMemory:
            enabled = True
            user_input = None

            def get_context(self, user_input=None):
                self.user_input = user_input
                return "记忆：用户喜欢猫"

        memory = _FakeMemory()
        agent = HuesaeMainAgent(llm=llm, memory_service=memory)
        prompt = agent._build_system_prompt("我喜欢什么动物？").content

        assert "用户喜欢猫" in prompt
        assert "Honcho 长期记忆 / 持久记忆" in prompt
        assert memory.user_input == "我喜欢什么动物？"


class TestAgentMiddlewares:
    """测试 DeerFlow 风格中间件管道。"""

    def test_pipeline_runs_hooks_in_order(self):
        """Pipeline 应按生命周期顺序执行中间件钩子。"""
        calls = []

        class _RecordingMiddleware(AgentMiddleware):
            def before_agent(self, state, runtime):
                calls.append(("before_agent", state["user_input"]))

            def before_model(self, state, runtime):
                calls.append(("before_model", state["step"]))

            def after_model(self, state, runtime):
                calls.append(("after_model", state["messages"][-1].content))

            def after_agent(self, state, runtime):
                calls.append(("after_agent", state["result"]["messages"][0].content))

        pipeline = MiddlewarePipeline([_RecordingMiddleware()])
        state = pipeline.run_before_agent({"messages": [], "user_input": "你好"})
        state["step"] = 0
        state = pipeline.run_before_model(state)
        state["messages"] = [AIMessage(content="模型回复")]
        state = pipeline.run_after_model(state)
        state["result"] = {"messages": [AIMessage(content="最终回复")]}
        pipeline.run_after_agent(state)

        assert calls == [
            ("before_agent", "你好"),
            ("before_model", 0),
            ("after_model", "模型回复"),
            ("after_agent", "最终回复"),
        ]

    def test_pipeline_appends_message_updates(self):
        """中间件返回 messages 时应使用追加语义。"""

        class _AppendMessageMiddleware(AgentMiddleware):
            def before_model(self, state, runtime):
                return {"messages": [HumanMessage(content="补充消息")], "flag": True}

        pipeline = MiddlewarePipeline([_AppendMessageMiddleware()])
        state = pipeline.run_before_model({"messages": [HumanMessage(content="原始消息")]})

        assert [message.content for message in state["messages"]] == ["原始消息", "补充消息"]
        assert state["flag"] is True

    def test_token_usage_middleware_logs_usage(self, caplog):
        """TokenUsageMiddleware 应记录 AIMessage usage_metadata。"""
        middleware = TokenUsageMiddleware()
        message = AIMessage(
            content="你好",
            usage_metadata={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        )

        with caplog.at_level("INFO"):
            middleware.after_model({"messages": [message]}, runtime=None)

        assert "LLM token usage: input=3 output=4 total=7" in caplog.text

    def test_build_middlewares_respects_token_usage_config(self):
        """关闭配置后不应组装 TokenUsageMiddleware。"""
        try:
            set_middleware_config(
                MiddlewareConfig(token_usage=TokenUsageConfig(enabled=False))
            )
            pipeline = build_middlewares()

            assert not any(
                isinstance(middleware, TokenUsageMiddleware)
                for middleware in pipeline.middlewares
            )
        finally:
            reset_middleware_config()

    def test_main_agent_invokes_middleware_around_model(self, llm):
        """主Agent ReAct 模型调用前后应触发中间件。"""
        calls = []

        class _MainAgentMiddleware(AgentMiddleware):
            def before_agent(self, state, runtime):
                calls.append("before_agent")

            def before_model(self, state, runtime):
                calls.append("before_model")

            def after_model(self, state, runtime):
                calls.append("after_model")

            def after_agent(self, state, runtime):
                calls.append("after_agent")

        agent = HuesaeMainAgent(
            llm=llm,
            mcp_tools_loader=lambda *args, **kwargs: [],
            middleware_pipeline=MiddlewarePipeline([_MainAgentMiddleware()]),
        )
        result = agent.process({"messages": []}, "你好")

        assert "你好呀" in result["messages"][0].content
        assert calls == ["before_agent", "before_model", "after_model", "after_agent"]


class TestDanbooruTags:
    """测试 Danbooru 标签生成。"""

    def test_generate_tags(self, llm):
        """标签生成函数应返回清洗后的标签列表。"""
        tags = generate_tags("一个银发红瞳的少女在樱花树下", llm)

        assert tags[:3] == ["1girl", "silver hair", "red eyes"]
        assert "anime style" in tags

    def test_tags_to_prompt(self):
        """标签列表应拼接为逗号分隔提示词。"""
        from huesaeagents.huesae.subagents.image import tags_to_prompt

        tags = ["1girl", "silver hair", "red eyes"]
        prompt = tags_to_prompt(tags)

        assert prompt == "1girl, silver hair, red eyes"


class TestExpandPrompt:
    """测试提示词扩写。"""

    def test_expand_prompt(self, llm):
        """扩写函数应返回更长的自然语言描述。"""
        expanded = expand_prompt("夕阳下的战舰", llm)

        assert isinstance(expanded, str)
        assert len(expanded) > len("夕阳下的战舰")
        assert "战舰" in expanded


class TestProviders:
    """测试 Provider 注册。"""

    def test_register_provider(self, llm):
        """生图Agent应支持动态注册Provider。"""
        agent = ImageSubAgent(llm=llm, providers=[])

        assert len(agent.providers) == 0

        agent.register_provider(DoubaoProvider())

        assert "doubao" in agent.providers

    def test_default_provider(self, llm):
        """默认创建生图Agent时应注册豆包Provider。"""
        agent = create_image_agent(llm=llm)
        names = agent.get_available_providers()

        assert "doubao" in names
