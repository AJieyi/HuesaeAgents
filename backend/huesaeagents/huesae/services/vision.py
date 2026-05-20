"""图片理解服务。"""

from __future__ import annotations

from typing import Literal

from ..models.models_factory import create_vision_client

ReversePromptStyle = Literal["default", "alternative"]


REVERSE_PROMPT_INSTRUCTIONS: dict[ReversePromptStyle, str] = {
    "default": """请仔细观察这张图片，反推一段可直接用于 AI 绘画或二次元生图的中文提示词。
要求：
1. 描述主体、角色、动作、表情、场景、构图、光线、色调和氛围。
2. 如图片包含二次元、漫画、插画、厚涂、赛璐璐等画风特征，请自然写入提示词。
3. 不要输出分析过程，不要使用列表，不要输出英文标签。
4. 控制在 100 到 200 个中文字符，只输出提示词正文。""",
    "alternative": """请重新观察这张图片，基于同一张图反推另一版中文 AI 绘画提示词。
上一版提示词：
{previous_prompt}

要求：
1. 保持图片核心内容一致，但换一种构图、光线、氛围或细节组织方式来描述。
2. 不要输出分析过程，不要使用列表，不要输出英文标签。
3. 控制在 100 到 200 个中文字符，只输出提示词正文。""",
}


class VisionService:
    """图片反推提示词服务。"""

    def __init__(self, client=None):
        self.client = client or create_vision_client()

    def reverse_prompt(
        self,
        image_path: str,
        style: ReversePromptStyle = "default",
        previous_prompt: str = "",
    ) -> str:
        """根据图片路径反推 AI 绘画提示词。"""
        if style not in REVERSE_PROMPT_INSTRUCTIONS:
            raise ValueError(f"未知反推风格：{style}")

        instruction = REVERSE_PROMPT_INSTRUCTIONS[style]
        if style == "alternative":
            instruction = instruction.format(previous_prompt=previous_prompt or "暂无")
        return self.client.understand_image(image_path, instruction).strip()
