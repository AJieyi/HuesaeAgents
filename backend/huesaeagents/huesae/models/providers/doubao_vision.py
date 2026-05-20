"""豆包多模态视觉理解 Provider。"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class DoubaoVisionError(Exception):
    """豆包视觉理解调用异常。"""


class DoubaoVisionClient:
    """封装火山方舟 Doubao 多模态图片理解接口。"""

    BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    DEFAULT_MODEL = "doubao-seed-2-0-mini-260428"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
    ):
        self.model = model or os.getenv("DOUBAO_MODEL_NAME") or self.DEFAULT_MODEL
        self.api_key = api_key or os.getenv("DOUBAO_API_KEY")
        self.timeout = timeout

        if not self.api_key:
            raise DoubaoVisionError("未配置 DOUBAO_API_KEY，无法调用豆包视觉理解模型")

        self.client = OpenAI(
            base_url=self.BASE_URL,
            api_key=self.api_key,
            timeout=timeout,
        )

    def understand_image(self, image_path: str, prompt: str) -> str:
        """发送图片和文本指令给多模态模型，返回中文文本结果。"""
        image_url = self._resolve_image_url(image_path)
        response = self.client.responses.create(
            model=self.model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": image_url},
                    {"type": "input_text", "text": prompt},
                ],
            }],
        )
        output_text = str(getattr(response, "output_text", "") or "").strip()
        if not output_text:
            raise DoubaoVisionError("豆包视觉理解模型没有返回可展示文本")
        return output_text

    def _resolve_image_url(self, image_path: str) -> str:
        """把本地图片路径转成 data URI；网络图片 URL 原样传递。"""
        image_path = str(image_path or "").strip().strip('"')
        if not image_path:
            raise DoubaoVisionError("图片路径不能为空")

        if image_path.startswith(("http://", "https://", "data:")):
            return image_path

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片不存在：{image_path}")
        if not path.is_file():
            raise DoubaoVisionError(f"图片路径不是文件：{image_path}")

        mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"


def create_doubao_vision_client(**kwargs) -> DoubaoVisionClient:
    """创建豆包视觉理解客户端。"""
    return DoubaoVisionClient(**kwargs)
