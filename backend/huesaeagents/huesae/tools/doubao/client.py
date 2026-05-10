"""豆包图片生成 API 客户端"""
import os
from typing import Generator, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class DoubaoImageError(Exception):
    """豆包图片生成异常"""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class DoubaoClient:
    """豆包图片生成 API 客户端（单例）"""

    BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    MODEL = os.getenv("DOUBAO_SEEDREAM_MODEL_NAME")

    _instance: Optional["DoubaoClient"] = None

    def __new__(cls, api_key: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化豆包客户端

        Args:
            api_key: API Key，不传则从环境变量 ARK_API_KEY 读取
        """
        if self._initialized:
            return

        self.api_key = api_key or os.getenv("DOUBAO_SEEDREAM_API_KEY")
        if not self.api_key:
            raise ValueError("ARK_API_KEY must be set in environment or passed as argument")

        self.client = OpenAI(
            base_url=self.BASE_URL,
            api_key=self.api_key,
        )
        self._initialized = True

    def generate_image(
        self,
        prompt: str,
        size: str = "2K",
        response_format: str = "url",
        watermark: bool = True,
        timeout: int = 120,
    ) -> str:
        """
        生成图片

        Args:
            prompt: 文本提示词
            size: 图片尺寸，支持 1K, 2K, 4K 等
            response_format: 返回格式，url 或 b64_json
            watermark: 是否添加水印
            timeout: 超时时间（秒）

        Returns:
            图片 URL 或 base64 JSON

        Raises:
            DoubaoImageError: 生成失败时抛出
        """
        try:
            response = self.client.images.generate(
                model=self.MODEL,
                prompt=prompt,
                size=size,
                response_format=response_format,
                extra_body={
                    "watermark": watermark,
                },
                timeout=timeout,
            )

            if not response.data:
                raise DoubaoImageError(-1, "No image data returned")

            if response_format == "url":
                return response.data[0].url
            else:
                return response.data[0].b64_json

        except Exception as e:
            if isinstance(e, DoubaoImageError):
                raise
            raise DoubaoImageError(-1, str(e))

    def generate_image_debug(
        self,
        prompt: str,
        size: str = "2K",
        response_format: str = "url",
        watermark: bool = True,
        timeout: int = 120,
    ) -> dict:
        """
        生成图片（调试模式，返回完整响应）

        Returns:
            dict包含完整的API响应
        """
        response = self.client.images.generate(
            model=self.MODEL,
            prompt=prompt,
            size=size,
            response_format=response_format,
            extra_body={
                "watermark": watermark,
            },
            timeout=timeout,
        )
        return {
            "response_format": response_format,
            "url": response.data[0].url if hasattr(response.data[0], 'url') else None,
            "b64_json": response.data[0].b64_json if hasattr(response.data[0], 'b64_json') else None,
            "raw": response.model_dump(),
        }

    def generate_images_stream(
        self,
        prompt: str,
        size: str = "2K",
        max_images: int = 4,
        watermark: bool = True,
        timeout: int = 300,
    ) -> Generator[str, None, None]:
        """
        流式生成一组图片（最多4张）

        Args:
            prompt: 提示词，需描述生成一组连贯图片
            size: 图片尺寸
            max_images: 最大图片数量，默认4
            watermark: 是否添加水印
            timeout: 超时时间（秒）

        Yields:
            str: 每张图片的 base64 JSON

        Raises:
            DoubaoImageError: 生成失败时抛出
        """
        try:
            response = self.client.images.generate(
                model=self.MODEL,
                prompt=prompt,
                size=size,
                response_format="b64_json",
                stream=True,
                extra_body={
                    "watermark": watermark,
                    "sequential_image_generation": "auto",
                    "sequential_image_generation_options": {
                        "max_images": max_images,
                    },
                },
                timeout=timeout,
            )

            for event in response:
                if event is None:
                    continue
                elif event.type == "image_generation.partial_succeeded":
                    if event.b64_json is not None:
                        yield event.b64_json
                elif event.type == "image_generation.completed":
                    # 流式生成结束，不再 yield
                    break

        except Exception as e:
            if isinstance(e, DoubaoImageError):
                raise
            raise DoubaoImageError(-1, str(e))


def create_doubao_client() -> DoubaoClient:
    """
    工厂函数：从环境变量创建 DoubaoClient

    Returns:
        DoubaoClient 实例
    """
    return DoubaoClient()