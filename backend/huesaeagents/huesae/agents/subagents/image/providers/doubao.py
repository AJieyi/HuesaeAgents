"""豆包Seedream图片生成提供者

对接已有的 huesae.tools.doubao 工具
"""
import os

from .base import ImageProvider, GenerationResult

try:
    from huesae.tools.image import generate_image_by_doubao
except ImportError:
    from huesaeagents.huesae.tools.image import generate_image_by_doubao


class DoubaoProvider(ImageProvider):
    """豆包Seedream图片生成提供者"""

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        self.api_key = api_key or os.getenv("DOUBAO_SEEDREAM_API_KEY")
        self.model_name = model_name or os.getenv("DOUBAO_SEEDREAM_MODEL_NAME", "doubao-seedream-5-0-260128")

    @property
    def name(self) -> str:
        return "doubao"

    async def generate(
        self,
        prompt: str,
        size: str = "2K",
        **kwargs,
    ) -> GenerationResult:
        """调用豆包API生成图片

        Args:
            prompt: 提示词（Danbooru标签拼接）
            size: 图片尺寸，支持 1K, 2K, 4K
            **kwargs: 额外参数

        Returns:
            GenerationResult: 包含图片URL的生成结果
        """
        url = await generate_image_by_doubao(
            prompt=prompt,
            size=size,
        )
        return GenerationResult(
            url=url,
            provider=self.name,
            prompt=prompt,
            size=size,
        )
