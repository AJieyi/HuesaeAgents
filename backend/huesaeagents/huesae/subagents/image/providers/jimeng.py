"""即梦AI图片生成提供者

对接已有的 huesae.tools.jimeng 工具
"""
import asyncio
import os

from .base import ImageProvider, GenerationResult

try:
    from huesae.tools.jimeng import JimengClient, create_jimeng_client
except ImportError:
    from huesaeagents.huesae.tools.jimeng import JimengClient, create_jimeng_client


class JimengProvider(ImageProvider):
    """即梦AI图片生成提供者"""

    def __init__(self, access_key: str | None = None, secret_key: str | None = None):
        self.access_key = access_key or os.getenv("JIMENG_ACCESS_KEY_ID")
        self.secret_key = secret_key or os.getenv("JIMENG_SECRET_ACCESS_KEY")

    @property
    def name(self) -> str:
        return "jimeng"

    async def generate(
        self,
        prompt: str,
        size: str = "2K",
        **kwargs,
    ) -> GenerationResult:
        """调用即梦AI API生成图片

        Args:
            prompt: 自然语言
            size: 图片尺寸（映射到width/height）
            **kwargs: 额外参数

        Returns:
            GenerationResult: 包含图片URL的生成结果
        """
        size_map = {
            "1K": (1024, 1024),
            "2K": (2048, 2048),
            "4K": (4096, 4096),
        }
        width, height = size_map.get(size, (2048, 2048))

        client = self._create_client()
        image_urls = await asyncio.to_thread(
            client.generate_image,
            prompt,
            width,
            height,
        )
        if not image_urls:
            raise ValueError("即梦没有返回图片地址")
        url = image_urls[0]
        return GenerationResult(
            url=url,
            provider=self.name,
            prompt=prompt,
            size=f"{width}x{height}",
        )

    def _create_client(self) -> JimengClient:
        """创建即梦客户端，优先使用显式传入的 AK/SK。"""
        if self.access_key and self.secret_key:
            return JimengClient(self.access_key, self.secret_key)
        return create_jimeng_client()
