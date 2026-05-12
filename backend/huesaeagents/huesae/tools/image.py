"""图片工具模块"""
from dataclasses import dataclass
from typing import Optional

try:
    from .jimeng import JimengClient, JimengAIError, create_jimeng_client
    from .doubao import (
        DoubaoClient,
        DoubaoImageError,
        create_doubao_client,
        generate_images_by_doubao as doubao_generate_images,
    )
except ImportError:
    from huesaeagents.huesae.tools.jimeng import JimengClient, JimengAIError, create_jimeng_client
    from huesaeagents.huesae.tools.doubao import (
        DoubaoClient,
        DoubaoImageError,
        create_doubao_client,
        generate_images_by_doubao as doubao_generate_images,
    )


# ============== 即梦图片生成（Jimeng）==============


@dataclass
class JimengAIError(Exception):
    """即梦AI异常"""

    code: int
    message: str

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


async def generate_image_by_jimeng(
    prompt: str,
    width: int = 2048,
    height: int = 2048,
    scale: float = 0.5,
    force_single: bool = True,
    timeout: int = 120,
    image_urls: Optional[list[str]] = None,
) -> str:
    """
    自然语言生成图片（即梦AI）

    Args:
        prompt: 自然语言描述，如 "一个银发红瞳的少女在樱花树下"
        width: 图片宽度，默认2048
        height: 图片高度，默认2048
        scale: 文本影响程度(0-1)，默认0.5
        force_single: 是否强制生成单图，默认True
        timeout: 超时时间（秒），默认120
        image_urls: 参考图片URL列表（用于图生图），可选

    Returns:
        生成的图片URL（单张）

    Raises:
        JimengAIError: 生成失败时抛出
        TimeoutError: 超时时抛出

    Example:
        >>> from tools.image import generate_image_by_jimeng
        >>> image_url = await generate_image_by_jimeng(
        ...     prompt="一个银发红瞳的少女在樱花树下"
        ... )
        >>> print(image_url)
    """
    client = create_jimeng_client()

    image_urls_result = client.generate_image(
        prompt=prompt,
        width=width,
        height=height,
        scale=scale,
        force_single=force_single,
        timeout=timeout,
        image_urls=image_urls,
    )

    if not image_urls_result:
        raise JimengAIError(-1, "No image URL returned")

    return image_urls_result[0]


async def generate_image_by_jimeng_from_text(
    prompt: str,
    width: int = 2048,
    height: int = 2048,
    scale: float = 0.5,
    force_single: bool = True,
    timeout: int = 120,
    image_urls: Optional[list[str]] = None,
) -> str:
    """
    自然语言生成图片（即梦AI，generate_image_by_jimeng 的别名）

    Args:
        prompt: 自然语言描述
        width: 图片宽度
        height: 图片高度
        scale: 文本影响程度
        force_single: 是否强制生成单图
        timeout: 超时时间
        image_urls: 参考图片URL列表

    Returns:
        生成的图片URL
    """
    return await generate_image_by_jimeng(
        prompt=prompt,
        width=width,
        height=height,
        scale=scale,
        force_single=force_single,
        timeout=timeout,
        image_urls=image_urls,
    )


# ============== 豆包图片生成（Doubao）==============


@dataclass
class DoubaoImageError(Exception):
    """豆包图片生成异常"""

    code: int
    message: str

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


async def generate_image_by_doubao(
    prompt: str,
    size: str = "2K",
    watermark: bool = False,
    response_format: str = "url",
    output_format: str = "jpeg",
    timeout: int = 120,
) -> str:
    """
    自然语言生成图片（豆包）

    Args:
        prompt: 自然语言描述，如 "星际穿越，黑洞，黑洞里冲出一辆快支离破碎的复古列车"
        size: 图片尺寸，支持 1K, 2K, 4K 等，默认 2K
        watermark: 是否添加水印，默认 False
        response_format: 返回格式，url 或 b64_json，默认 url
        output_format: 输出图片格式，支持 jpeg/png，默认 jpeg
        timeout: 超时时间（秒），默认 120

    Returns:
        生成的图片 URL 或 base64 JSON

    Raises:
        DoubaoImageError: 生成失败时抛出

    Example:
        >>> from tools.image import generate_image_by_doubao
        >>> image_url = await generate_image_by_doubao(
        ...     prompt="一个银发红瞳的少女在樱花树下"
        ... )
        >>> print(image_url)
    """
    client = create_doubao_client()
    return client.generate_image(
        prompt=prompt,
        size=size,
        watermark=watermark,
        response_format=response_format,
        output_format=output_format,
        timeout=timeout,
    )


async def generate_images_by_doubao(
    prompt: str,
    size: str = "2K",
    max_images: int = 10,
    watermark: bool = False,
    response_format: str = "url",
    output_format: str = "jpeg",
    timeout: int = 300,
) -> list[str]:
    """
    生成一组图片（豆包，组图模式）

    Args:
        prompt: 提示词，需描述生成一组连贯图片，如 "生成一组共4张连贯插画，核心为同一庭院一角的四季变迁"
        size: 图片尺寸，支持 1K, 2K, 4K 等，默认 2K
        max_images: 最大图片数量，默认 10
        watermark: 是否添加水印，默认 False
        output_format: 输出图片格式，支持 jpeg/png，默认 jpeg
        timeout: 超时时间（秒），默认 300

    Returns:
        图片 base64 JSON 列表

    Raises:
        DoubaoImageError: 生成失败时抛出

    Example:
        >>> from tools.image import generate_images_by_doubao
        >>> images = await generate_images_by_doubao(
        ...     prompt="生成一组共4张连贯插画，核心为同一庭院一角的四季变迁"
        ... )
        >>> print(f"生成了 {len(images)} 张图片")
    """
    return await doubao_generate_images(
        prompt=prompt,
        size=size,
        max_images=max_images,
        watermark=watermark,
        response_format=response_format,
        output_format=output_format,
        timeout=timeout,
    )


# 保持向后兼容
JimengAIError = JimengAIError
DoubaoImageError = DoubaoImageError


if __name__ == "__main__":
    import asyncio
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    os.chdir(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    async def main():
        url = await generate_image_by_jimeng("一个银发红瞳的少女在樱花树下")
        print(f"生成的图片URL: {url}")

    asyncio.run(main())