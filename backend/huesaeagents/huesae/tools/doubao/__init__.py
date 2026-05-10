"""豆包图片生成工具"""
from dataclasses import dataclass

from .client import DoubaoClient, DoubaoImageError, create_doubao_client

__all__ = [
    "DoubaoClient",
    "DoubaoImageError",
    "create_doubao_client",
    "generate_image_by_text",
    "generate_image",
    "generate_images_by_doubao",
]


@dataclass
class DoubaoImageResult:
    """豆包图片生成结果"""
    url: str
    revised_prompt: str = ""


async def generate_image_by_text(
    prompt: str,
    size: str = "2K",
    watermark: bool = True,
    timeout: int = 120,
) -> str:
    """
    自然语言生成图片（豆包）

    Args:
        prompt: 自然语言描述，如 "星际穿越，黑洞，黑洞里冲出一辆快支离破碎的复古列车"
        size: 图片尺寸，支持 1K, 2K, 4K 等，默认 2K
        watermark: 是否添加水印，默认 True
        timeout: 超时时间（秒），默认 120

    Returns:
        生成的图片 URL

    Raises:
        DoubaoImageError: 生成失败时抛出

    Example:
        >>> from tools.doubao import generate_image_by_text
        >>> image_url = await generate_image_by_text(
        ...     prompt="一个银发红瞳的少女在樱花树下"
        ... )
        >>> print(image_url)
    """
    client = create_doubao_client()
    return client.generate_image(
        prompt=prompt,
        size=size,
        response_format="url",
        watermark=watermark,
        timeout=timeout,
    )


async def generate_image(
    prompt: str,
    size: str = "2K",
    watermark: bool = True,
    timeout: int = 120,
) -> str:
    """
    自然语言生成图片（generate_image_by_text 的别名）

    Args:
        prompt: 自然语言描述
        size: 图片尺寸
        watermark: 是否添加水印
        timeout: 超时时间

    Returns:
        生成的图片 URL
    """
    return await generate_image_by_text(
        prompt=prompt,
        size=size,
        watermark=watermark,
        timeout=timeout,
    )


async def generate_images_by_doubao(
    prompt: str,
    size: str = "2K",
    max_images: int = 4,
    watermark: bool = True,
    timeout: int = 300,
) -> list[str]:
    """
    生成一组图片（豆包）

    Args:
        prompt: 提示词，需描述生成一组连贯图片，如 "生成一组共4张连贯插画，核心为同一庭院一角的四季变迁"
        size: 图片尺寸，支持 1K, 2K, 4K 等，默认 2K
        max_images: 最大图片数量，默认4
        watermark: 是否添加水印，默认 True
        timeout: 超时时间（秒），默认 300

    Returns:
        图片 base64 JSON 列表

    Raises:
        DoubaoImageError: 生成失败时抛出

    Example:
        >>> from tools.doubao import generate_images_by_doubao
        >>> images = await generate_images_by_doubao(
        ...     prompt="生成一组共4张连贯插画，核心为同一庭院一角的四季变迁"
        ... )
        >>> print(f"生成了 {len(images)} 张图片")
    """
    client = create_doubao_client()
    images = []
    for b64_json in client.generate_images_stream(
        prompt=prompt,
        size=size,
        max_images=max_images,
        watermark=watermark,
        timeout=timeout,
    ):
        images.append(b64_json)
    return images


if __name__ == "__main__":
    import asyncio

    async def main():
        url = await generate_image_by_text(
            "一个银发红瞳的少女在樱花树下，电影感，温暖光线"
        )
        print(f"生成的图片URL: {url}")

    asyncio.run(main())