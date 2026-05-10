"""Agent 工具定义"""
from langchain.tools import tool

try:
    from ..tools.image import generate_image_by_jimeng, generate_image_by_doubao, generate_images_by_doubao
except ImportError:
    from huesaeagents.huesae.tools.image import generate_image_by_jimeng, generate_image_by_doubao, generate_images_by_doubao


@tool
async def generate_image_jimeng(prompt: str, width: int = 1024, height: int = 1024) -> str:
    """
    生成图片（即梦AI）

    Args:
        prompt: 自然语言描述，如 "一个银发红瞳的少女在樱花树下"
        width: 图片宽度，默认 1024
        height: 图片高度，默认 1024

    Returns:
        生成的图片 URL
    """
    return await generate_image_by_jimeng(prompt=prompt, width=width, height=height)


@tool
async def generate_image_doubao(prompt: str, size: str = "2K") -> str:
    """
    生成图片（豆包）

    Args:
        prompt: 自然语言描述，如 "一个银发红瞳的少女在樱花树下，电影感"
        size: 图片尺寸，支持 1K, 2K, 4K 等，默认 2K

    Returns:
        生成的图片 URL
    """
    return await generate_image_by_doubao(prompt=prompt, size=size)


@tool
async def generate_images_doubao(prompt: str, size: str = "2K", max_images: int = 4) -> str:
    """
    生成一组图片（豆包，最多4张）

    Args:
        prompt: 提示词，需描述生成一组连贯图片，如 "生成一组共4张连贯插画，核心为同一庭院一角的四季变迁"
        size: 图片尺寸，支持 1K, 2K, 4K 等，默认 2K
        max_images: 最大图片数量，默认 4

    Returns:
        图片 base64 JSON 列表
    """
    return await generate_images_by_doubao(prompt=prompt, size=size, max_images=max_images)


# 导出所有工具
IMAGE_TOOLS = [generate_image_jimeng, generate_image_doubao, generate_images_doubao]