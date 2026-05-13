"""Agent 工具定义"""
from langchain.tools import tool

try:
    from ..tools.image import generate_image_by_jimeng, generate_image_by_doubao
except ImportError:
    from huesaeagents.huesae.tools.image import generate_image_by_jimeng, generate_image_by_doubao


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
async def generate_image_doubao(
    prompt: str,
    size: str = "2K",
    watermark: bool = False,
    response_format: str = "url",
    output_format: str = "jpeg",
) -> str:
    """
    生成图片（豆包）

    Args:
        prompt: 自然语言描述，如 "一个银发红瞳的少女在樱花树下，比例16:9"
        size: 图片尺寸，支持 1K, 2K, 4K 等，默认 2K
        watermark: 是否添加水印，默认 False
        response_format: 返回格式，url 或 b64_json，默认 url
        output_format: 输出图片格式，支持 jpeg/png，默认 jpeg

    Returns:
        生成的图片 URL 或 base64 JSON
    """
    return await generate_image_by_doubao(
        prompt=prompt,
        size=size,
        watermark=watermark,
        response_format=response_format,
        output_format=output_format,
    )


# @tool
# async def generate_images_doubao(
#     prompt: str,
#     size: str = "2K",
#     watermark: bool = False,
#     response_format: str = "url",
#     output_format: str = "jpeg",
# ) -> str:
#     """
#     生成一组图片（豆包，组图模式）

#     Args:
#         prompt: 提示词，需描述生成一组连贯图片，如 "生成一组共4张连贯插画，核心为同一庭院一角的四季变迁"
#         size: 图片尺寸，支持 1K, 2K, 4K 等，默认 2K
#         watermark: 是否添加水印，默认 False
#         output_format: 输出图片格式，支持 jpeg/png，默认 jpeg

#     Returns:
#         图片 base64 JSON 列表
#     """
#     return await generate_images_by_doubao(
#         prompt=prompt,
#         size=size,
#         watermark=watermark,
#         response_format=response_format,
#         output_format=output_format,
#     )


# 导出所有工具
IMAGE_TOOLS = [generate_image_jimeng, generate_image_doubao]
