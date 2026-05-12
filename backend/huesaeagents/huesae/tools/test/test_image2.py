# backend/test_doubao_tools.py
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

os.chdir(os.path.dirname(__file__))

from huesaeagents.huesae.tools.doubao import generate_image_by_doubao, generate_images_by_doubao

async def test():
    # 测试单图
    print("测试单图生成...")
    url = await generate_image_by_doubao(
        prompt="一个银发红瞳的少女在樱花树下，比例为16:9",
        size="2K",
        watermark=False
    )
    print(f"单图 URL: {url}")
    
    # 测试组图
    # print("\n测试组图生成...")
    # images = await generate_images_by_doubao(
    #     prompt="生成一组4张连贯的插画，同一庭院一角的四季变迁",
    #     size="2K",
    #     max_images=4,
    #     watermark=False
    # )
    # print(f"组图数量: {len(images)}")

asyncio.run(test())
