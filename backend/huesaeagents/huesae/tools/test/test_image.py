"""图片生成测试文件"""
import asyncio
import base64
import datetime
import os
import sys

# 将项目根目录添加到Python路径
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)
os.chdir(_project_root)

from huesaeagents.huesae.tools.image import (
    generate_image_by_jimeng,
    generate_image_by_doubao,
    generate_images_by_doubao,
)


async def test_jimeng():
    """测试即梦AI文生图"""
    print("=" * 50)
    print("测试即梦AI文生图...")
    print("=" * 50)
    try:
        url = await generate_image_by_jimeng(
            prompt="一个银发红瞳的少女在樱花树下",
            width=1024,
            height=1024,
        )
        print(f"即梦生成成功: {url}")
        return True
    except Exception as e:
        print(f"即梦生成失败: {e}")
        return False


async def test_doubao():
    """测试豆包文生图"""
    print("=" * 50)
    print("测试豆包文生图...")
    print("=" * 50)
    try:
        url = await generate_image_by_doubao(
            prompt="一个银发红瞳的少女在樱花树下，电影感，温暖光线",
            size="2K",
        )
        print(f"豆包生成成功: {url}")
        return True
    except Exception as e:
        print(f"豆包生成失败: {e}")
        return False


async def test_doubao_images():
    """测试豆包文生一组图"""
    print("=" * 50)
    print("测试豆包文生一组图...")
    print("=" * 50)
    try:
        images = await generate_images_by_doubao(
            prompt="生成一组共4张连贯插画，核心为一个银发红瞳的少女在樱花树下，电影感，温暖光线，举起手中的樱花",
            size="2K",
            max_images=4,
        )
        print(f"豆包生成一组图成功，共 {len(images)} 张图片")

        # 保存图片到本地
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(_project_root, "output_images")
        os.makedirs(output_dir, exist_ok=True)

        saved_paths = []
        for i, b64 in enumerate(images):
            filename = f"doubao_images_{timestamp}_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            image_data = base64.b64decode(b64)
            with open(filepath, "wb") as f:
                f.write(image_data)
            saved_paths.append(filepath)
            print(f"  第{i+1}张图片已保存: {filepath}")

        return True
    except Exception as e:
        print(f"豆包生成一组图失败: {e}")
        return False


async def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("图片生成工具测试")
    print("=" * 50 + "\n")

    # jimeng_ok = await test_jimeng()
    print()
    # doubao_ok = await test_doubao()
    print()
    doubao_images_ok = await test_doubao_images()

    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    # print(f"即梦: {'通过' if jimeng_ok else '失败'}")
    # print(f"豆包: {'通过' if doubao_ok else '失败'}")
    print(f"豆包一组图: {'通过' if doubao_images_ok else '失败'}")


if __name__ == "__main__":
    asyncio.run(main())
