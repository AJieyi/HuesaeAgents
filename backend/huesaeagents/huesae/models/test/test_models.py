"""
测试 models 模块

使用方法:
1. 激活虚拟环境: conda activate HuesaeAgents
2. 安装依赖: pip install langchain langchain-core langchain-deepseek python-dotenv
3. 在项目根目录创建 .env 文件: DEEPSEEK_API_KEY=your-key
4. 运行: python huesae/models/test/test_models.py
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

# 添加项目路径（指向 backend 目录，使 huesaeagents 可作为包导入）
# 文件: backend/huesaeagents/huesae/models/test/test_models.py
# 往上5层: test -> models -> huesae -> huesaeagents -> backend
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

# 加载 .env 文件
load_dotenv()

from huesaeagents.huesae.models.models_factory import create_chat_model
from huesaeagents.huesae.models.providers import create_deepseek_model
from langchain_core.messages import HumanMessage


def test_sync_invoke():
    """测试同步调用方法"""
    print("\n1. 测试同步invoke方法")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("   跳过：需要有效的DEEPSEEK_API_KEY")
        return

    try:
        model = create_deepseek_model(model="deepseek-v4-flash", temperature=0.7)
        print("   正在调用 DeepSeek API (同步)...")
        response = model.invoke("你好，你是什么模型?是deepseek-v4-flash吗？")
        print(f"   Response: {response.content}")
        print("   ✅ 同步invoke测试成功")
    except Exception as e:
        print(f"   ❌ 同步invoke失败: {e}")
        print(f"   错误类型: {type(e).__name__}")


async def test_async_invoke():
    """测试异步调用方法"""
    print("\n2. 测试异步ainvoke方法")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("   跳过：需要有效的DEEPSEEK_API_KEY")
        return

    try:
        model = create_deepseek_model(model="deepseek-v4-flash", temperature=0.7)
        print("   正在调用 DeepSeek API (异步)...")
        response = await model.ainvoke([HumanMessage(content="你好，你是谁？")])
        print(f"   Response: {response.content}")
        print("   ✅ 异步ainvoke测试成功")
    except Exception as e:
        print(f"   ❌ 异步ainvoke失败: {e}")
        print(f"   错误类型: {type(e).__name__}")


def test_factory():
    """测试工厂函数 create_chat_model"""
    print("\n3. 测试create_chat_model工厂函数")
    try:
        model = create_chat_model(provider="deepseek", model="deepseek-v4-flash")
        print(f"   创建模型: {type(model).__name__}")
        response = model.invoke("你好")
        print(f"   Response: {response.content}")
        print("   ✅ create_chat_model测试成功")
    except Exception as e:
        print(f"   ❌ create_chat_model失败: {e}")


if __name__ == "__main__":
    test_sync_invoke()
    # asyncio.run(test_async_invoke())
    # test_factory()