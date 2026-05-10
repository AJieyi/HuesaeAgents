# Agent 集成图片生成能力计划

## Context

用户已完成豆包、即梦的图片生成工具开发，现在需要将这些工具集成到 Agent 工作流中。

**注意：** 现有 `graph.py` 后续会移至新文件夹，当前专注于图片生成工具的 Agent 集成。

---

## 实现方案

使用 LangGraph 推荐的 `create_react_agent` 模式，将图片生成工具绑定到 Agent。

### 实现步骤

#### 步骤 1：定义图片生成工具为 LangChain Tool

**文件：** `backend/huesaeagents/huesae/agents/tools.py`（新增）

```python
from langchain_core.tools import tool
from ..tools.image import generate_image_by_jimeng, generate_image_by_doubao, generate_images_by_doubao

@tool
async def generate_image_jimeng(prompt: str, width: int = 1024, height: int = 1024) -> str:
    """生成图片（即梦AI）"""
    return await generate_image_by_jimeng(prompt=prompt, width=width, height=height)

@tool
async def generate_image_doubao(prompt: str, size: str = "2K") -> str:
    """生成图片（豆包）"""
    return await generate_image_by_doubao(prompt=prompt, size=size)

@tool
async def generate_images_doubao(prompt: str, size: str = "2K", max_images: int = 4) -> str:
    """生成一组图片（豆包，最多4张）"""
    return await generate_images_by_doubao(prompt=prompt, size=size, max_images=max_images)
```

#### 步骤 2：在 Agent 工厂中集成工具

**文件：** `backend/huesaeagents/huesae/agents/factory.py`

```python
from .tools import generate_image_jimeng, generate_image_doubao, generate_images_doubao

def create_huesae_agent(model: BaseChatModel | None = None, **kwargs) -> Any:
    if model is None:
        from ..models.factory import create_chat_model
        model = create_chat_model("deepseek")

    tools = [generate_image_jimeng, generate_image_doubao, generate_images_doubao]
    agent = create_react_agent(model, tools)
    return agent
```

#### 步骤 3：新增测试用例

**文件：** `backend/huesaeagents/huesae/agents/test/test_agents.py`

---

## 关键文件

| 文件 | 操作 |
|------|------|
| `backend/huesaeagents/huesae/agents/tools.py` | 新增 - 工具定义 |
| `backend/huesaeagents/huesae/agents/factory.py` | 修改 - 绑定工具到 Agent |
| `backend/huesaeagents/huesae/agents/test/test_agents.py` | 修改 - 新增测试用例 |

## 验证方式

```python
# 测试 Agent 调用图片生成
from agents.factory import create_huesae_agent

agent = create_huesae_agent()
result = agent.invoke({
    "messages": [HumanMessage(content="画一个银发红瞳的少女")],
})
# Agent 应自动调用 generate_image_doubao 工具并返回结果
```