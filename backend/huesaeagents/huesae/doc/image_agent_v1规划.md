# HuesaeAgents - 生图功能 Phase 文档

## Context

用户希望优先开发生图功能，需要一份详细的Phase文档。要求：
1. 遵循**开闭原则**：对扩展开放，对修改关闭
2. 开发过程中保留修改记录
3. 模块化设计，便于后续扩展

---

## 一、生图功能架构

### 1.1 核心流程

```
用户输入（自然语言）
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              Image Agent（生图智能体）               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ① 自然语言 → Danbooru标签                         │
│      └─ LLM生成标签                                 │
│      └─ 用户确认/重新生成                           │
│                                                     │
│  ② Danbooru标签 → 生图                             │
│      └─ ComfyUI / Midjourney / 即梦 / 豆包          │
│                                                     │
│  ③ 图片反推标签                                     │
│      └─ 用户上传图片 → 返回Danbooru标签             │
│                                                     │
│  ④ Pixiv爬取                                       │
│      └─ 搜索 → 返回图片URL                          │
│                                                     │
└─────────────────────────────────────────────────────┘
    │
    ▼
返回结果给用户
```

### 1.2 目录结构（遵循开闭原则）

```
backend/huesaeagents/huesae/
├── agents/
│   └── subagents/
│       ├── __init__.py
│       ├── image_agent.py       # 【核心】生图智能体（仅定义接口）
│       │
│       ├── image/               # 【扩展】生图模块（可插拔）
│       │   ├── __init__.py
│       │   ├── base.py          # 生图基类（抽象接口）
│       │   ├── danbooru.py       # Danbooru标签生成
│       │   ├── providers/       # 【扩展点】生图提供者
│       │   │   ├── __init__.py
│       │   │   ├── base.py      # 提供者基类
│       │   │   ├── comfyui.py   # ComfyUI
│       │   │   ├── midjourney.py # Midjourney
│       │   │   ├── jimeng.py    # 即梦AI（已存在）
│       │   │   └── doubao.py     # 豆包Seedream（已存在）
│       │   │
│       │   └── reverser.py      # 图片反推标签
│       │
│       └── pixiv/               # 【扩展】Pixiv爬取
│           ├── __init__.py
│           └── crawler.py
```

---

## 二、功能模块详细设计

### 2.1 生图基类（base.py）

```python
from abc import ABC, abstractmethod
from typing import Protocol

class ImageGenerator(Protocol):
    """生图提供者协议（Protocol用于 duck typing）"""

    @property
    def name(self) -> str:
        """提供者名称"""
        ...

    async def generate(self, prompt: str, **kwargs) -> str:
        """生成图片，返回URL"""
        ...
```

### 2.2 生图流程状态机

```python
class ImageState(TypedDict):
    """生图流程状态"""
    step: str                              # 当前步骤
    user_input: str                        # 用户原始输入
    danbooru_tags: list[str] | None       # 生成的Danbooru标签
    selected_provider: str | None          # 选择的生图提供者
    generated_image_url: str | None        # 生成的图片URL
    confirmed: bool                        # 用户是否确认
    history: list[dict]                    # 操作历史（可回溯）
```

### 2.3 步骤定义

```python
class ImageStep:
    """生图流程步骤常量"""
    INPUT = "input"                        # 用户输入
    TAG_GENERATE = "tag_generate"           # 生成标签
    TAG_CONFIRM = "tag_confirm"            # 确认标签
    IMAGE_GENERATE = "image_generate"       # 生成图片
    FINISH = "finish"                      # 完成
```

---

## 三、Image Agent 设计

### 3.1 核心接口

```python
class ImageAgent:
    """生图智能体"""

    def __init__(self, llm, providers: list[ImageGenerator]):
        self.llm = llm
        self.providers = {p.name: p for p in providers}  # 注入式扩展

    async def process(self, state: ImageState) -> ImageState:
        """处理生图请求"""
        ...

    async def generate_tags(self, user_input: str) -> list[str]:
        """自然语言 → Danbooru标签"""
        ...

    async def generate_image(self, tags: list[str], provider: str) -> str:
        """Danbooru标签 → 图片"""
        ...
```

### 3.2 扩展点设计（开闭原则）

```python
# 扩展方式1：注册新的生图提供者
agent.register_provider(ComfyUIProvider())
agent.register_provider(MidjourneyProvider())

# 扩展方式2：注册新的标签生成器
agent.register_tagger(DanbooruTagger())

# 扩展方式3：注册新的反推器
agent.register_reverser(VLMReverser())
```

---

## 四、Provider 接口设计

### 4.1 基类定义

```python
class BaseImageProvider(ABC):
    """图片生成提供者基类"""

    def __init__(self, config: dict):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """提供者名称"""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        size: str = "1K",
        **kwargs
    ) -> str:
        """生成图片，返回URL"""
        pass

    def validate_prompt(self, prompt: str) -> bool:
        """验证prompt格式（模板方法）"""
        return len(prompt) <= self.max_prompt_length
```

### 4.2 ComfyUI Provider（示例）

```python
class ComfyUIProvider(BaseImageProvider):
    """ComfyUI生图提供者"""

    @property
    def name(self) -> str:
        return "comfyui"

    async def generate(
        self,
        prompt: str,
        size: str = "1K",
        **kwargs
    ) -> str:
        # ComfyUI API 调用实现
        ...
```

---

## 五、实现顺序

### Phase 1：基础设施

| 顺序 | 文件 | 说明 |
|------|------|------|
| 1 | `agents/subagents/image_agent.py` | 生图智能体核心 |
| 2 | `agents/subagents/image/base.py` | 生图接口定义 |
| 3 | `agents/subagents/image/danbooru.py` | Danbooru标签生成 |
| 4 | `agents/subagents/image/providers/base.py` | Provider基类 |

### Phase 2：提供者实现

| 顺序 | 文件 | 说明 |
|------|------|------|
| 5 | `agents/subagents/image/providers/jimeng.py` | 集成已有jimeng |
| 6 | `agents/subagents/image/providers/doubao.py` | 集成已有doubao |
| 7 | `agents/subagents/image/providers/comfyui.py` | ComfyUI |
| 8 | `agents/subagents/image/providers/midjourney.py` | Midjourney |

### Phase 3：扩展功能

| 顺序 | 文件 | 说明 |
|------|------|------|
| 9 | `agents/subagents/image/reverser.py` | 图片反推标签 |
| 10 | `agents/subagents/pixiv/crawler.py` | Pixiv爬取 |

---

## 六、代码模板

### 6.1 Image Agent 核心

```python
# agents/subagents/image_agent.py
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class ImageState(TypedDict):
    """生图流程状态"""
    messages: Annotated[list, add_messages]
    step: str
    user_input: str
    danbooru_tags: list[str] | None
    selected_provider: str | None
    generated_image_url: str | None
    confirmed: bool

class ImageStep:
    INPUT = "input"
    TAG_GENERATE = "tag_generate"
    TAG_CONFIRM = "tag_confirm"
    IMAGE_GENERATE = "image_generate"
    FINISH = "finish"

class ImageAgent:
    """生图智能体"""

    def __init__(self, llm, providers: list["ImageGenerator"] = None):
        self.llm = llm
        self.providers: dict[str, ImageGenerator] = {}
        if providers:
            for p in providers:
                self.register_provider(p)

    def register_provider(self, provider: "ImageGenerator") -> None:
        """注册生图提供者（扩展点）"""
        self.providers[provider.name] = provider

    async def generate_tags(self, user_input: str) -> list[str]:
        """自然语言 → Danbooru标签"""
        prompt = f"""将以下描述转换为Danbooru格式标签：
描述：{user_input}

要求：
1. 使用英文标签
2. 用逗号分隔
3. 包含角色特征、表情、姿势、场景等

直接输出标签，不要解释。"""
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        return [tag.strip() for tag in response.content.split(",")]

    async def generate_image(
        self,
        tags: list[str],
        provider: str = "doubao"
    ) -> str:
        """Danbooru标签 → 图片"""
        if provider not in self.providers:
            raise ValueError(f"Unknown provider: {provider}")

        prompt = ", ".join(tags)
        return await self.providers[provider].generate(prompt)

    async def process(self, state: ImageState) -> ImageState:
        """处理生图请求"""
        step = state["step"]

        if step == ImageStep.INPUT:
            return {"step": ImageStep.TAG_GENERATE}

        elif step == ImageStep.TAG_GENERATE:
            tags = await self.generate_tags(state["user_input"])
            return {
                "step": ImageStep.TAG_CONFIRM,
                "danbooru_tags": tags
            }

        elif step == ImageStep.TAG_CONFIRM:
            if state["confirmed"]:
                return {"step": ImageStep.IMAGE_GENERATE}
            # 用户要求重新生成
            tags = await self.generate_tags(state["user_input"])
            return {"danbooru_tags": tags}

        elif step == ImageStep.IMAGE_GENERATE:
            url = await self.generate_image(
                state["danbooru_tags"],
                state.get("selected_provider", "doubao")
            )
            return {
                "step": ImageStep.FINISH,
                "generated_image_url": url
            }

        return state
```

### 6.2 Provider 基类

```python
# agents/subagents/image/providers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class GenerationResult:
    """生图结果"""
    url: str
    provider: str
    prompt: str
    size: str | None = None

class ImageGenerator(ABC):
    """图片生成器抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """提供者名称"""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        size: str = "1K",
        **kwargs
    ) -> GenerationResult:
        """生成图片"""
        pass

    def validate_prompt(self, prompt: str) -> bool:
        """验证prompt（可被子类重写）"""
        return bool(prompt and len(prompt) <= 2000)
```

### 6.3 Doubao Provider 示例

```python
# agents/subagents/image/providers/doubao.py
from .base import ImageGenerator, GenerationResult

class DoubaoProvider(ImageGenerator):
    """豆包Seedream生图提供者"""

    def __init__(self, api_key: str, model_name: str = "doubao-seedream-5-0-260128"):
        self.api_key = api_key
        self.model_name = model_name

    @property
    def name(self) -> str:
        return "doubao"

    async def generate(
        self,
        prompt: str,
        size: str = "2K",
        **kwargs
    ) -> GenerationResult:
        # 调用豆包API
        from huesae.tools.doubao import generate_image_by_doubao

        url = await generate_image_by_doubao(prompt=prompt, size=size)
        return GenerationResult(url=url, provider=self.name, prompt=prompt, size=size)
```

---

## 七、开闭原则实现

### 7.1 扩展方式

```python
# 1. 添加新的Provider，无需修改现有代码
class MyProvider(ImageGenerator):
    @property
    def name(self) -> str:
        return "my_provider"

    async def generate(self, prompt: str, **kwargs) -> GenerationResult:
        ...

agent.register_provider(MyProvider())

# 2. 替换标签生成器
agent.tagger = MyTagger()

# 3. 添加新的工作流步骤
ImageStep.NEW_STEP = "new_step"
```

### 7.2 禁止修改的原则

- `ImageAgent` 核心逻辑不修改，只通过扩展点添加功能
- `BaseImageProvider` 接口稳定，子类实现即可
- 状态结构 `ImageState` 扩展字段，不删除已有字段

---

## 八、验证方案

```bash
# 1. 激活环境
conda activate HuesaeAgents

# 2. 测试标签生成
python -c "
from agents.subagents.image_agent import ImageAgent
from models.factory import create_chat_model

agent = ImageAgent(llm=create_chat_model('deepseek'))
tags = agent.generate_tags_sync('一个银发红瞳的少女在樱花树下')
print(tags)
"

# 3. 测试完整流程
python -c "
from agents.subagents.image_agent import ImageAgent, ImageState, ImageStep
from models.factory import create_chat_model

agent = ImageAgent(llm=create_chat_model('deepseek'))
agent.register_provider(DoubaoProvider())

state = ImageState(
    messages=[],
    step=ImageStep.INPUT,
    user_input='画一个银发红瞳的少女',
    danbooru_tags=None,
    confirmed=True
)

result = agent.process_sync(state)
print(result)
"
```

---

## 九、修改记录

| 日期 | 修改内容 | 原因 |
|------|---------|------|
| 2026-05-10 | 初始创建 | 规划生图功能Phase |
