# HuesaeAgents 架构文档 — 主Agent与生图Agent

> 文档生成日期：2026/05/13  
> 对应代码版本：commit `b838544`（初步主Agent和生图Agent雏形）

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [主Agent（Lead Agent）](#2-主agentlead-agent)
3. [生图子Agent（Image SubAgent）](#3-生图子agentimage-subagent)
4. [状态管理](#4-状态管理)
5. [工具与Provider层](#5-工具与provider层)
6. [交互流程示例](#6-交互流程示例)
7. [扩展指南](#7-扩展指南)

---

## 1. 整体架构概览

HuesaeAgents 采用 **DeerFlow Harness Engineering** 模式设计：

- **主Agent** 作为对话核心，通过 **ReAct 循环** 让 LLM 自主决策如何处理用户请求
- **子Agent** 作为可委派组件，处理需要多轮对话的专业任务
- **工具选择完全由 LLM 决定**，系统只提供工具列表和描述，无需硬编码分流逻辑

```
┌─────────────────────────────────────────────────────────────┐
│                        用户输入                              │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    HuesaeMainAgent                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ 安全检查    │  │ ReAct 循环  │  │ 子Agent上下文管理   │ │
│  │ (敏感词)    │  │ (最多3步)   │  │ (active_subagent)   │ │
│  └─────────────┘  └──────┬──────┘  └─────────────────────┘ │
└──────────────────────────┼──────────────────────────────────┘
                           ▼
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      ┌─────────┐   ┌──────────┐   ┌────────────┐
      │ 直接回复 │   │ 调用工具  │   │ 委托子Agent │
      └─────────┘   └──────────┘   └─────┬──────┘
                                         ▼
                              ┌─────────────────────┐
                              │    ImageSubAgent    │
                              │  (生图多轮对话)      │
                              └─────────┬───────────┘
                                        ▼
                              ┌─────────────────────┐
                              │   DoubaoProvider    │
                              │  (豆包Seedream API)  │
                              └─────────────────────┘
```

### 核心设计原则

1. **LLM 自主决策**：工具选择由 LLM 根据描述自主决定，不硬编码分类逻辑
2. **新增子Agent = 新增工具**：注册新子Agent即可，无需修改主Agent分流代码
3. **子Agent无状态**：每次调用接收完整对话历史做决策，主Agent负责维护上下文
4. **异步生图分离**：生图作为异步操作，由主Agent统一调度执行

---

## 2. 主Agent（Lead Agent）

### 2.1 文件位置

- `backend/huesaeagents/huesae/agents/lead_agent/lead_agent.py` — 核心类
- `backend/huesaeagents/huesae/agents/lead_agent/prompts.py` — 系统提示词
- `backend/huesaeagents/huesae/agents/lead_agent/chat_loop.py` — 终端交互入口

### 2.2 核心类：`HuesaeMainAgent`

```python
class HuesaeMainAgent:
    MAX_STEPS = 3  # ReAct 循环最大步数
```

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm` | `BaseChatModel` | 必填 | 大语言模型实例 |
| `character_id` | `str` | `"gentle_sister"` | 角色ID，控制回复语气 |

#### 主要方法

| 方法 | 说明 |
|------|------|
| `process(state, user_input)` | 主入口，处理用户输入，返回结果dict |
| `register_sub_agent(agent)` | 注册子Agent到注册表 |
| `execute_image_generation(...)` | 异步执行生图（供chat_loop调用） |

### 2.3 ReAct 循环流程

```
用户输入
    │
    ▼
┌─────────────────┐
│ 1. 安全检查     │ ──→ 命中敏感词 → 返回安全回复
│ (敏感词匹配)    │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 2. 子Agent上下文│ ──→ 存在active_subagent → 直接委托给子Agent
│    检查         │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 3. ReAct 循环   │
│   (最多3步)     │
└─────────────────┘
    │
    ├──→ LLM 输出 Action（结构化JSON）
    │       ├── type="reply" → 直接返回回复
    │       ├── type="tool_call" → 执行工具
    │       │       ├── 生图工具 → 返回 pending_generation 标记
    │       │       ├── 快速工具 → 结果加入上下文，继续循环
    │       │       └── task_tool → 启动子Agent
    │       └── 异常 → Fallback 到聊天回复
    │
    └──→ 超过MAX_STEPS → Fallback 到聊天回复
```

### 2.4 LLM 决策模型：Action

主Agent使用结构化输出让LLM决策：

```python
class Action(BaseModel):
    thought: str          # 分析用户需求
    type: Literal["reply", "tool_call"]
    tool_name: str | None # 工具名称
    tool_args: dict | None # 工具参数
    response: str | None  # 直接回复内容
```

### 2.5 可用工具列表

| 工具名 | 触发场景 | 实际行为 |
|--------|---------|---------|
| `generate_image_tool` | 用户明确要求生成单张图片 | 同步调用豆包API，返回URL |
| `generate_images_tool` | 用户明确要求生成多张图片 | 同步调用豆包组图API |
| `expand_prompt_tool` | 用户要求扩写描述 | 调用LLM扩写提示词 |
| `convert_tags_tool` | 用户要求转成Danbooru标签 | 调用LLM生成标签 |
| `task_tool` | 需求模糊、需多轮对话 | 返回结构化标记，由主Agent启动子Agent |

### 2.6 子Agent委托机制

`task_tool` 不实际执行子Agent，而是返回编码后的任务标记：

```python
# 编码
"__SUBAGENT_TASK__:{subagent_type}:{description}"

# 主Agent解析后启动子Agent
self._start_subagent(state, subagent_type, description)
```

子Agent上下文结构：

```python
{
    "agent_type": "image",
    "agent": agent_instance,
    "state": sub_state,        # 子Agent内部状态
    "history": [...messages],  # 子Agent对话历史
}
```

### 2.7 角色语气系统

支持三种预设角色，通过 `character_id` 切换：

| 角色ID | 名称 | 语气特点 |
|--------|------|---------|
| `gentle_sister` | 温柔系 | 可爱、温暖，适当使用颜文字和动作描述 |
| `tsundere` | 傲娇系 | 口是心非、带点害羞，偶尔露出温柔 |
| `furry_fox` | 兽耳娘 | 可爱活泼，偶尔发出拟声词 |

角色语气通过系统提示词注入，影响：
- 主Agent直接聊天回复
- 生图完成后的包装语生成

### 2.8 安全机制

敏感词列表覆盖中英文自杀/自残相关词汇。命中时返回心理援助热线信息：

```
*轻轻握住你的手*

我在这里陪着你，你不是一个人...

如果你感到痛苦或绝望，请一定要寻求专业帮助：
- 心理危机干预热线：400-161-9995
- 北京心理危机研究与干预中心：010-82951332
- 生命热线：400-821-1215
```

---

## 3. 生图子Agent（Image SubAgent）

### 3.1 文件位置

- `backend/huesaeagents/huesae/agents/subagents/image_agent.py` — 核心类
- `backend/huesaeagents/huesae/agents/subagents/image/prompts.py` — 系统提示词
- `backend/huesaeagents/huesae/agents/subagents/image/expand_prompt.py` — 扩写器
- `backend/huesaeagents/huesae/agents/subagents/image/providers/doubao.py` — 豆包Provider
- `backend/huesaeagents/huesae/agents/subagents/image/providers/base.py` — Provider基类

### 3.2 核心类：`ImageSubAgent`

继承自 `BaseSubAgent`，实现标准化接口。

```python
class ImageSubAgent(BaseSubAgent):
    name = "image"
```

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm` | `BaseChatModel` | 必填 | 大语言模型实例 |
| `providers` | `list[ImageProvider]` | `[DoubaoProvider()]` | 生图Provider列表 |
| `default_provider` | `str` | `"doubao"` | 默认Provider |

### 3.3 LLM 决策模型：ImageDecision

```python
class ImageDecision(BaseModel):
    thought: str
    action: Literal[
        "ask_prompt",   # 追问：缺少提示词
        "recommend",    # 推荐：主动生成推荐提示词
        "expand",       # 扩写：将简短描述扩写
        "ask_confirm",  # 确认：推荐/扩写后询问用户
        "generate",     # 生图：调用Provider生成图片
        "show_image",   # 展示：图片已生成，展示给用户
        "finish",       # 结束：对话完成
    ]
    response: str       # 给用户的回复（二次元语气）
    prompt: str | None  # 当前确认的提示词
    size: str | None    # 图片尺寸
    output_format: str | None  # 输出格式
    is_batch: bool | None      # 是否组图模式
```

### 3.4 生图对话工作流

```
用户说"我想生成图片"
    │
    ▼
┌─────────────┐
│ ask_prompt  │ ──→ "请告诉我您想要生成什么样的图片？"
└─────────────┘
    │
    ▼
用户描述需求
    │
    ▼
┌─────────────┐     ┌─────────────┐
│  描述太短   │ ──→ │   expand    │ ──→ 调用扩写器 ──→ ask_confirm
│  (少于6字)  │     │  (用户要求) │
└─────────────┘     └─────────────┘
    │
    ▼
用户说"帮我推荐"
    │
    ▼
┌─────────────┐
│  recommend  │ ──→ 生成1-3个推荐提示词 ──→ ask_confirm
└─────────────┘
    │
    ▼
┌─────────────┐
│ ask_confirm │ ──→ "这个描述可以吗？需要修改哪里吗？"
└─────────────┘
    │
    ▼
用户确认"可以"
    │
    ▼
┌─────────────┐
│  generate   │ ──→ 返回 pending_generation 标记
└─────────────┘      (实际生图由主Agent异步执行)
    │
    ▼
图片生成完成
    │
    ▼
用户说"换一张" ──→ generate (用上次prompt重生成)
用户说"扩写"   ──→ expand
用户说"换一个" ──→ ask_prompt
用户说"不用了" ──→ finish
```

### 3.5 关键行为规则

| 场景 | 行为 |
|------|------|
| 用户确认后（`image_task_type=generate_image`） | 必须返回 `generate`，绝不能 `finish` |
| 用户确认后（`image_task_type=expand_prompt`） | 返回 `finish`，表示扩写任务完成 |
| 用户未要求扩写 | **绝不自动扩写** |
| 用户明确说生成数量（如"4张"） | `is_batch=true`，使用组图模式 |
| 重新生成时历史prompt含数量描述 | 保持组图模式 |
| 默认风格 | 添加"二次元动漫风格"前缀（除非用户要求真人/写实） |

### 3.6 提示词扩写器

`expand_prompt.py` 将简短描述扩写为丰富的自然语言：

- 保留核心元素（角色、场景、动作）
- 增加光线、氛围、视角、情绪、材质等细节
- 输出为自然语言，非标签格式
- 长度控制在100字以内

### 3.7 Danbooru 标签转换

将中文描述转换为高质量Danbooru标签：

- 英文标签，逗号分隔
- 包含维度：角色特征、表情动作、服装、场景环境、光线氛围、画风
- 自动添加质量标签：`masterpiece, best quality, highly detailed`

---

## 4. 状态管理

### 4.1 文件位置

- `backend/huesaeagents/huesae/agents/state/huesae_state.py` — 状态类
- `backend/huesaeagents/huesae/agents/state_manager.py` — 状态管理器

### 4.2 HuesaeState

```python
class HuesaeState:
    messages: list           # 主对话历史（LangChain Message对象）
    active_subagent: dict    # 当前活跃的子Agent上下文
```

### 4.3 StateManager

仅内存存储，不持久化到文件：

```python
class StateManager:
    _states: dict[str, HuesaeState]  # 多会话管理

    def get_state(session_id) -> HuesaeState  # 获取/创建状态
    def save_state(session_id) -> None        # 空操作（内存存储）
    def clear_state(session_id) -> None       # 清除状态
```

---

## 5. 工具与Provider层

### 5.1 Provider 抽象

```python
@dataclass
class GenerationResult:
    url: str
    provider: str
    prompt: str
    size: str | None

class ImageProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str

    @abstractmethod
    async def generate(self, prompt, size="2K", **kwargs) -> GenerationResult
```

### 5.2 豆包Provider（DoubaoProvider）

对接豆包Seedream图片生成API。

**底层客户端**：`backend/huesaeagents/huesae/tools/doubao/client.py`

| 方法 | 说明 |
|------|------|
| `generate_image()` | 单图生成，返回URL |
| `generate_images()` | 组图生成（非流式），最多12张 |
| `generate_images_stream()` | 组图生成（流式），逐张yield URL |

**API 参数**：

| 参数 | 说明 |
|------|------|
| `model` | `doubao-seedream-5-0-260128`（默认） |
| `size` | 支持 `1K`, `2K`, `3K`, `4K` |
| `output_format` | `jpeg` 或 `png` |
| `watermark` | 是否添加水印 |
| `sequential_image_generation` | 组图模式，`auto` |
| `sequential_image_generation_options.max_images` | 最大生成数量 |

**环境变量**：

```bash
ARK_API_KEY=your_api_key
DOUBAO_SEEDREAM_MODEL_NAME=doubao-seedream-5-0-260128
```

### 5.3 单图 vs 组图

| 模式 | 触发条件 | 调用方法 |
|------|---------|---------|
| 单图 | 用户未说明数量，或明确说"1张" | `generate_image()` |
| 组图 | 用户明确说数量（如"4张"） | `generate_images()`，`sequential_image_generation: auto` |

---

## 6. 交互流程示例

### 6.1 直接生图（单轮）

```
用户: 生成一张夕阳下的大海

主Agent ReAct:
  thought: "用户明确要求生成单张图片，prompt明确"
  type: "tool_call"
  tool_name: "generate_image_tool"
  tool_args: {"prompt": "夕阳下的大海", "size": "2K"}

→ 返回 pending_generation
→ chat_loop 调用 execute_image_generation()
→ 异步生图完成

AI: 这是生成好的图片哦~（温柔语气包装语）
    [图片] https://...
```

### 6.2 多轮生图（子Agent）

```
用户: 我想生成图片

主Agent:
  thought: "用户想生图但没有具体描述，需要多轮对话"
  type: "tool_call"
  tool_name: "task_tool"

→ 启动 ImageSubAgent

AI (子Agent): 请告诉我您想要生成什么样的图片？
              可以描述一下角色、场景、风格等~，
              图像格式可以选择png,jpeg，图片尺寸可以选择2K,3K,4K

用户: 一个猫娘

子Agent:
  action: "ask_confirm"
  thought: "描述较短，但用户没有要求扩写，直接确认"
  response: "这个描述可以吗？需要修改哪里吗？"

用户: 可以

子Agent:
  action: "generate"
  prompt: "图片风格为 二次元，一个猫娘"

→ 返回 pending_generation
→ 异步生图

AI: 图片生成好啦~ 快来看看吧~
    [图片] https://...

用户: 换一张

子Agent:
  action: "generate"  (用上次确认的prompt重生成)
```

### 6.3 扩写流程

```
用户: 扩写：一个少女在樱花树下

主Agent:
  tool_name: "expand_prompt_tool"

→ 直接调用扩写器返回结果

AI: 扩写结果：一位银发如瀑布般倾泻的少女...
```

---

## 7. 扩展指南

### 7.1 新增子Agent

1. 继承 `BaseSubAgent`，实现 `process()` 和 `name` 属性
2. 定义子Agent的决策模型（Pydantic BaseModel）
3. 在主Agent中注册：

```python
main_agent.register_sub_agent(MyNewAgent())
```

### 7.2 新增生图Provider

1. 继承 `ImageProvider`，实现 `name` 和 `generate()`
2. 在创建ImageSubAgent时传入：

```python
from huesae.agents.subagents.image.providers import ImageProvider

class MyProvider(ImageProvider):
    @property
    def name(self): return "my_provider"

    async def generate(self, prompt, size="2K", **kwargs):
        # 调用第三方API
        return GenerationResult(url=url, provider=self.name, prompt=prompt)

agent = create_image_agent(providers=[MyProvider(), DoubaoProvider()])
```

### 7.3 新增角色语气

在 `lead_agent/prompts.py` 中添加：

```python
CHARACTER_TONE_NEW = "你是..."

tone_map = {
    "gentle_sister": CHARACTER_TONE_GENTLE,
    "new_role": CHARACTER_TONE_NEW,
}
```

---

## 附录：关键文件清单

| 文件 | 职责 |
|------|------|
| `agents/lead_agent/lead_agent.py` | 主Agent核心类 |
| `agents/lead_agent/prompts.py` | 主Agent系统提示词、角色语气 |
| `agents/lead_agent/chat_loop.py` | 终端交互入口 |
| `agents/subagents/image_agent.py` | 生图子Agent核心类 |
| `agents/subagents/image/prompts.py` | 生图对话系统提示词 |
| `agents/subagents/image/expand_prompt.py` | 提示词扩写器 |
| `agents/subagents/image/providers/base.py` | Provider抽象基类 |
| `agents/subagents/image/providers/doubao.py` | 豆包Provider实现 |
| `agents/subagents/base.py` | 子Agent基类 |
| `agents/state/huesae_state.py` | 对话状态类 |
| `agents/state_manager.py` | 状态管理器（内存） |
| `subagents/registry.py` | 子Agent注册表 |
| `tools/tools.py` | 主Agent工具工厂 |
| `tools/doubao/client.py` | 豆包API客户端 |
