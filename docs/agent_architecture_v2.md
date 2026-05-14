# HuesaeAgents 架构文档 — 主Agent与生图Agent

> 文档生成日期：2026/05/14
> 对应代码版本：commit `6f1ff17`（清理旧机制后）
> 技术栈：LangChain 1.2.17 + LangGraph 1.1.10

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [主Agent（Lead Agent）](#2-主agentlead-agent)
3. [生图子Agent（Image SubAgent）](#3-生图子agentimage-subagent)
4. [状态管理](#4-状态管理)
5. [工具层](#5-工具层)
6. [生图Provider层](#6-生图provider层)
7. [子Agent注册表](#7-子agent注册表)
8. [终端交互入口](#8-终端交互入口)
9. [扩展指南](#9-扩展指南)

---

## 1. 整体架构概览

HuesaeAgents 采用 **DeerFlow Harness Engineering** 模式设计，核心架构为：

```
用户输入 → 主Agent (ReAct循环) → [直接回复 | 调用工具 | 委托子Agent]
```

### 架构图

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

### 目录结构

```
backend/huesaeagents/huesae/
├── agents/
│   ├── lead_agent/
│   │   ├── lead_agent.py     # 主Agent核心实现
│   │   ├── chat_loop.py      # 终端交互入口
│   │   └── prompts.py        # 角色语气提示词
│   ├── state/
│   │   └── huesae_state.py   # 状态类定义
│   └── state_manager.py      # 状态管理器（内存版）
├── models/
│   ├── models_factory.py     # 模型工厂
│   └── providers/
│       └── deepseek.py       # DeepSeek模型
├── subagents/
│   ├── base.py               # 子Agent基类
│   ├── image_agent.py        # 生图子Agent
│   ├── registry.py           # 子Agent注册表
│   └── image/                # 生图模块
│       ├── prompts.py        # 生图对话提示词
│       ├── expand_prompt.py  # 提示词扩写
│       ├── danbooru.py       # Danbooru标签生成
│       └── providers/        # 生图Provider
│           ├── base.py       # Provider抽象基类
│           ├── doubao.py     # 豆包Provider
│           └── jimeng.py     # 即梦Provider
└── tools/
    ├── tools.py              # 工具定义（ReAct）
    └── doubao/               # 豆包API客户端
```

### 核心设计原则

| 原则 | 说明 |
|------|------|
| **LLM 自主决策** | 工具选择由 LLM 根据描述自主决定，不硬编码分类逻辑 |
| **新增子Agent = 新增工具** | 注册新子Agent即可，无需修改主Agent分流代码 |
| **子Agent无状态** | 每次调用接收完整对话历史做决策，主Agent负责维护上下文 |
| **异步生图分离** | 生图作为异步操作，由主Agent统一调度执行 |
| **角色系统** | 支持多种角色语气（温柔姐姐、傲娇、兽耳娘） |

---

## 2. 主Agent（Lead Agent）

### 2.1 文件位置

- [backend/huesaeagents/huesae/agents/lead_agent/lead_agent.py](backend/huesaeagents/huesae/agents/lead_agent/lead_agent.py) — 核心类
- [backend/huesaeagents/huesae/agents/lead_agent/prompts.py](backend/huesaeagents/huesae/agents/lead_agent/prompts.py) — 角色语气提示词
- [backend/huesaeagents/huesae/agents/lead_agent/chat_loop.py](backend/huesaeagents/huesae/agents/lead_agent/chat_loop.py) — 终端交互入口

### 2.2 核心类：`HuesaeMainAgent`

```python
class HuesaeMainAgent:
    """主Agent：LLM 自主工具选择的 ReAct 循环"""

    MAX_STEPS = 3  # ReAct 循环最大步数

    def __init__(self, llm: BaseChatModel, character_id: str = "gentle_sister"):
        self.llm = llm
        self.character_id = character_id
        self.subagent_registry = SubAgentRegistry()
        self.tools = []
        self.tool_map = {}
        self._refresh_tools()
```

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm` | `BaseChatModel` | 必填 | 大语言模型实例 |
| `character_id` | `str` | `"gentle_sister"` | 角色ID，控制回复语气 |

#### 主要方法

| 方法 | 说明 |
|------|------|
| `process(state, user_input)` | 主入口，ReAct循环处理用户输入 |
| `_build_system_prompt()` | 构建含工具描述的系统提示词 |
| `_execute_tool(tool_name, tool_args)` | 执行指定工具 |
| `_start_subagent(state, subagent_type, description)` | 启动子Agent |
| `_handle_subagent(state, user_input)` | 继续子Agent对话 |
| `execute_image_generation(...)` | 异步执行生图（供chat_loop调用） |
| `_chat_reply(state, user_input)` | 直接聊天回复 |
| `_check_safety(user_input)` | 安全检查 |

### 2.3 ReAct 循环流程

`process()` 方法是主Agent的核心入口，流程如下：

```
1. 安全检查（最高优先级）
   └─ 检测到敏感词 → 返回安全回复

2. 子Agent上下文检查
   └─ 如果 active_subagent 存在 → 直接委托给子Agent

3. ReAct 循环（最多3步）
   a. 构建系统提示词（含工具描述 + 角色语气）
   b. 调用LLM获取 Action（结构化输出：reply/tool_call）
   c. 如果是 reply → 直接返回回复
   d. 如果是 tool_call → 执行工具
      ├─ 生图工具 → 返回 pending_generation，由外层异步执行
      ├─ task_tool → 启动子Agent
      └─ 其他工具 → 结果加入上下文，继续循环

4. 超过最大步数 → Fallback到聊天回复
```

#### LLM决策模型：`Action`

定义在 [tools.py](backend/huesaeagents/huesae/tools/tools.py) 中：

```python
class Action(BaseModel):
    thought: str          # 分析用户需求
    type: Literal["reply", "tool_call"]  # 行动类型
    tool_name: str | None # 工具名称
    tool_args: dict | None# 工具参数
    response: str | None  # 直接回复内容
```

### 2.4 子Agent委托机制

#### 启动子Agent (`_start_subagent`)

当 `task_tool` 被调用时，主Agent：
1. 从注册表获取对应子Agent
2. 创建子Agent的初始状态
3. 调用子Agent的 `process()` 方法
4. 根据子Agent返回的 action 决定下一步：
   - `ask_prompt/recommend/ask_confirm` → 继续对话，保存上下文
   - `generate` → 标记 pending_generation，异步生图
   - `finish` → 清除子Agent上下文

#### 继续子Agent对话 (`_handle_subagent`)

当用户处于子Agent上下文中时：
1. 从 `active_subagent` 恢复子Agent状态和历史
2. 将用户输入追加到子Agent历史
3. 调用子Agent `process()`
4. 更新历史并返回结果

### 2.5 安全机制

```python
_SAFE_KEYWORDS = [
    "自杀", "自残", "想死", "不想活", "结束生命", "活着没意思",
    "kill myself", "suicide", "self-harm",
]
```

检测到敏感词时，立即返回关怀回复（包含心理危机干预热线）。

### 2.6 角色系统

支持三种角色语气：

| 角色ID | 名称 | 语气特点 |
|--------|------|----------|
| `gentle_sister` | 温柔姐姐 | 可爱、温暖，适当使用颜文字和动作描述 |
| `tsundere` | 傲娇 | 口是心非、带点害羞，偶尔露出温柔 |
| `furry_fox` | 兽耳娘 | 可爱、活泼，偶尔发出拟声词 |

---

## 3. 生图子Agent（Image SubAgent）

### 3.1 文件位置

- [backend/huesaeagents/huesae/subagents/image_agent.py](backend/huesaeagents/huesae/subagents/image_agent.py) — 核心类
- [backend/huesaeagents/huesae/subagents/image/prompts.py](backend/huesaeagents/huesae/subagents/image/prompts.py) — 生图对话系统提示词
- [backend/huesaeagents/huesae/subagents/base.py](backend/huesaeagents/huesae/subagents/base.py) — 子Agent基类

### 3.2 核心类：`ImageSubAgent`

```python
class ImageSubAgent(BaseSubAgent):
    """生图子Agent - 无状态组件，每次调用接收完整对话历史"""

    name = "image"

    def __init__(self, llm, providers=None, default_provider="doubao"):
        self.llm = llm
        self.providers = {}
        if providers:
            for p in providers:
                self.register_provider(p)
        self.default_provider = default_provider
```

### 3.3 LLM决策模型：`ImageDecision`

```python
class ImageDecision(BaseModel):
    thought: str
    action: Literal[
        "ask_prompt",   # 追问：缺少提示词
        "recommend",    # 推荐：主动生成推荐提示词
        "expand",       # 扩写：将简短描述扩写
        "ask_confirm",  # 确认：推荐/扩写后询问
        "generate",     # 生图：调用provider生成
        "show_image",   # 展示：图片已生成
        "finish",       # 结束：对话完成
    ]
    response: str
    prompt: str | None
    provider: str | None
    size: str | None = "2K"
    output_format: str | None = "jpeg"
    is_batch: bool | None = False
```

### 3.4 工作流程

```
1. 了解需求
   └─ 用户没有提供具体描述 → 追问

2. 主动推荐
   └─ 用户要求推荐 → 生成1-3个推荐提示词

3. 智能扩写（绝不自动扩写）
   └─ 只有用户明确要求时才执行扩写

4. 确认闭环
   └─ 推荐/扩写后必须询问是否满意
   └─ 根据 image_task_type 决定下一步：
      ├─ generate_image → 确认后必须生图
      ├─ expand_prompt → 确认后返回 finish
      └─ convert_tags → 确认后返回 finish

5. 执行生图
   └─ 用户确认后调用provider
   └─ 组图判断：用户明确说数量 → is_batch=true

6. 图片展示后处理
   ├─ "换一张" → generate（使用上次提示词）
   ├─ "扩写" → expand
   ├─ "修改" → ask_prompt
   └─ "不用了" → finish

7. 结束对话
   └─ 只有用户明确说"不用了""结束"时才 finish
```

### 3.5 风格处理

```python
@staticmethod
def _ensure_anime_style(prompt: str) -> str:
    """确保提示词包含动漫风格前缀"""
    if not prompt:
        return prompt
    lower = prompt.lower()
    # 用户明确要求非动漫风格
    if any(kw in lower for kw in ["真人", "写实", "照片", "photorealistic", "realistic", "real person"]):
        return prompt
    # 默认添加动漫风格前缀
    return f"图片风格为 二次元，{prompt}"
```

### 3.6 标准化返回格式

所有子Agent返回标准化的 dict：

```python
{
    "action": str,       # 动作类型
    "response": str,     # 给用户的回复
    "prompt": str|None,  # 确认的提示词
    "provider": str|None,# 选择的生图工具
    "data": dict,        # 额外数据（size, output_format, is_batch等）
}
```

---

## 4. 状态管理

### 4.1 `HuesaeState` — 状态类

[backend/huesaeagents/huesae/agents/state/huesae_state.py](backend/huesaeagents/huesae/agents/state/huesae_state.py)

```python
class HuesaeState:
    def __init__(self):
        self.messages: list = []           # 主对话历史
        self.active_subagent: dict | None = None  # 当前活跃子Agent上下文

    def to_dict(self) -> dict:
        return {
            "messages": self.messages,
            "active_subagent": self.active_subagent,
        }

    def clear_subagent(self) -> None:
        self.active_subagent = None
```

### 4.2 `StateManager` — 状态管理器

[backend/huesaeagents/huesae/agents/state_manager.py](backend/huesaeagents/huesae/agents/state_manager.py)

```python
class StateManager:
    """状态管理器（仅内存）"""

    def __init__(self):
        self._states: dict[str, HuesaeState] = {}

    def get_state(self, session_id: str = "default") -> HuesaeState:
        if session_id not in self._states:
            self._states[session_id] = HuesaeState()
        return self._states[session_id]
```

> 当前为内存存储，退出后状态丢失。`save_state()` 为空操作，预留持久化扩展点。

---

## 5. 工具层

### 5.1 工具定义

[backend/huesaeagents/huesae/tools/tools.py](backend/huesaeagents/huesae/tools/tools.py)

#### 可用工具列表

| 工具名 | 功能 | 使用场景 |
|--------|------|----------|
| `generate_image_tool` | 生成单张图片 | 用户明确要求生成1张图片 |
| `generate_images_tool` | 生成组图（多张） | 用户明确要求生成多张图片 |
| `expand_prompt_tool` | 扩写图片提示词 | 用户要求扩写、丰富描述 |
| `convert_tags_tool` | 转成Danbooru标签 | 用户要求生成Danbooru标签 |
| `task_tool` | 委托子Agent | 需要多轮对话的复杂任务 |

#### 子Agent任务编码

```python
SUBAGENT_TASK_PREFIX = "__SUBAGENT_TASK__"

def encode_subagent_task(subagent_type: str, description: str) -> str:
    return f"{SUBAGENT_TASK_PREFIX}:{subagent_type}:{description}"

def parse_subagent_task(result: str) -> tuple[str, str] | None:
    if not result.startswith(SUBAGENT_TASK_PREFIX):
        return None
    parts = result.split(":", 2)
    return parts[1], parts[2]
```

---

## 6. 生图Provider层

### 6.1 Provider抽象基类

[backend/huesaeagents/huesae/subagents/image/providers/base.py](backend/huesaeagents/huesae/subagents/image/providers/base.py)

```python
@dataclass
class GenerationResult:
    url: str
    provider: str
    prompt: str
    size: str | None = None

class ImageProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: pass

    @abstractmethod
    async def generate(self, prompt: str, size: str = "2K", **kwargs) -> GenerationResult: pass
```

### 6.2 豆包Provider

[backend/huesaeagents/huesae/subagents/image/providers/doubao.py](backend/huesaeagents/huesae/subagents/image/providers/doubao.py)

```python
class DoubaoProvider(ImageProvider):
    def __init__(self, api_key=None, model_name=None):
        self.api_key = api_key or os.getenv("DOUBAO_SEEDREAM_API_KEY")
        self.model_name = model_name or os.getenv("DOUBAO_SEEDREAM_MODEL_NAME", "doubao-seedream-5-0-260128")

    @property
    def name(self) -> str:
        return "doubao"

    async def generate(self, prompt: str, size: str = "2K", output_format: str = "jpeg", **kwargs) -> GenerationResult:
        client = create_doubao_client()
        url = await asyncio.to_thread(
            client.generate_image,
            prompt=prompt,
            size=size,
            output_format=output_format,
        )
        return GenerationResult(url=url, provider=self.name, prompt=prompt, size=size)
```

### 6.3 即梦Provider

[backend/huesaeagents/huesae/subagents/image/providers/jimeng.py](backend/huesaeagents/huesae/subagents/image/providers/jimeng.py)

```python
class JimengProvider(ImageProvider):
    def __init__(self, access_key=None, secret_key=None):
        self.access_key = access_key or os.getenv("JIMENG_ACCESS_KEY_ID")
        self.secret_key = secret_key or os.getenv("JIMENG_SECRET_ACCESS_KEY")

    @property
    def name(self) -> str:
        return "jimeng"

    async def generate(self, prompt: str, size: str = "2K", **kwargs) -> GenerationResult:
        size_map = {"1K": (1024, 1024), "2K": (2048, 2048), "4K": (4096, 4096)}
        width, height = size_map.get(size, (2048, 2048))
        client = create_jimeng_client()
        image_urls = await asyncio.to_thread(client.generate_image, prompt, width, height)
        ...
```

---

## 7. 子Agent注册表

[backend/huesaeagents/huesae/subagents/registry.py](backend/huesaeagents/huesae/subagents/registry.py)

```python
class SubAgentRegistry:
    """管理当前运行时可用的子Agent"""

    def __init__(self):
        self._agents: dict[str, RegisteredSubAgent] = {}
        self._descriptions: dict[str, str] = {}

    def register(self, agent: RegisteredSubAgent, description: str | None = None) -> None:
        self._agents[agent.name] = agent
        self._descriptions[agent.name] = description or self._default_description(agent)

    def get(self, name: str) -> RegisteredSubAgent | None:
        return self._agents.get(name)

    def names(self) -> list[str]:
        return list(self._agents.keys())

    def format_for_prompt(self) -> str:
        return "\n".join(f"- {info.name}: {info.description}" for info in self.infos())
```

---

## 8. 终端交互入口

[backend/huesaeagents/huesae/agents/lead_agent/chat_loop.py](backend/huesaeagents/huesae/agents/lead_agent/chat_loop.py)

### 8.1 运行流程

```python
def run_chat_loop():
    # 1. 创建主Agent并注册子Agent
    main_agent = create_main_agent()
    main_agent.register_sub_agent(create_image_agent())

    # 2. 初始化状态管理器
    state_manager = StateManager()
    session_id = "terminal_user"
    conv_state = state_manager.get_state(session_id)

    while True:
        user_input = input("用户: ").strip()

        # 3. 构建state
        state = {
            "messages": conv_state.messages,
            "active_subagent": conv_state.active_subagent,
        }

        # 4. 调用主Agent
        result = main_agent.process(state, user_input)

        # 5. 处理pending_generation（异步生图）
        if result.get("pending_generation"):
            image_result = asyncio.run(main_agent.execute_image_generation(...))
            ...

        # 6. 更新状态
        conv_state.messages.append(HumanMessage(content=user_input))
        conv_state.messages.extend(result.get("messages", []))
        state_manager.save_state(session_id)
```

### 8.2 打字机效果

```python
def print_stream(text: str, prefix: str = "AI: ", delay: float = 0.025) -> None:
    print(prefix, end="", flush=True)
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()
```

---

## 9. 扩展指南

### 9.1 新增子Agent

1. 继承 `BaseSubAgent`，实现 `name` 属性和 `process()` 方法
2. 在 `HuesaeMainAgent.register_sub_agent()` 中添加描述
3. 在 `chat_loop.py` 中注册实例

```python
class MySubAgent(BaseSubAgent):
    name = "my_agent"

    def process(self, state: dict, user_input: str) -> dict:
        return {
            "action": "finish",
            "response": "处理完成",
            "prompt": None,
            "provider": None,
            "data": {},
        }
```

### 9.2 新增生图Provider

1. 继承 `ImageProvider`，实现 `name` 属性和 `generate()` 方法
2. 在创建 `ImageSubAgent` 时传入

```python
class MyProvider(ImageProvider):
    @property
    def name(self) -> str:
        return "my_provider"

    async def generate(self, prompt: str, size: str = "2K", **kwargs) -> GenerationResult:
        # 实现生图逻辑
        return GenerationResult(url="...", provider=self.name, prompt=prompt, size=size)
```

### 9.3 新增工具

在 [tools.py](backend/huesaeagents/huesae/tools/tools.py) 的 `create_tools()` 函数中添加：

```python
@tool
def my_tool(param: str) -> str:
    """工具描述（LLM据此决策是否调用）"""
    return f"结果：{param}"

registry.register(my_tool)
```

### 9.4 新增角色

在 [prompts.py](backend/huesaeagents/huesae/agents/lead_agent/prompts.py) 中添加：

```python
CHARACTER_TONE_NEW = "你是一位..."

tone_map = {
    "gentle_sister": CHARACTER_TONE_GENTLE,
    "tsundere": CHARACTER_TONE_TSUNDERE,
    "furry_fox": CHARACTER_TONE_FURRY,
    "new_role": CHARACTER_TONE_NEW,
}
```

---

## 附录：核心文件速查

| 功能 | 文件路径 |
|------|----------|
| 主Agent | `backend/huesaeagents/huesae/agents/lead_agent/lead_agent.py` |
| 角色提示词 | `backend/huesaeagents/huesae/agents/lead_agent/prompts.py` |
| 终端入口 | `backend/huesaeagents/huesae/agents/lead_agent/chat_loop.py` |
| 状态定义 | `backend/huesaeagents/huesae/agents/state/huesae_state.py` |
| 状态管理器 | `backend/huesaeagents/huesae/agents/state_manager.py` |
| 工具定义 | `backend/huesaeagents/huesae/tools/tools.py` |
| 子Agent基类 | `backend/huesaeagents/huesae/subagents/base.py` |
| 生图Agent | `backend/huesaeagents/huesae/subagents/image_agent.py` |
| 生图提示词 | `backend/huesaeagents/huesae/subagents/image/prompts.py` |
| 子Agent注册表 | `backend/huesaeagents/huesae/subagents/registry.py` |
| Provider基类 | `backend/huesaeagents/huesae/subagents/image/providers/base.py` |
| 豆包Provider | `backend/huesaeagents/huesae/subagents/image/providers/doubao.py` |
| 即梦Provider | `backend/huesaeagents/huesae/subagents/image/providers/jimeng.py` |
| 模型工厂 | `backend/huesaeagents/huesae/models/models_factory.py` |
| DeepSeek模型 | `backend/huesaeagents/huesae/models/providers/deepseek.py` |
