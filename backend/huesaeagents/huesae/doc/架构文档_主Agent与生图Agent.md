# HuesaeAgents 架构文档：主Agent与生图Agent

## 一、架构总览

```
用户输入 → 主Agent(HuesaeMainAgent) → 意图分类
                    │
        ┌───────────┴───────────┐
        │                       │
   intent=chat              intent=image
        │                       │
   直接聊天回复                ImageSubAgent
        │                       │
        │              ┌────────┴────────┐
        │              │                 │
        │         LLM决策(_decide)   生图执行
        │              │                 │
        │         [ask_prompt]      generate_image()
        │         [recommend]       generate_images()
        │         [expand]                │
        │         [ask_confirm]           │
        │         [generate] ←────────────┘
        │         [finish]
        │              │
        └──────────────┘
                  │
            主Agent包装
                  │
            展示给用户
```

**核心设计原则：**
- **主Agent始终是对话核心**，子Agent作为可调用组件
- **无状态设计**：主Agent和子Agent每次调用都接收完整对话历史，不依赖内部状态
- **上下文隔离**：生图子Agent使用独立的 `image_context`，避免追问/扩写过程污染主对话历史
- **标准化接口**：子Agent通过 `BaseSubAgent` 基类统一接口，便于扩展

---

## 二、主Agent（HuesaeMainAgent）

**文件：** `backend/huesaeagents/huesae/agents/lead_agent/lead_agent.py`

### 2.1 职责

1. **意图分类**：每轮用LLM分析用户输入，判断主意图（chat/image/voice/memory/search/remind）和 image 子意图（generate_image/expand_prompt/convert_tags）
2. **安全检查**：检测自杀/自残关键词，最高优先级拦截
3. **子Agent委派**：当意图匹配已注册的子Agent时，调用子Agent处理
4. **结果包装**：子Agent返回的结果由主Agent用角色语气包装展示
5. **聊天回复**：非子Agent意图时，主Agent直接用角色语气聊天回复

### 2.2 核心方法

#### `process(state, user_input) -> dict`

主入口，处理单轮用户输入。

**逻辑流程：**
1. 安全检查 → 命中则返回安全回复
2. 意图分类 → 已有 `image_intent` 则复用（避免每轮调用LLM）
3. 子Agent匹配 → 调用 `_handle_sub_agent()`
4. 直接聊天 → 调用 `_chat_reply()`

**返回结果（dict）：**
| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | `list[AIMessage]` | AI回复消息列表 |
| `pending_generation` | `bool` | 是否需要异步生图 |
| `prompt` | `str` | 生图提示词（`pending_generation=true`时） |
| `size` | `str` | 图片尺寸（默认2K） |
| `output_format` | `str` | 输出格式（jpeg/png） |
| `is_batch` | `bool` | 是否组图模式 |
| `image_intent` | `str` | 当前image子意图 |
| `image_context` | `list` | 生图子Agent的对话历史 |
| `clear_image_intent` | `bool` | 是否清除image状态 |
| `safety_flag` | `bool` | 是否触发安全检查 |

#### `_classify_intent(state, user_input) -> IntentResult`

用LLM做意图分类，传入最近6条对话历史。使用 Pydantic 结构化输出 (`IntentResult`)，包含 `intent`、`image_intent`、`reason`。

**Fallback机制：** LLM调用失败时默认返回 `IntentResult(intent="chat")`。

#### `_handle_sub_agent(intent, state, user_input) -> dict`

调用子Agent并包装结果。

**关键设计：上下文隔离**
- 子Agent使用独立的 `image_context` 作为对话历史
- 主Agent的 `messages` 只保留核心业务消息（追问/扩写等中间过程不污染）
- 子Agent完成后，`image_context` 返回给调用者持久化

**对子Agent返回action的处理：**

| action | 处理逻辑 |
|--------|---------|
| `ask_prompt` | 直接展示子Agent回复，追加到image_context |
| `recommend` | 同上 |
| `ask_confirm` | 同上 |
| `generate` | 返回 `pending_generation=true`，保存生图参数（prompt/size/output_format/is_batch） |
| `finish` | `generate_image`意图：主Agent角色语气结束语；`expand_prompt`/`convert_tags`：直接展示结果 |

#### `execute_image_generation(prompt, size, output_format, is_batch) -> dict`

异步执行生图，供外部（如chat_loop）调用。

**单图 vs 组图分支：**
- `is_batch=false`（默认）：调用 `agent.generate_image()` → 返回 `{"wrap_message", "image_url"}`
- `is_batch=true`：调用 `agent.generate_images()` → 返回 `{"wrap_message", "image_urls": [...]}`

包装语 `wrap_message` 由 `_create_wrap_message()` 用角色语气动态生成。

---

## 三、生图子Agent（ImageSubAgent）

**文件：** `backend/huesaeagents/huesae/agents/subagents/image_agent.py`

### 3.1 职责

1. **LLM决策**：每轮分析完整对话历史，输出下一步action
2. **提示词扩写**：调用 `expand_prompt()` 将简短描述扩写为详细提示词
3. **风格处理**：自动添加"二次元动漫风格"前缀（除非用户要求真人风格）
4. **生图执行**：单图调用 `generate_image()`，组图调用 `generate_images()`

### 3.2 核心方法

#### `process(state, user_input) -> dict`

主入口，返回标准化结果。

**逻辑流程：**
1. `_decide()` → LLM输出 `ImageDecision`
2. 根据 `action` 分发处理：
   - `expand` → `_handle_expand()` → 调用 `expand_prompt()`
   - `generate` → `_handle_generate()` → 构造生图参数
   - 其他 → 直接返回决策中的 `response`

**返回结果（标准化dict）：**
| 字段 | 类型 | 说明 |
|------|------|------|
| `action` | `str` | 动作类型 |
| `response` | `str` | 给用户的回复 |
| `prompt` | `str` | 确认的提示词 |
| `provider` | `str` | 生图工具（固定doubao） |
| `size` | `str` | 图片尺寸 |
| `output_format` | `str` | 输出格式 |
| `is_batch` | `bool` | 是否组图 |
| `data` | `dict` | 额外数据（扩展用） |

#### `_decide(state, user_input) -> ImageDecision`

LLM决策核心方法。

**构建prompt：**
- 当前已确认的提示词
- 用户原始意图（`image_intent`）
- 可用生图工具列表
- 最近6条对话历史

**`ImageDecision` 模型字段：**
| 字段 | 类型 | 说明 |
|------|------|------|
| `thought` | `str` | 分析当前对话状态和用户需求 |
| `action` | `str` | 下一步动作（ask_prompt/recommend/expand/ask_confirm/generate/show_image/finish） |
| `response` | `str` | 给用户的回复消息 |
| `prompt` | `str` | 当前确认的纯描述提示词（保留数量描述，不含颜文字） |
| `size` | `str` | 图片尺寸（1K/2K/3K/4K） |
| `output_format` | `str` | 输出格式（jpeg/png） |
| `is_batch` | `bool` | 是否组图模式 |

**Fallback机制：** LLM调用失败时返回 `ImageDecision(action="ask_prompt")`。

#### `_ensure_anime_style(prompt) -> str`

自动添加动漫风格前缀：
- 用户要求真人/写实风格（关键词："真人""写实""照片""photorealistic"等）→ 不添加
- 其他情况 → 前缀 `图片风格为 二次元，`

#### `generate_image(prompt, provider_name, size, output_format) -> GenerationResult`

单图生成，委托给Provider的 `generate()` 方法。

#### `generate_images(prompt, provider_name, size, output_format) -> list[GenerationResult]`

组图生成（非流式）：
1. 调用 `DoubaoClient.generate_images()`（`sequential_image_generation="auto"`，`max_images=12`）
2. 从 `response.data` 中提取所有图片URL
3. 返回 `list[GenerationResult]`

**为什么是12张上限：** `max_images` 是最大限制，豆包内部根据 prompt 中的自然语言描述（如"生成4张"）决定实际生成张数。

---

## 四、生图对话流程（Prompt控制）

**文件：** `backend/huesaeagents/huesae/agents/subagents/image/prompts.py`

### 4.1 工作流程（`IMAGE_CONVERSATION_PROMPT`）

```
用户: "我想生成图片"
  → ask_prompt: "请告诉我您想要生成什么样的图片？"

用户: "一个女孩子在樱花树下"
  → ask_confirm + expand: "描述有点简短呢..." 或 直接 ask_confirm

用户: "扩写一下"
  → expand → ask_confirm: "扩写后的描述：...这个描述可以吗？"

用户: "可以"
  → generate: 返回生图参数（pending_generation=true）

chat_loop 执行生图 → 展示图片

用户: "换一组"
  → generate: 使用上一次保存的 prompt 重新生图

用户: "不用了"
  → finish: 结束对话
```

### 4.2 关键Prompt规则

1. **绝不自动扩写**：用户没有明确要求时，严禁返回 `expand` action
2. **确认闭环**：推荐/扩写后必须询问用户是否满意
3. **数量保留**：`prompt` 字段保留用户指定的数量描述（如"3张""一组"）
4. **比例不主动询问**：用户可在描述中自然提及比例，模型自动判断；仅用户明确说"使用4:3比例"时才传参
5. **组图判断**：用户明确说明生成数量（如"生成4张"）时 `is_batch=true`，否则默认单图

---

## 五、状态管理

### 5.1 HuesaeState

**文件：** `backend/huesaeagents/huesae/agents/state/huesae_state.py`

```python
class HuesaeState:
    messages: list              # 主对话历史
    image_context: list         # 生图子Agent独立对话历史
    intent: str | None          # 当前主意图
    image_intent: str | None    # image子意图（generate_image/expand_prompt/convert_tags）
    current_image_prompt: str   # 当前确认的提示词（用于换图）
```

### 5.2 StateManager

**文件：** `backend/huesaeagents/huesae/agents/state_manager.py`

- 支持内存 + 文件持久化（JSON格式）
- 自动序列化/反序列化 LangChain Message 对象
- 启动时检测残留 `image_intent`，自动重置防止上次异常退出影响

### 5.3 状态生命周期

```
用户说"我想生成图片"
  → image_intent="generate_image"（持久化）
  → image_context 开始累积追问/确认对话

生图完成
  → current_image_prompt=确认后的prompt（持久化）
  → image_context 追加"图片生成完成"包装语

用户说"换一组"
  → 使用 current_image_prompt 重新生图
  → image_intent 和 image_context 保留

用户说"不用了"
  → clear_image()：image_intent=None, image_context=[], current_image_prompt=None
```

---

## 六、Provider层

### 6.1 抽象基类

**文件：** `backend/huesaeagents/huesae/agents/subagents/image/providers/base.py`

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
    def name(self) -> str

    @abstractmethod
    async def generate(prompt, size, **kwargs) -> GenerationResult
```

### 6.2 豆包Provider

**文件：** `backend/huesaeagents/huesae/agents/subagents/image/providers/doubao.py`

- 封装 `DoubaoClient`
- `generate()` 使用 `asyncio.to_thread()` 将同步API调用包装为异步
- 当前唯一注册的Provider

### 6.3 豆包客户端

**文件：** `backend/huesaeagents/huesae/tools/doubao/client.py`

**单例模式：** 通过 `__new__` 实现，确保全局只有一个 `DoubaoClient` 实例。

**方法：**
| 方法 | 说明 |
|------|------|
| `generate_image()` | 单图生成，`response.data[0].url` |
| `generate_images()` | 组图生成（非流式），`sequential_image_generation="auto"`，遍历 `response.data` 提取所有URL |
| `generate_images_stream()` | 流式组图（保留但当前未使用），逐事件 yield |
| `generate_image_debug()` | 调试模式，返回完整响应 |

---

## 七、交互层

**文件：** `backend/huesaeagents/huesae/agents/lead_agent/chat_loop.py`

### 7.1 运行流程

```python
while True:
    user_input = input("用户: ")

    # 1. 构建 state
    state = {
        "messages": conv_state.messages,
        "image_context": conv_state.image_context,
        "image_intent": conv_state.image_intent,
        "current_image_prompt": conv_state.current_image_prompt,
    }

    # 2. 调用主Agent
    result = main_agent.process(state, user_input)

    # 3. 打印AI回复（打字机效果）
    for msg in result["messages"]:
        print_stream(msg.content)

    # 4. 处理生图（异步）
    if result.get("pending_generation"):
        image_result = asyncio.run(
            main_agent.execute_image_generation(...)
        )
        # 单图: [图片] url
        # 组图: [图片] url1, [图片] url2, ...

    # 5. 更新并持久化状态
    conv_state.messages.append(HumanMessage(user_input))
    conv_state.messages.extend(result["messages"])
    state_manager.save_state(session_id)
```

### 7.2 关键设计

- **异步生图**：`asyncio.run()` 在同步循环中执行异步生图，避免阻塞
- **状态持久化**：每轮结束后自动保存到JSON文件
- **异常处理**：生图失败时打印错误并清除image状态

---

## 八、数据流详解

### 8.1 单图生成完整链路

```
用户: "生成一张夕阳下的大海"
    ↓
[主Agent] process()
  _classify_intent() → intent=IMAGE, image_intent=generate_image
  _handle_sub_agent()
    [ImageSubAgent] process()
      _decide() → action=generate, prompt="夕阳下的大海", is_batch=false
      _handle_generate() → {"action":"generate", "prompt":"图片风格为 二次元，夕阳下的大海", "is_batch":false}
    ← 返回 {"pending_generation":true, "prompt":"...", "is_batch":false}
    ↓
[chat_loop] pending_generation 处理
  execute_image_generation(is_batch=false)
    [ImageSubAgent] generate_image()
      [DoubaoProvider] generate()
        [DoubaoClient] generate_image() → "https://url"
    ← 返回 {"wrap_message":"这是生成好的图片哦~", "image_url":"https://url"}
  打印: [图片] https://url
```

### 8.2 组图生成完整链路

```
用户: "生成4张图，四季插画"
    ↓
[主Agent] _handle_sub_agent()
  [ImageSubAgent] _decide() → action=generate, prompt="生成4张图，四季插画", is_batch=true
    ↓
[chat_loop] execute_image_generation(is_batch=true)
  [ImageSubAgent] generate_images()
    [DoubaoClient] generate_images(
      sequential_image_generation="auto",
      sequential_image_generation_options={"max_images":12}
    ) → response.data = [img1, img2, img3, img4]
  ← 返回 {"wrap_message":"...", "image_urls":["url1","url2","url3","url4"]}
  打印:
    [图片] url1
    [图片] url2
    [图片] url3
    [图片] url4
```

---

## 九、扩展指南

### 9.1 添加新的子Agent

1. 继承 `BaseSubAgent`，实现 `process()` 和 `name` 属性
2. 在主Agent中注册：`main_agent.register_sub_agent(new_agent)`
3. 在 `_classify_intent()` 的prompt中增加对新意图的分类规则

### 9.2 添加新的生图Provider

1. 继承 `ImageProvider`，实现 `name` 和 `generate()`
2. 在 `create_image_agent()` 工厂函数中注册

### 9.3 修改角色语气

在 `prompts.py` 的 `get_character_system_message()` 中增加新的角色映射。

---

## 十、文件清单

| 文件 | 职责 |
|------|------|
| `agents/lead_agent/lead_agent.py` | 主Agent：意图分类、委派、包装 |
| `agents/lead_agent/chat_loop.py` | 终端交互循环、异步生图、状态持久化 |
| `agents/subagents/base.py` | 子Agent抽象基类 |
| `agents/subagents/image_agent.py` | 生图子Agent：LLM决策、扩写、生图执行 |
| `agents/subagents/image/prompts.py` | 所有系统提示词（意图识别、扩写、生图对话管理、角色语气） |
| `agents/subagents/image/providers/base.py` | Provider抽象基类 + GenerationResult |
| `agents/subagents/image/providers/doubao.py` | 豆包Provider封装 |
| `tools/doubao/client.py` | 豆包API客户端（单例）：单图/组图/流式/调试 |
| `tools/image.py` | 图片工具封装（jimeng/doubao） |
| `agents/utils/agent_tools.py` | LangChain Tool定义（供未来工具链使用） |
| `agents/state/huesae_state.py` | 状态数据类 |
| `agents/state_manager.py` | 状态管理器（内存+文件持久化） |
