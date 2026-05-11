# LangChain Messages 关键代码写法总结

## 目录

- [基础概念](#基础概念)
- [消息类型](#消息类型)
- [基本用法](#基本用法)
- [SystemMessage](#systemmessage)
- [HumanMessage](#humanmessage)
- [AIMessage](#aimessage)
- [ToolMessage](#toolmessage)
- [消息内容 (Content)](#消息内容-content)
- [标准内容块 (Content Blocks)](#标准内容块-content-blocks)
- [多模态 (Multimodal)](#多模态-multimodal)
- [流式处理](#流式处理)

---

## 基础概念

Messages 是 LangChain 中模型交互的基本单元，包含：
- **Role** - 标识消息类型（system、user、assistant）
- **Content** - 消息的实际内容（文本、图像、音频等）
- **Metadata** - 可选字段，如响应信息、消息ID、token使用量

---

## 消息类型

| 类型 | 说明 |
|------|------|
| `SystemMessage` | 设定模型行为和上下文的初始指令 |
| `HumanMessage` | 用户输入和交互 |
| `AIMessage` | 模型生成的响应，包括文本、工具调用、元数据 |
| `ToolMessage` | 工具执行结果，传递给模型 |

---

## 基本用法

### 字符串输入

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano")

response = model.invoke("Write a haiku about spring")
```

### 消息对象列表

```python
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model("gpt-5-nano")

messages = [
    SystemMessage("You are a poetry expert"),
    HumanMessage("Write a haiku about spring"),
    AIMessage("Cherry blossoms bloom...")
]
response = model.invoke(messages)
```

### 字典格式（OpenAI 兼容）

```python
messages = [
    {"role": "system", "content": "You are a poetry expert"},
    {"role": "user", "content": "Write a haiku about spring"},
    {"role": "assistant", "content": "Cherry blossoms bloom..."}
]
response = model.invoke(messages)
```

---

## SystemMessage

设定模型行为的初始指令，可用于设置语气、定义角色、制定响应指南。

### 基本用法

```python
from langchain.messages import SystemMessage, HumanMessage

system_msg = SystemMessage("You are a helpful coding assistant.")
messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)
```

### 详细角色定义

```python
system_msg = SystemMessage("""
You are a senior Python developer with expertise in web frameworks.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.
""")
messages = [system_msg, HumanMessage("How do I create a REST API?")]
response = model.invoke(messages)
```

---

## HumanMessage

用户输入，可以包含文本、图像、音频、文件等**多模态内容**。

### 文本内容

```python
from langchain.messages import HumanMessage

# Message 对象
response = model.invoke([HumanMessage("What is machine learning?")])
```

### 带元数据

```python
human_msg = HumanMessage(
    content="Hello!",
    name="alice",      # 可选：标识不同用户
    id="msg_123",      # 可选：唯一标识符，用于追踪
)
```

---

## AIMessage

模型输出，包含文本内容、工具调用、提供商特定元数据。

### 基本用法

```python
response = model.invoke("Explain AI")
print(type(response))  # <class 'langchain.messages.AIMessage'>
```

### 手动创建 AIMessage

```python
from langchain.messages import AIMessage, SystemMessage, HumanMessage

# 手动创建 AI 消息（如用于对话历史）
ai_msg = AIMessage("I'd be happy to help you with that question!")

# 添加到对话历史
messages = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage("Can you help me?"),
    ai_msg,  # 插入为模型输出
    HumanMessage("Great! What's 2+2?")
]
response = model.invoke(messages)
```

### AIMessage 属性

| 属性 | 说明 |
|------|------|
| `text` | 文本内容（字符串） |
| `content` | 原始内容（字符串或字典列表） |
| `content_blocks` | 标准内容块列表 |
| `tool_calls` | 模型调用的工具列表 |
| `id` | 消息唯一标识符 |
| `usage_metadata` | Token 使用量 |
| `response_metadata` | 响应元数据 |

### 工具调用

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano")

model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("What's the weather in Paris?")

for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
    print(f"ID: {tool_call['id']}")
```

### Token 使用量

```python
response = model.invoke("Hello!")
response.usage_metadata
# {'input_tokens': 8, 'output_tokens': 304, 'total_tokens': 312,
#  'input_token_details': {'audio': 0, 'cache_read': 0},
#  'output_token_details': {'audio': 0, 'reasoning': 256}}
```

---

## ToolMessage

将工具执行结果传回模型。

### 基本用法

```python
from langchain.messages import AIMessage, ToolMessage

# 模型生成工具调用
ai_message = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "San Francisco"},
        "id": "call_123"
    }]
)

# 执行工具并创建结果消息
weather_result = "Sunny, 72°F"
tool_message = ToolMessage(
    content=weather_result,
    tool_call_id="call_123"  # 必须匹配工具调用的 ID
)

# 继续对话
messages = [
    HumanMessage("What's the weather in San Francisco?"),
    ai_message,       # 模型的工具调用
    tool_message       # 工具执行结果
]
response = model.invoke(messages)
```

### ToolMessage 属性

| 属性 | 说明 |
|------|------|
| `content` | 工具输出的字符串形式（必需） |
| `tool_call_id` | 对应工具调用的 ID（必需） |
| `name` | 被调用工具的名称 |
| `artifact` | 附加数据，不发送给模型但可编程访问 |

### 使用 artifact 存储附加数据

```python
from langchain.messages import ToolMessage

# 发送给模型的内容
message_content = "It was the best of times, it was the worst of times."

# 不发送给模型但可编程访问的数据
artifact = {
    "document_id": "doc_123",
    "page": 0
}

tool_message = ToolMessage(
    content=message_content,
    tool_call_id="call_123",
    name="search_books",
    artifact=artifact,
)
```

---

## 消息内容 (Content)

消息的 `content` 属性支持：
- 字符串
- 内容块列表（provider-native 格式或 LangChain 标准格式）

### 多模态输入示例

```python
from langchain.messages import HumanMessage

# 字符串内容
human_message = HumanMessage("Hello, how are you?")

# Provider-native 格式（如 OpenAI）
human_message = HumanMessage(content=[
    {"type": "text", "text": "Hello, how are you?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])

# 标准内容块列表
human_message = HumanMessage(content_blocks=[
    {"type": "text", "text": "Hello, how are you?"},
    {"type": "image", "url": "https://example.com/image.jpg"},
])
```

---

## 标准内容块 (Content Blocks)

LangChain 提供跨提供商的标准内容块表示，通过 `content_blocks` 属性访问。

### 内容块类型

| 类型 | 用途 |
|------|------|
| `TextContentBlock` | 标准文本输出 |
| `ReasoningContentBlock` | 模型推理步骤 |
| `ImageContentBlock` | 图像数据 |
| `AudioContentBlock` | 音频数据 |
| `VideoContentBlock` | 视频数据 |
| `FileContentBlock` | 通用文件（PDF等） |
| `PlainTextContentBlock` | 文档文本（.txt, .md） |
| `ToolCall` | 函数调用 |
| `ToolCallChunk` | 流式工具调用片段 |
| `ServerToolCall` | 服务端工具调用 |
| `ServerToolResult` | 搜索结果 |

### Anthropic 格式解析

```python
from langchain.messages import AIMessage

message = AIMessage(
    content=[
        {"type": "thinking", "thinking": "...", "signature": "WaUjzkyp..."},
        {"type": "text", "text": "..."},
    ],
    response_metadata={"model_provider": "anthropic"}
)

message.content_blocks
# [{'type': 'reasoning', 'reasoning': '...', 'extras': {'signature': 'WaUjzkyp...'}},
#  {'type': 'text', 'text': '...'}]
```

### OpenAI 格式解析

```python
message = AIMessage(
    content=[
        {
            "type": "reasoning",
            "id": "rs_abc123",
            "summary": [
                {"type": "summary_text", "text": "summary 1"},
                {"type": "summary_text", "text": "summary 2"},
            ],
        },
        {"type": "text", "text": "...", "id": "msg_abc123"},
    ],
    response_metadata={"model_provider": "openai"}
)

message.content_blocks
# [{'type': 'reasoning', 'id': 'rs_abc123', 'reasoning': 'summary 1'},
#  {'type': 'reasoning', 'id': 'rs_abc123', 'reasoning': 'summary 2'},
#  {'type': 'text', 'text': '...', 'id': 'msg_abc123'}]
```

### 内容块参考

**TextContentBlock:**
```python
{"type": "text", "text": "Hello world", "annotations": []}
```

**ReasoningContentBlock:**
```python
{"type": "reasoning", "reasoning": "The user is asking about...", "extras": {...}}
```

**ImageContentBlock:**
```python
{"type": "image", "url": "https://example.com/image.jpg"}
{"type": "image", "base64": "AAAAIGZ0eXBtcDQy...", "mime_type": "image/jpeg"}
```

**AudioContentBlock:**
```python
{"type": "audio", "url": "https://example.com/audio.mp3"}
{"type": "audio", "base64": "...", "mime_type": "audio/mpeg"}
```

**VideoContentBlock:**
```python
{"type": "video", "url": "https://example.com/video.mp4"}
```

**FileContentBlock:**
```python
{"type": "file", "url": "https://example.com/doc.pdf"}
{"type": "file", "base64": "...", "mime_type": "application/pdf"}
```

**ToolCall:**
```python
{"type": "tool_call", "name": "search", "args": {"query": "weather"}, "id": "call_123"}
```

---

## 多模态 (Multimodal)

### 图像输入

**从 URL：**
```python
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {"type": "image", "url": "https://example.com/path/to/image.jpg"},
    ]
}
```

**从 base64：**
```python
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {
            "type": "image",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAA...",
            "mime_type": "image/jpeg"
        },
    ]
}
```

**从 provider 管理的 File ID：**
```python
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {"type": "image", "file_id": "file-abc123"},
    ]
}
```

### PDF 文档输入

```python
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Summarize this document."},
        {"type": "file", "url": "https://example.com/doc.pdf"},
    ]
}
```

### 音频输入

```python
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Transcribe this audio."},
        {"type": "audio", "url": "https://example.com/audio.mp3"},
    ]
}
```

---

## 流式处理

流式输出时，接收 `AIMessageChunk` 对象，可合并为完整消息。

### 基本流式

```python
chunks = []
full_message = None

for chunk in model.stream("Hi"):
    chunks.append(chunk)
    print(chunk.text)
    full_message = chunk if full_message is None else full_message + chunk
```

### 流式工具调用

```python
for chunk in model_with_tools.stream("What's the weather in Boston?"):
    for tool_chunk in chunk.tool_call_chunks:
        if name := tool_chunk.get("name"):
            print(f"Tool: {name}")
        if args := tool_chunk.get("args"):
            print(f"Args: {args}")
```

---

## 完整示例

```python
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.tools import tool

# 1. 初始化模型
model = init_chat_model("gpt-5.4")

# 2. 定义工具
@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Sunny in {location}"

# 3. 绑定工具
model_with_tools = model.bind_tools([get_weather])

# 4. 对话历史
messages = [
    SystemMessage("You are a helpful weather assistant."),
    HumanMessage("What's the weather in Boston?"),
]

# 5. 模型调用
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# 6. 工具执行
if ai_msg.tool_calls:
    for tool_call in ai_msg.tool_calls:
        result = get_weather.invoke(tool_call)
        tool_msg = ToolMessage(
            content=result,
            tool_call_id=tool_call["id"]
        )
        messages.append(tool_msg)

# 7. 最终响应
final_response = model_with_tools.invoke(messages)
print(final_response.text)
```

---

## 参考链接

- 官方文档：https://docs.langchain.com/oss/python/langchain/messages
- 完整文档索引：https://docs.langchain.com/llms.txt
