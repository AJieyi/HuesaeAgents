# LangChain Models 关键代码写法总结

## 目录

- [基础概念](#基础概念)
- [初始化模型](#初始化模型)
- [调用方法 (Invocation)](#调用方法-invocation)
- [工具调用 (Tool Calling)](#工具调用-tool-calling)
- [结构化输出 (Structured Output)](#结构化输出-structured-output)
- [流式输出 (Streaming)](#流式输出-streaming)
- [批量处理 (Batch)](#批量处理-batch)
- [高级特性](#高级特性)

---

## 基础概念

LangChain Models 是 Agent 的推理引擎，支持：
- **Tool calling** - 调用外部工具
- **Structured output** - 约束输出格式
- **Multimodality** - 处理图像、音频、视频
- **Reasoning** - 多步推理

---

## 初始化模型

### 使用 init_chat_model（推荐方式）

```python
from langchain.chat_models import init_chat_model
```

### OpenAI

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

model = init_chat_model("gpt-5.4")
```

### Anthropic

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-..."

model = init_chat_model("claude-sonnet-4-6")
```

### Azure OpenAI

```python
import os
os.environ["AZURE_OPENAI_API_KEY"] = "..."
os.environ["AZURE_OPENAI_ENDPOINT"] = "..."
os.environ["OPENAI_API_VERSION"] = "2025-03-01-preview"

model = init_chat_model(
    "azure_openai:gpt-5.4",
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)
```

### Google Gemini

```python
import os
os.environ["GOOGLE_API_KEY"] = "..."

model = init_chat_model("google_genai:gemini-2.5-flash-lite")
```

### AWS Bedrock

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_provider="bedrock_converse",
)
```

### HuggingFace

```python
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_..."

model = init_chat_model(
    "microsoft/Phi-3-mini-4k-instruct",
    model_provider="huggingface",
    temperature=0.7,
    max_tokens=1024,
)
```

### OpenRouter

```python
import os
os.environ["OPENROUTER_API_KEY"] = "sk-..."

model = init_chat_model("auto", model_provider="openrouter")
```

### 自定义 Base URL（OpenAI 兼容 API）

```python
model = init_chat_model(
    model="MODEL_NAME",
    model_provider="openai",
    base_url="BASE_URL",
    api_key="YOUR_API_KEY",
)
```

### 模型参数

```python
model = init_chat_model(
    "claude-sonnet-4-6",
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
    max_retries=6,  # 默认 6，不可靠网络可设为 10-15
)
```

---

## 调用方法 (Invocation)

### Invoke（同步调用）

```python
# 单消息
response = model.invoke("Why do parrots have colorful feathers?")
print(response)
```

### 消息格式

**字典格式：**

```python
conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to French."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "Translate: I love building applications."}
]
response = model.invoke(conversation)
```

**Message 对象格式：**

```python
from langchain.messages import HumanMessage, AIMessage, SystemMessage

conversation = [
    SystemMessage("You are a helpful assistant that translates English to French."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Translate: I love building applications.")
]
response = model.invoke(conversation)
```

---

## 工具调用 (Tool Calling)

### 绑定工具

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."

model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("What's the weather like in Boston?")
for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
```

### 工具执行循环

```python
# 绑定工具
model_with_tools = model.bind_tools([get_weather])

# Step 1: 模型生成工具调用
messages = [{"role": "user", "content": "What's the weather in Boston?"}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# Step 2: 执行工具并收集结果
for tool_call in ai_msg.tool_calls:
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

# Step 3: 将结果传回模型获取最终响应
final_response = model_with_tools.invoke(messages)
print(final_response.text)
```

### 强制使用工具

```python
# 强制使用任意一个工具
model_with_tools = model.bind_tools([tool_1], tool_choice="any")

# 强制使用特定工具
model_with_tools = model.bind_tools([tool_1], tool_choice="get_weather")
```

### 并行工具调用

```python
response = model_with_tools.invoke("What's the weather in Boston and Tokyo?")
print(response.tool_calls)
# [{'name': 'get_weather', 'args': {'location': 'Boston'}, 'id': 'call_1'},
#  {'name': 'get_weather', 'args': {'location': 'Tokyo'}, 'id': 'call_2'}]
```

### 禁用并行工具调用

```python
model.bind_tools([get_weather], parallel_tool_calls=False)
```

### 流式工具调用

```python
for chunk in model_with_tools.stream("What's the weather in Boston and Tokyo?"):
    for tool_chunk in chunk.tool_call_chunks:
        if name := tool_chunk.get("name"):
            print(f"Tool: {name}")
        if args := tool_chunk.get("args"):
            print(f"Args: {args}")

# 累积构建完整工具调用
gathered = None
for chunk in model_with_tools.stream("What's the weather in Boston?"):
    gathered = chunk if gathered is None else gathered + chunk
print(gathered.tool_calls)
```

### 服务端工具调用

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5.4-mini")
tool = {"type": "web_search"}

model_with_tools = model.bind_tools([tool])
response = model_with_tools.invoke("What was a positive news story from today?")
print(response.content_blocks)
# [{'type': 'server_tool_call', 'name': 'web_search', ...},
#  {'type': 'server_tool_result', 'tool_call_id': 'ws_abc123', ...},
#  {'type': 'text', 'text': 'Here are some positive news stories...'}]
```

---

## 结构化输出 (Structured Output)

### Pydantic 模型（推荐，功能最丰富）

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")
    rating: float = Field(description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("Provide details about the movie Inception")
print(response)
# Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)
```

### TypedDict（轻量级，无需运行时验证）

```python
from typing_extensions import TypedDict, Annotated

class MovieDict(TypedDict):
    """A movie with details."""
    title: Annotated[str, ..., "The title of the movie"]
    year: Annotated[int, ..., "The year the movie was released"]
    director: Annotated[str, ..., "The director of the movie"]
    rating: Annotated[float, ..., "The movie's rating out of 10"]

model_with_structure = model.with_structured_output(MovieDict)
response = model_with_structure.invoke("Provide details about the movie Inception")
# {'title': 'Inception', 'year': 2010, 'director': 'Christopher Nolan', 'rating': 8.8}
```

### JSON Schema

```python
import json

json_schema = {
    "title": "Movie",
    "description": "A movie with details",
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "The title of the movie"},
        "year": {"type": "integer", "description": "The year the movie was released"},
        "director": {"type": "string", "description": "The director of the movie"},
        "rating": {"type": "number", "description": "The movie's rating out of 10"}
    },
    "required": ["title", "year", "director", "rating"]
}

model_with_structure = model.with_structured_output(json_schema, method="json_schema")
```

### 嵌套结构

```python
from pydantic import BaseModel, Field

class Actor(BaseModel):
    name: str
    role: str

class MovieDetails(BaseModel):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: float | None = Field(None, description="Budget in millions USD")

model_with_structure = model.with_structured_output(MovieDetails)
```

### 包含原始消息

```python
model_with_structure = model.with_structured_output(Movie, include_raw=True)
response = model_with_structure.invoke("Provide details about the movie Inception")
# {"raw": AIMessage(...), "parsed": Movie(...), "parsing_error": None}
```

### Structured Output 方法

| 方法 | 说明 |
|------|------|
| `json_schema` | 使用提供者原生结构化输出功能 |
| `function_calling` | 通过强制工具调用生成结构化输出 |
| `json_mode` | 生成有效 JSON，需在提示词中描述格式 |

---

## 流式输出 (Streaming)

### 基础文本流

```python
for chunk in model.stream("Why do parrots have colorful feathers?"):
    print(chunk.text, end="|", flush=True)
```

### 累积消息

```python
full = None
for chunk in model.stream("What color is the sky?"):
    full = chunk if full is None else full + chunk
print(full.text)
print(full.content_blocks)
```

### 流式事件

```python
async for event in model.astream_events("Hello"):
    if event["event"] == "on_chat_model_start":
        print(f"Input: {event['data']['input']}")
    elif event["event"] == "on_chat_model_stream":
        print(f"Token: {event['data']['chunk'].text}")
    elif event["event"] == "on_chat_model_end":
        print(f"Full message: {event['data']['output'].text}")
```

---

## 批量处理 (Batch)

### 批量调用

```python
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])
for response in responses:
    print(response)
```

### 按完成顺序返回

```python
for response in model.batch_as_completed([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]):
    print(response)
```

### 控制并发数

```python
model.batch(list_of_inputs, config={'max_concurrency': 5})
```

---

## 高级特性

### 模型配置

```python
# Log probabilities
model = init_chat_model("gpt-5.4", model_provider="openai").bind(logprobs=True)
response = model.invoke("Why do parrots talk?")
print(response.response_metadata["logprobs"])
```

### 速率限制器

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,    # 每 10 秒 1 请求
    check_every_n_seconds=0.1,  # 每 100ms 检查
    max_bucket_size=10,          # 最大突发大小
)

model = init_chat_model(
    model="gpt-5.4",
    model_provider="openai",
    rate_limiter=rate_limiter
)
```

### 代理配置

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5.4",
    openai_proxy="http://proxy.example.com:8080"
)
```

### Token 使用追踪

**Callback 方式：**

```python
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler

model_1 = init_chat_model(model="gpt-5.4-mini")
model_2 = init_chat_model(model="claude-haiku-4-5-20251001")

callback = UsageMetadataCallbackHandler()
result_1 = model_1.invoke("Hello", config={"callbacks": [callback]})
result_2 = model_2.invoke("Hello", config={"callbacks": [callback]})

print(callback.usage_metadata)
```

**Context Manager 方式：**

```python
from langchain_core.callbacks import get_usage_metadata_callback

model_1 = init_chat_model(model="gpt-5.4-mini")
model_2 = init_chat_model(model="claude-haiku-4-5-20251001")

with get_usage_metadata_callback() as cb:
    model_1.invoke("Hello")
    model_2.invoke("Hello")

print(cb.usage_metadata)
```

### 调用配置 (Invocation Config)

```python
response = model.invoke(
    "Tell me a joke",
    config={
        "run_name": "joke_generation",      # 自定义运行名称
        "tags": ["humor", "demo"],           # 分类标签
        "metadata": {"user_id": "123"},      # 自定义元数据
        "callbacks": [my_callback_handler],  # 回调处理器
    }
)
```

### 可配置模型

**基础用法：**

```python
from langchain.chat_models import init_chat_model

configurable_model = init_chat_model(temperature=0)

configurable_model.invoke("what's your name", config={"configurable": {"model": "gpt-5-nano"}})
configurable_model.invoke("what's your name", config={"configurable": {"model": "claude-sonnet-4-6"}})
```

**带默认值的可配置模型：**

```python
first_model = init_chat_model(
    model="gpt-5.4-mini",
    temperature=0,
    configurable_fields=("model", "model_provider", "temperature", "max_tokens"),
    config_prefix="first",
)

first_model.invoke("what's your name")
first_model.invoke(
    "what's your name",
    config={"configurable": {
        "first_model": "claude-sonnet-4-6",
        "first_temperature": 0.5,
        "first_max_tokens": 100,
    }},
)
```

### 可配置模型链式调用

```python
from pydantic import BaseModel, Field

class GetWeather(BaseModel):
    """Get the current weather in a given location"""
    location: str = Field(description="The city and state, e.g. San Francisco, CA")

class GetPopulation(BaseModel):
    """Get the current population in a given location"""
    location: str = Field(description="The city and state, e.g. San Francisco, CA")

model = init_chat_model(temperature=0)
model_with_tools = model.bind_tools([GetWeather, GetPopulation])

# 使用不同模型调用
result = model_with_tools.invoke(
    "what's bigger in 2024 LA or NYC",
    config={"configurable": {"model": "gpt-5.4-mini"}}
)
```

### 模型 Profile

```python
# 查看模型能力
print(model.profile)
# {"max_input_tokens": 400000, "image_inputs": True, "reasoning_output": True, ...}

# 自定义 profile
custom_profile = {
    "max_input_tokens": 100_000,
    "tool_calling": True,
    "structured_output": True,
}
model = init_chat_model("...", profile=custom_profile)

# 更新 profile
new_profile = model.profile | {"key": "value"}
model.model_copy(update={"profile": new_profile})
```

### 多模态

**输入多模态：**

```python
# 通过 content blocks 传递（非文本数据）
response = model.invoke([...])  # 包含图像、音频等
```

**输出多模态：**

```python
response = model.invoke("Create a picture of a cat")
print(response.content_blocks)
# [{"type": "text", "text": "Here's a picture of a cat"},
#  {"type": "image", "base64": "...", "mime_type": "image/jpeg"}]
```

### 推理输出

```python
for chunk in model.stream("Why do parrots have colorful feathers?"):
    reasoning_steps = [r for r in chunk.content_blocks if r["type"] == "reasoning"]
    print(reasoning_steps if reasoning_steps else chunk.text)
```

### 本地模型 (Ollama)

```python
# 使用 Ollama 运行本地模型
model = init_chat_model("llama3", model_provider="ollama")
```

### 提示缓存

```python
# 某些提供商会自动缓存
# OpenAI, Gemini: 隐式缓存

# 显式缓存（Anthropic 等）
# 使用 SystemMessage 的 cache_control 字段
```

---

## 完整示例

```python
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from pydantic import BaseModel, Field

# 1. 初始化模型
model = init_chat_model("gpt-5.4", temperature=0.7)

# 2. 定义工具
@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Sunny in {location}"

# 3. 绑定工具
model_with_tools = model.bind_tools([get_weather])

# 4. 结构化输出 schema
class WeatherResponse(BaseModel):
    location: str
    condition: str
    temperature: float

model_with_output = model.with_structured_output(WeatherResponse)

# 5. 调用
response = model_with_tools.invoke("What's the weather in Boston?")
print(response.tool_calls)

# 6. 提取结构化结果
weather = model_with_output.invoke("Boston is sunny, 72F")
print(weather)
```

---

## 参考链接

- 官方文档：https://docs.langchain.com/oss/python/langchain/models
- 支持的提供者列表：https://docs.langchain.com/llms.txt
- init_chat_model 参考：模型初始化详细参数
