# LangChain Streaming 关键代码写法总结

## 目录

- [概述](#概述)
- [流模式](#流模式)
- [Agent 进度流](#agent-进度流)
- [LLM Token 流](#llm-token-流)
- [自定义更新流](#自定义更新流)
- [多模式流](#多模式流)
- [流式推理/思考 Token](#流式推理思考-token)
- [流式工具调用](#流式工具调用)
- [访问已完成消息](#访问已完成消息)
- [人机交互流](#人机交互流)
- [子 Agent 流](#子-agent-流)
- [禁用流式](#禁用流式)
- [v2 流式格式](#v2-流式格式)

---

## 概述

流式系统让应用能够实时展示 Agent 运行中的更新，提升用户体验。

**LangChain 流式能力：**
- 流式 Agent 进度 - 每个步骤后的状态更新
- 流式 LLM token - 模型生成时实时展示
- 流式推理/思考 token - 展示模型推理过程
- 流式自定义更新 - 发出用户定义的信号
- 多模式流 - 同时流式多种类型数据

---

## 流模式

| 模式 | 说明 |
|------|------|
| `updates` | 流式每个 Agent 步骤后的状态更新 |
| `messages` | 流式 LLM 调用生成的 (token, metadata) 元组 |
| `custom` | 使用 stream writer 从节点内部流式自定义数据 |

---

## Agent 进度流

使用 `stream_mode="updates"` 流式 Agent 进度：

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(model="gpt-5-nano", tools=[get_weather])

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="updates",
    version="v2",
):
    if chunk["type"] == "updates":
        for step, data in chunk["data"].items():
            print(f"step: {step}")
            print(f"content: {data['messages'][-1].content_blocks}")
```

---

## LLM Token 流

使用 `stream_mode="messages"` 流式 token：

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        print(f"node: {metadata['langgraph_node']}")
        print(f"content: {token.content_blocks}")
```

---

## 自定义更新流

在工具中使用 `get_stream_writer` 流式自定义数据：

```python
from langchain.agents import create_agent
from langgraph.config import get_stream_writer

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(model="claude-sonnet-4-6", tools=[get_weather])

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="custom",
    version="v2",
):
    if chunk["type"] == "custom":
        print(chunk["data"])
```

> 注意：在工具内使用 `get_stream_writer` 后，无法在 LangGraph 执行上下文外调用该工具。

---

## 多模式流

同时流式多种模式：

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["updates", "custom"],
    version="v2",
):
    print(f"stream_mode: {chunk['type']}")
    print(f"content: {chunk['data']}")
```

---

## 流式推理/思考 Token

某些模型支持内部推理，使用 `stream_mode="messages"` 并过滤 `reasoning` 类型的内容块：

```python
from langchain.agents import create_agent
from langchain.messages import AIMessageChunk
from langchain_anthropic import ChatAnthropic

def get_weather(city: str) -> str:
    return f"It's always sunny in {city}!"

model = ChatAnthropic(
    model_name="claude-sonnet-4-6",
    thinking={"type": "enabled", "budget_tokens": 5000},
)

agent = create_agent(model=model, tools=[get_weather])

for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages",
):
    if not isinstance(token, AIMessageChunk):
        continue
    reasoning = [b for b in token.content_blocks if b["type"] == "reasoning"]
    text = [b for b in token.content_blocks if b["type"] == "text"]
    if reasoning:
        print(f"[thinking] {reasoning[0]['reasoning']}", end="")
    if text:
        print(text[0]["text"], end="")
```

---

## 流式工具调用

### 基本模式

```python
from typing import Any
from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage

def get_weather(city: str) -> str:
    return f"It's always sunny in {city}!"

agent = create_agent("openai:gpt-5.4", tools=[get_weather])

def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)

def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in Boston?"}]},
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)
    elif chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source in ("model", "tools"):
                _render_completed_message(update["messages"][-1])
```

---

## 访问已完成消息

### 方式 1：通过 state 更新（状态追踪的消息）

```python
stream_mode=["messages", "updates"]
```

### 方式 2：通过 custom 更新

在 middleware 中使用 stream writer：

```python
from langchain.agents.middleware import after_agent, AgentState
from langgraph.runtime import Runtime
from langgraph.config import get_stream_writer

@after_agent(can_jump_to=["end"])
def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    stream_writer = get_stream_writer()
    # ... 安全检查逻辑
    stream_writer(result)
```

### 方式 3：累积消息块

```python
full_message = None
for chunk in agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)
            full_message = token if full_message is None else full_message + token
            if token.chunk_position == "last":
                if full_message.tool_calls:
                    print(f"Tool calls: {full_message.tool_calls}")
                full_message = None
```

---

## 人机交互流

结合 human-in-the-loop 中间件和 checkpointer：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt

def get_weather(city: str) -> str:
    return f"It's always sunny in {city}!"

checkpointer = InMemorySaver()
agent = create_agent(
    "openai:gpt-5.4",
    tools=[get_weather],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"get_weather": True})],
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "some_id"}}
interrupts = []

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Can you look up the weather in Boston and San Francisco?"}]},
    config=config,
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source == "__interrupt__":
                interrupts.extend(update)
```

### 恢复执行

```python
decisions = {}
for interrupt in interrupts:
    decisions[interrupt.id] = {"decisions": [...]}

for chunk in agent.stream(
    Command(resume=decisions),
    config=config,
    stream_mode=["messages", "updates"],
    version="v2",
):
    # 处理流式输出
    ...
```

---

## 子 Agent 流

当 Agent 包含多个 LLM 时，通过 `name` 参数区分来源：

```python
# 创建子 agent
weather_agent = create_agent(
    model=weather_model,
    tools=[get_weather],
    name="weather_agent",
)

# 创建 supervisor
supervisor_agent = create_agent(
    model=supervisor_model,
    tools=[call_weather_agent],
    name="supervisor",
)

# 流式时指定 subgraphs=True
current_agent = None
for chunk in agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates"],
    subgraphs=True,
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if agent_name := metadata.get("lc_agent_name"):
            if agent_name != current_agent:
                print(f"🤖 {agent_name}:")
                current_agent = agent_name
```

---

## 禁用流式

在模型初始化时设置 `streaming=False`：

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-5.4", streaming=False)
```

部署到 LangSmith 时，对不需要流式输出的模型设置 `streaming=False`。

> 如果模型不支持 `streaming` 参数，使用 `disable_streaming=True`。

---

## v2 流式格式

LangGraph >= 1.1 支持 v2 格式，提供统一输出格式。

### 统一格式

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["updates", "custom"],
    version="v2",
):
    print(chunk["type"])   # "updates" or "custom"
    print(chunk["data"])  # payload
```

### invoke() 返回 GraphOutput

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    version="v2",
)

print(result.value)     # state
print(result.interrupts)  # tuple of Interrupt objects
```

---

## 完整示例

```python
from langchain.agents import create_agent
from langchain.messages import AIMessageChunk, AIMessage, AnyMessage, ToolMessage
from langchain.graph.config import get_stream_writer

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="gpt-5.4",
    tools=[get_weather],
)

def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)

def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")

input_message = {"role": "user", "content": "What is the weather in Boston?"}

for chunk in agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates", "custom"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)
    elif chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source in ("model", "tools"):
                _render_completed_message(update["messages"][-1])
    elif chunk["type"] == "custom":
        print(f"Custom: {chunk['data']}")
```

---

## 相关链接

- 官方文档：https://docs.langchain.com/oss/python/langchain/streaming
- Frontend streaming：构建 React UI
- 流式聊天模型：直接流式 token
- 推理聊天模型：配置推理输出
- 标准内容块：内容块格式
- LangGraph 流式：高级流式选项
