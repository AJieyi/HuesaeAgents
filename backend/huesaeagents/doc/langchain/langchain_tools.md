# LangChain Tools 关键代码写法总结

## 目录

- [基础概念](#基础概念)
- [创建工具](#创建工具)
- [自定义工具属性](#自定义工具属性)
- [高级 Schema 定义](#高级-schema-定义)
- [访问上下文 (Runtime)](#访问上下文-runtime)
- [短时记忆 (State)](#短时记忆-state)
- [上下文 (Context)](#上下文-context)
- [长期记忆 (Store)](#长期记忆-store)
- [流式输出 (Stream Writer)](#流式输出-stream-writer)
- [执行信息](#执行信息)
- [服务端信息](#服务端信息)
- [ToolNode](#toolnode)
- [工具返回值](#工具返回值)
- [错误处理](#错误处理)
- [条件路由](#条件路由)

---

## 基础概念

Tools 扩展了 Agent 的能力，让它们能够获取实时数据、执行代码、查询外部数据库等。

- Tools 是可调用的函数，有明确定义的输入输出
- 模型根据对话上下文决定何时调用工具
- 工具名称推荐使用 snake_case（如 `web_search`）

---

## 创建工具

### 基本定义

```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"
```

> **注意：** 类型提示是必需的，用于定义工具的输入 schema。

---

## 自定义工具属性

### 自定义工具名称

```python
@tool("web_search")
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

print(search.name)  # web_search
```

### 自定义工具描述

```python
@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))
```

---

## 高级 Schema 定义

### Pydantic 模型

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(
    location: str,
    units: str = "celsius",
    include_forecast: bool = False
) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```

### JSON Schema

```python
# 通过 args_schema 参数传入 JSON Schema
@tool(args_schema=json_schema)
def my_tool(...) -> str:
    ...
```

---

## 访问上下文 (Runtime)

工具通过 `ToolRuntime` 参数访问运行时信息：

| 组件 | 说明 |
|------|------|
| `state` | 短时记忆 - 当前对话的可变数据 |
| `context` | 上下文 - 不可变的配置数据 |
| `store` | 长期记忆 - 跨会话持久化存储 |
| `stream_writer` | 流式写入器 |
| `execution_info` | 执行信息 |
| `server_info` | 服务端信息 |
| `tool_call_id` | 当前工具调用 ID |

> **保留参数名：** `config` 和 `runtime` 是保留名称，不能用作工具参数。

---

## 短时记忆 (State)

State 表示当前对话期间存在的短时记忆，包含消息历史和自定义字段。

### 访问 state

```python
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage

@tool
def get_last_user_message(runtime: ToolRuntime) -> str:
    """Get the most recent message from the user."""
    messages = runtime.state["messages"]
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return "No user messages found"
```

### 访问自定义 state 字段

```python
@tool
def get_user_preference(pref_name: str, runtime: ToolRuntime) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")
```

### 更新 state

使用 `Command` 更新 agent 的 state：

```python
from langchain.agents import AgentState
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

class CustomState(AgentState):
    user_name: str

@tool
def set_user_name(new_name: str, runtime: ToolRuntime[None, CustomState]) -> Command:
    """Set the user's name in the conversation state."""
    return Command(
        update={
            "user_name": new_name,
            "messages": [
                ToolMessage(
                    content=f"User name set to {new_name}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
```

---

## 上下文 (Context)

Context 提供不可变的配置数据，通过 `runtime.context` 访问：

```python
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

USER_DATABASE = {
    "user123": {"name": "Alice Johnson", "account_type": "Premium", "balance": 5000},
    "user456": {"name": "Bob Smith", "account_type": "Standard", "balance": 1200},
}

@dataclass
class UserContext:
    user_id: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id
    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    return "User not found"

model = ChatOpenAI(model="gpt-5.4")
agent = create_agent(
    model,
    tools=[get_account_info],
    context_schema=UserContext,
    system_prompt="You are a financial assistant."
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my current balance?"}]},
    context=UserContext(user_id="user123")
)
```

---

## 长期记忆 (Store)

Store 提供跨会话持久化的存储，通过 `runtime.store` 访问：

```python
from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI

# 读取记忆
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

# 保存记忆
@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """Save user info."""
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

model = ChatOpenAI(model="gpt-5.4")
store = InMemoryStore()
agent = create_agent(
    model,
    tools=[get_user_info, save_user_info],
    store=store
)

# 第一次会话：保存用户信息
agent.invoke({
    "messages": [{"role": "user", "content": "Save the following user: userid: abc123, name: Foo, age: 25"}]
})

# 第二次会话：获取用户信息
agent.invoke({
    "messages": [{"role": "user", "content": "Get user info for user with id 'abc123'"}]
})
```

---

## 流式输出 (Stream Writer)

工具执行期间发出实时更新：

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    return f"It's always sunny in {city}!"
```

> 注意：`runtime.stream_writer` 需要在 LangGraph 执行上下文中使用。

---

## 执行信息

通过 `runtime.execution_info` 访问：

```python
from langchain.tools import tool, ToolRuntime

@tool
def log_execution_context(runtime: ToolRuntime) -> str:
    """Log execution identity information."""
    info = runtime.execution_info
    print(f"Thread: {info.thread_id}, Run: {info.run_id}")
    print(f"Attempt: {info.node_attempt}")
    return "done"
```

---

## 服务端信息

当工具运行在 LangGraph Server 上时，通过 `runtime.server_info` 访问：

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_assistant_scoped_data(runtime: ToolRuntime) -> str:
    """Fetch data scoped to the current assistant."""
    server = runtime.server_info
    if server is not None:
        print(f"Assistant: {server.assistant_id}, Graph: {server.graph_id}")
        if server.user is not None:
            print(f"User: {server.user.identity}")
    return "done"
```

> `server_info` 在非 LangGraph Server 环境下为 None。

---

## ToolNode

ToolNode 是预建的节点，用于在 LangGraph 工作流中执行工具。

### 基本用法

```python
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

# 创建 ToolNode
tool_node = ToolNode([search, calculator])

# 在图构建器中使用
builder = StateGraph(MessagesState)
builder.add_node("tools", tool_node)
# ... 添加其他节点和边
```

---

## 工具返回值

### 返回字符串

返回人类可读的文本：

```python
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"It is currently sunny in {city}."
```

### 返回对象

返回结构化数据供模型解析：

```python
@tool
def get_weather_data(city: str) -> dict:
    """Get structured weather data for a city."""
    return {
        "city": city,
        "temperature_c": 22,
        "conditions": "sunny",
    }
```

### 返回 Command

需要更新 graph state 时使用：

```python
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

@tool
def set_language(language: str, runtime: ToolRuntime) -> Command:
    """Set the preferred response language."""
    return Command(
        update={
            "preferred_language": language,
            "messages": [
                ToolMessage(
                    content=f"Language set to {language}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
```

---

## 错误处理

```python
from langgraph.prebuilt import ToolNode

# 默认：捕获调用错误，重新抛出执行错误
tool_node = ToolNode(tools)

# 捕获所有错误并返回错误消息给 LLM
tool_node = ToolNode(tools, handle_tool_errors=True)

# 自定义错误消息
tool_node = ToolNode(tools, handle_tool_errors="Something went wrong, please try again.")

# 自定义错误处理器
def handle_error(e: ValueError) -> str:
    return f"Invalid input: {e}"

tool_node = ToolNode(tools, handle_tool_errors=handle_error)

# 只捕获特定异常类型
tool_node = ToolNode(tools, handle_tool_errors=(ValueError, TypeError))
```

---

## 条件路由

使用 `tools_condition` 根据 LLM 是否调用工具进行条件路由：

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, MessagesState, START, END

builder = StateGraph(MessagesState)
builder.add_node("llm", call_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)  # 路由到 "tools" 或 END
builder.add_edge("tools", "llm")

graph = builder.compile()
```

---

## 完整示例

```python
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.types import Command
from dataclasses import dataclass

# 1. 定义工具
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def set_user_language(language: str, runtime: ToolRuntime) -> Command:
    """Set the user's preferred language."""
    return Command(
        update={
            "preferred_language": language,
            "messages": [
                ToolMessage(
                    content=f"Language preference set to {language}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )

# 2. 创建 Agent
model = ChatOpenAI(model="gpt-5.4")
agent = create_agent(
    model,
    tools=[get_weather, set_user_language],
    system_prompt="You are a helpful assistant."
)

# 3. 调用 Agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Boston?"}]
})

print(result["messages"][-1].content_blocks)
```

---

## 参考链接

- 官方文档：https://docs.langchain.com/oss/python/langchain/tools
- 预建工具列表：Tools and toolkits integration page
- Server-side tool use：服务端工具调用
