# LangChain Short-term Memory 关键代码写法总结

## 目录

- [概述](#概述)
- [基本用法](#基本用法)
- [生产环境配置](#生产环境配置)
- [自定义 Agent Memory](#自定义-agent-memory)
- [常见模式](#常见模式)
- [访问 Memory](#访问-memory)

---

## 概述

短期记忆让应用记住单个线程或对话中的先前交互。对话历史是最常见的短期记忆形式。

**挑战：**
- 长对话可能超出 LLM 的上下文窗口
- 即使模型支持完整上下文长度，长上下文仍可能导致性能下降

**解决方案：**
- 修整消息 (Trim)
- 删除消息 (Delete)
- 总结消息 (Summarize)

---

## 基本用法

使用 `checkpointer` 添加短期记忆（线程级持久化）：

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    "gpt-5.4",
    tools=[get_user_info],
    checkpointer=InMemorySaver(),
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    {"configurable": {"thread_id": "1"}},
)
```

---

## 生产环境配置

使用数据库支持的 checkpointer：

```bash
pip install langgraph-checkpoint-postgres
```

```python
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 自动创建表
    agent = create_agent(
        "gpt-5.4",
        tools=[get_user_info],
        checkpointer=checkpointer,
    )
```

**其他 Checkpointer 选项：**
- SQLite
- Postgres
- Azure Cosmos DB

详见 Persistence 文档。

---

## 自定义 Agent Memory

扩展 `AgentState` 添加自定义字段：

```python
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver

class CustomAgentState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent(
    "gpt-5.4",
    tools=[get_user_info],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    {"configurable": {"thread_id": "1"}},
)
```

---

## 常见模式

### 修整消息 (Trim Messages)

使用 `@before_model` 中间件修整消息历史：

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from typing import Any

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]
    if len(messages) <= 3:
        return None

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages]}

agent = create_agent(
    "gpt-5-nano",
    tools=[],
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)
```

### 删除消息 (Delete Messages)

删除特定消息：

```python
from langchain.messages import RemoveMessage

def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        # 删除最早的两条消息
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
```

删除所有消息：

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES

def delete_all_messages(state):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

使用 `@after_model` 中间件：

```python
from langchain.agents.middleware import after_model

@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    messages = state["messages"]
    if len(messages) > 2:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None
```

### 总结消息 (Summarize Messages)

使用 `SummarizationMiddleware` 总结消息历史：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-5.4",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=("tokens", 4000),  # 触发阈值
            keep=("messages", 20),      # 保留消息数
        )
    ],
    checkpointer=InMemorySaver(),
)
```

---

## 访问 Memory

### 在工具中读取短期记忆

使用 `ToolRuntime` 访问状态：

```python
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime

class CustomState(AgentState):
    user_id: str

@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """Look up user info."""
    user_id = runtime.state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_user_info],
    state_schema=CustomState,
)

result = agent.invoke(
    {"messages": "look up user information", "user_id": "user_123"}
)
```

### 从工具写入短期记忆

使用 `Command` 返回状态更新：

```python
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel

class CustomState(AgentState):
    user_name: str

class CustomContext(BaseModel):
    user_id: str

@tool
def update_user_info(runtime: ToolRuntime[CustomContext, CustomState]) -> Command:
    """Look up and update user info."""
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"

    return Command(
        update={
            "user_name": name,
            "messages": [
                ToolMessage(
                    "Successfully looked up user information",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )

@tool
def greet(runtime: ToolRuntime[CustomContext, CustomState]) -> str | Command:
    """Use this to greet the user once you found their info."""
    user_name = runtime.state.get("user_name", None)
    if user_name is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "Please call the 'update_user_info' tool first.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    return f"Hello {user_name}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[update_user_info, greet],
    state_schema=CustomState,
    context_schema=CustomContext,
)
```

### 在 Prompt 中访问

使用 `@dynamic_prompt` 创建动态提示词：

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from typing import TypedDict

class CustomContext(TypedDict):
    user_name: str

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    return f"You are a helpful assistant. Address the user as {user_name}."

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext,
)
```

### 在 @before_model 中访问

```python
from langchain.agents.middleware import before_model

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state["messages"]
    if len(messages) <= 3:
        return None
    # ... 修整逻辑
    return {"messages": [...]}
```

### 在 @after_model 中访问

```python
from langchain.agents.middleware import after_model

@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove messages containing sensitive words."""
    STOP_WORDS = ["password", "secret"]
    last_message = state["messages"][-1]

    if any(word in last_message.content for word in STOP_WORDS):
        return {"messages": [RemoveMessage(id=last_message.id)]}
    return None
```

---

## 完整示例

```python
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import SummarizationMiddleware, before_model
from langchain.tools import tool, ToolRuntime
from langchain.messages import RemoveMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel
from typing import Any

# 1. 自定义状态
class CustomAgentState(AgentState):
    user_name: str
    session_data: dict

class CustomContext(BaseModel):
    user_id: str

# 2. 工具定义
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"The weather in {city} is sunny!"

@tool
def remember_name(runtime: ToolRuntime[CustomContext, CustomAgentState]) -> Command:
    """Remember the user's name."""
    name = "User"  # 从某处获取
    return Command(
        update={
            "user_name": name,
            "messages": [
                ToolMessage(f"Remembered name: {name}", tool_call_id=runtime.tool_call_id)
            ],
        }
    )

# 3. 消息修整中间件
@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state["messages"]
    if len(messages) <= 5:
        return None
    # 保留系统消息和最近的消息
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), messages[0]] + messages[-4:]}

# 4. 创建 Agent
agent = create_agent(
    model="gpt-5.4",
    tools=[get_weather, remember_name],
    state_schema=CustomAgentState,
    context_schema=CustomContext,
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)

# 5. 调用
config = {"configurable": {"thread_id": "session_123"}}

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hi, I'm Bob"}],
        "user_name": "Bob",
    },
    config,
)
```

---

## 关键要点

1. **checkpointer** 是启用短期记忆的关键
2. **state_schema** 扩展默认状态添加自定义字段
3. **thread_id** 用于区分不同会话
4. **Trim/Delete/Summarize** 是管理长对话的常见模式
5. **ToolRuntime** 允许工具读写状态
6. **@before_model/@after_model/@dynamic_prompt** 用于中间件中访问状态

---

## 参考链接

- 官方文档：https://docs.langchain.com/oss/python/langchain/short-term-memory
- Persistence：Checkpointer 库列表
- Long-term memory：跨会话持久化存储
