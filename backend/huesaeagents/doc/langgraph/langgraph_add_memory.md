# LangGraph Add Memory 关键代码写法总结

## 目录

- [概述](#概述)
- [短期记忆 (Short-term Memory)](#短期记忆-short-term-memory)
- [生产环境配置](#生产环境配置)
- [子图中的短期记忆](#子图中的短期记忆)
- [长期记忆 (Long-term Memory)](#长期记忆-long-term-memory)
- [在节点中访问 Store](#在节点中访问-store)
- [语义搜索](#语义搜索)
- [管理短期记忆](#管理短期记忆)
- [管理检查点](#管理检查点)
- [数据库管理](#数据库管理)

---

## 概述

LangGraph 中的记忆分为两种类型：

- **短期记忆** (线程级持久化) - 启用多轮对话
- **长期记忆** - 跨会话存储用户特定或应用级数据

---

## 短期记忆 (Short-term Memory)

短期记忆通过 checkpointer 实现，支持多轮对话：

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

checkpointer = InMemorySaver()
builder = StateGraph(...)
graph = builder.compile(checkpointer=checkpointer)

graph.invoke(
    {"messages": [{"role": "user", "content": "hi! i am Bob"}]},
    {"configurable": {"thread_id": "1"}},
)
```

---

## 生产环境配置

### PostgreSQL Checkpointer

```bash
pip install -U "psycopg[binary,pool]" langgraph langgraph-checkpoint-postgres
```

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # checkpointer.setup()  # 首次使用时调用
    builder = StateGraph(...)
    graph = builder.compile(checkpointer=checkpointer)
```

**异步版本：**

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # await checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
```

### MongoDB Checkpointer

```bash
pip install -U pymongo langgraph langgraph-checkpoint-mongodb
```

```python
from langgraph.checkpoint.mongodb import MongoDBSaver

MONGODB_URI = "localhost:27017"
with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

**异步版本：**

```python
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

async with AsyncMongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

### Redis Checkpointer

```bash
pip install -U langgraph langgraph-checkpoint-redis
```

```python
from langgraph.checkpoint.redis import RedisSaver

DB_URI = "redis://localhost:6379"
with RedisSaver.from_conn_string(DB_URI) as checkpointer:
    # checkpointer.setup()  # 首次使用时调用
    graph = builder.compile(checkpointer=checkpointer)
```

**异步版本：**

```python
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

async with AsyncRedisSaver.from_conn_string(DB_URI) as checkpointer:
    # await checkpointer.asetup()
    graph = builder.compile(checkpointer=checkpointer)
```

### 完整示例：Postgres Checkpointer

```python
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres import PostgresSaver

model = init_chat_model(model="claude-haiku-4-5-20251001")

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "1"}}

    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
        config,
        stream_mode="values",
    ):
        chunk["messages"][-1].pretty_print()

    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "what's my name?"}]},
        config,
        stream_mode="values",
    ):
        chunk["messages"][-1].pretty_print()
```

---

## 子图中的短期记忆

如果图包含子图，只需在编译父图时提供 checkpointer，LangGraph 会自动将 checkpointer 传播到子图：

```python
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict

class State(TypedDict):
    foo: str

# 子图
def subgraph_node_1(state: State):
    return {"foo": state["foo"] + "bar"}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

# 父图
builder = StateGraph(State)
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

配置子图特定检查点行为：

```python
subgraph = subgraph_builder.compile(checkpointer=True)
```

---

## 长期记忆 (Long-term Memory)

长期记忆使用 Store 接口，跨会话存储用户数据：

```python
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph

store = InMemoryStore()
builder = StateGraph(...)
graph = builder.compile(store=store)
```

### 生产环境 Store

**Postgres Store：**

```bash
pip install -U "psycopg[binary,pool]" langgraph langgraph-checkpoint-postgres
```

```python
from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresStore.from_conn_string(DB_URI) as store:
    # store.setup()  # 首次使用时调用
    graph = builder.compile(store=store)
```

**Redis Store：**

```bash
pip install -U langgraph langgraph-checkpoint-redis
```

```python
from langgraph.store.redis import RedisStore

DB_URI = "redis://localhost:6379"
with RedisStore.from_conn_string(DB_URI) as store:
    # store.setup()  # 首次使用时调用
    graph = builder.compile(store=store)
```

---

## 在节点中访问 Store

编译图时配置 store 后，LangGraph 会自动将 store 注入节点函数。通过 Runtime 对象访问 store：

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, MessagesState, START
import uuid

@dataclass
class Context:
    user_id: str

async def call_model(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")

    # 搜索相关记忆
    memories = await runtime.store.asearch(
        namespace,
        query=state["messages"][-1].content,
        limit=3,
    )
    info = "\n".join([d.value["data"] for d in memories])

    # 存储新记忆
    await runtime.store.aput(
        namespace,
        str(uuid.uuid4()),
        {"data": "User prefers dark mode"},
    )

builder = StateGraph(MessagesState, context_schema=Context)
builder.add_node(call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(store=store)

# 调用时传递 context
graph.invoke(
    {"messages": [{"role": "user", "content": "hi"}]},
    {"configurable": {"thread_id": "1"}},
    context=Context(user_id="1"),
)
```

### 完整示例：长期记忆

```python
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.runtime import Runtime
import uuid

model = init_chat_model(model="claude-haiku-4-5-20251001")

@dataclass
class Context:
    user_id: str

async def call_model(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = ("memories", user_id)

    # 语义搜索记忆
    memories = await runtime.store.asearch(
        namespace,
        query=str(state["messages"][-1].content),
    )
    info = "\n".join([d.value["data"] for d in memories])
    system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

    # 如果用户要求记住，存储新记忆
    last_message = state["messages"][-1]
    if "remember" in last_message.content.lower():
        memory = "User name is Bob"
        await runtime.store.aput(namespace, str(uuid.uuid4()), {"data": memory})

    response = await model.ainvoke(
        [{"role": "system", "content": system_msg}] + state["messages"]
    )
    return {"messages": response}

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
async with (
    AsyncPostgresStore.from_conn_string(DB_URI) as store,
    AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
):
    # await store.setup()
    # await checkpointer.setup()

    builder = StateGraph(MessagesState, context_schema=Context)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")
    graph = builder.compile(checkpointer=checkpointer, store=store)

    config = {"configurable": {"thread_id": "1"}}
    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": "Hi! Remember: my name is Bob"}]},
        config,
        stream_mode="values",
        context=Context(user_id="1"),
    ):
        chunk["messages"][-1].pretty_print()

    # 跨线程（不同会话）访问同一用户记忆
    config = {"configurable": {"thread_id": "2"}}
    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": "what is my name?"}]},
        config,
        stream_mode="values",
        context=Context(user_id="1"),
    ):
        chunk["messages"][-1].pretty_print()
```

---

## 语义搜索

启用语义搜索，让代理通过语义相似度搜索记忆：

```python
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

# 创建带语义搜索的 store
embeddings = init_embeddings("openai:text-embedding-3-small")
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
    }
)

store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
store.put(("user_123", "memories"), "2", {"text": "I am a plumber"})

# 语义搜索
items = store.search(
    ("user_123", "memories"),
    query="I'm hungry",
    limit=1,
)
```

### 完整示例：带语义搜索的长期记忆

```python
from langchain.embeddings import init_embeddings
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.runtime import Runtime

model = init_chat_model("gpt-5.4-mini")

# 创建带语义搜索的 store
embeddings = init_embeddings("openai:text-embedding-3-small")
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
    }
)

store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
store.put(("user_123", "memories"), "2", {"text": "I am a plumber"})

async def chat(state: MessagesState, runtime: Runtime):
    # 基于用户最后一条消息搜索
    items = await runtime.store.asearch(
        ("user_123", "memories"),
        query=state["messages"][-1].content,
        limit=2,
    )
    memories = "\n".join(item.value["text"] for item in items)
    memories = f"## Memories of user\n{ memories }" if memories else ""

    response = await model.ainvoke(
        [
            {"role": "system", "content": f"You are a helpful assistant.\n{ memories }"},
            *state["messages"],
        ]
    )
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node(chat)
builder.add_edge(START, "chat")
graph = builder.compile(store=store)

async for message, metadata in graph.astream(
    input={"messages": [{"role": "user", "content": "I'm hungry"}]},
    stream_mode="messages",
):
    print(message.content, end="")
```

---

## 管理短期记忆

启用短期记忆后，长对话可能超出 LLM 的上下文窗口。常用解决方案：

- **Trim messages** - 删除前 N 或后 N 条消息
- **Delete messages** - 从状态中永久删除消息
- **Summarize messages** - 总结早期消息并替换

### Trim Messages

使用 `trim_messages` 函数修整消息历史：

```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

def call_model(state: MessagesState):
    messages = trim_messages(
        state["messages"],
        strategy="last",  # 保留最后 max_tokens
        token_counter=count_tokens_approximately,
        max_tokens=128,
        start_on="human",
        end_on=("human", "tool"),
    )
    response = model.invoke(messages)
    return {"messages": [response]}
```

**完整示例：**

```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

model = init_chat_model("claude-sonnet-4-6")

def call_model(state: MessagesState):
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=128,
        start_on="human",
        end_on=("human", "tool"),
    )
    response = model.invoke(messages)
    return {"messages": [response]}

checkpointer = InMemorySaver()
builder = StateGraph(MessagesState)
builder.add_node(call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "hi, my name is bob"}, config)
graph.invoke({"messages": "write a short poem about cats"}, config)
graph.invoke({"messages": "now do the same but for dogs"}, config)
final_response = graph.invoke({"messages": "what's my name?"}, config)
final_response["messages"][-1].pretty_print()
```

### Delete Messages

使用 `RemoveMessage` 删除特定消息：

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

def delete_messages(state):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

**完整示例：**

```python
from langchain.messages import RemoveMessage

def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

builder = StateGraph(MessagesState)
builder.add_sequence([call_model, delete_messages])
builder.add_edge(START, "call_model")

checkpointer = InMemorySaver()
app = builder.compile(checkpointer=checkpointer)

for event in app.stream(
    {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
    config,
    stream_mode="values",
):
    print([(message.type, message.content) for message in event["messages"]])

for event in app.stream(
    {"messages": [{"role": "user", "content": "what's my name?"}]},
    config,
    stream_mode="values",
):
    print([(message.type, message.content) for message in event["messages"]])

# 输出示例：
# [('human', "hi! I'm bob")]
# [('human', "hi! I'm bob"), ('ai', 'Hi Bob! ...')]
# [('human', "hi! I'm bob"), ('ai', 'Hi Bob! ...'), ('human', "what's my name?")]
# [('human', "hi! I'm bob"), ('ai', 'Hi Bob! ...'), ('human', "what's my name?"), ('ai', 'Your name is Bob.')]
# [('human', "what's my name?"), ('ai', 'Your name is Bob.')]  <- 旧消息被删除
```

### Summarize Messages

扩展 MessagesState 添加 summary 键：

```python
from langgraph.graph import MessagesState

class State(MessagesState):
    summary: str
```

使用 SummarizationNode 自动总结：

```python
from typing import Any, TypedDict
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode, RunningSummary

model = init_chat_model("claude-sonnet-4-6")
summarization_model = model.bind(max_tokens=128)

class State(MessagesState):
    context: dict[str, RunningSummary]

class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, RunningSummary]

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)

def call_model(state: LLMInputState):
    response = model.invoke(state["summarized_messages"])
    return {"messages": [response]}

checkpointer = InMemorySaver()
builder = StateGraph(State)
builder.add_node(call_model)
builder.add_node("summarize", summarization_node)
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "call_model")
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "hi, my name is bob"}, config)
graph.invoke({"messages": "write a short poem about cats"}, config)
graph.invoke({"messages": "now do the same but for dogs"}, config)
final_response = graph.invoke({"messages": "what's my name?"}, config)
final_response["messages"][-1].pretty_print()
print("\nSummary:", final_response["context"]["running_summary"].summary)
```

---

## 管理检查点

### 查看线程状态

**Graph API：**

```python
config = {"configurable": {"thread_id": "1"}}
graph.get_state(config)
```

**Checkpointer API：**

```python
config = {"configurable": {"thread_id": "1"}}
checkpointer.get_tuple(config)
```

返回 `StateSnapshot`，包含：
- `values` - 状态通道值
- `next` - 下一个执行的节点
- `config` - 包含 thread_id, checkpoint_ns, checkpoint_id
- `metadata` - 执行元数据
- `created_at` - ISO 8601 时间戳
- `parent_config` - 前一个检查点的配置

### 查看线程历史

**Graph API：**

```python
config = {"configurable": {"thread_id": "1"}}
list(graph.get_state_history(config))
```

**Checkpointer API：**

```python
config = {"configurable": {"thread_id": "1"}}
list(checkpointer.list(config))
```

### 删除线程的所有检查点

```python
thread_id = "1"
checkpointer.delete_thread(thread_id)
```

---

## 数据库管理

使用数据库支持的持久化（如 Postgres、Redis）时，需要在首次使用前运行迁移来设置所需的 schema。

每个数据库特定库都定义了 `setup()` 方法来运行迁移：

```python
# 首次使用时调用
checkpointer.setup()
store.setup()
```

建议将迁移作为独立的部署步骤运行，或确保在服务器启动时运行。

---

## 关键要点

1. **短期记忆** 通过 checkpointer 实现，使用 `thread_id` 区分不同会话
2. **长期记忆** 通过 Store 接口实现，跨会话持久化存储用户数据
3. **语义搜索** 需要配置 embeddings，支持自然语言查询记忆
4. **消息管理**：Trim（按 token 修整）、Delete（删除消息）、Summarize（总结消息）
5. **检查点管理**：通过 `get_state`、`get_state_history`、`delete_thread` 管理线程状态
6. **子图记忆**：只需在父图编译时提供 checkpointer，自动传播到子图
7. **数据库迁移**：首次使用前调用 `setup()` 方法

---

## 参考链接

- 官方文档：https://docs.langchain.com/oss/python/langgraph/add-memory
- Checkpointers：各种数据库支持的 checkpointer
- Stores：各种数据库支持的 store
- Memory：长期记忆管理
