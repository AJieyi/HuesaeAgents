# LangGraph Persistence 关键代码写法总结

## 目录

- [概述](#概述)
- [核心概念](#核心概念)
- [基本用法](#基本用法)
- [获取和更新状态](#获取和更新状态)
- [Memory Store](#memory-store)
- [Checkpointer 库](#checkpointer-库)
- [序列化与加密](#序列化与加密)

---

## 概述

LangGraph 内置持久化层，将图状态保存为检查点 (checkpoints)。启用后支持：
- **人机交互** - 检查点支持中断和批准工作流
- **记忆** - 线程间的对话记忆
- **时间旅行** - 回溯和重放先前执行
- **容错** - 节点失败时从最后成功步骤恢复

> 使用 Agent Server 时，checkpointing 自动处理，无需手动配置。

---

## 核心概念

### Thread (线程)

线程是每个检查点保存时的唯一标识符，包含一系列运行的累积状态。

```python
config = {"configurable": {"thread_id": "1"}}
```

### Checkpoint (检查点)

检查点是特定时间点的状态快照，表示为 `StateSnapshot` 对象。

### Super-step

LangGraph 在每个 super-step 边界创建检查点。Super-step 是图中所有调度节点执行的单个"tick"。

### Checkpoint Namespace

| Namespace | 含义 |
|-----------|------|
| `""` (空字符串) | 父（根）图 |
| `"node_name:uuid"` | 子图 |

```python
from langchain_core.runnables import RunnableConfig

def my_node(state: State, config: RunnableConfig):
    checkpoint_ns = config["configurable"]["checkpoint_ns"]
```

---

## 基本用法

### 内存 Checkpoint

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

workflow = StateGraph(State)
workflow.add_node("node_a", node_a)
workflow.add_node("node_b", node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": "", "bar": []}, config)
```

### 生产环境 PostgreSQL

```bash
pip install langgraph-checkpoint-postgres
```

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 自动创建表
    graph = workflow.compile(checkpointer=checkpointer)
```

### 其他 Checkpointer

| 库 | 说明 |
|-----|------|
| `langgraph-checkpoint-sqlite` | SQLite，适合本地开发 |
| `langgraph-checkpoint-postgres` | PostgreSQL，适合生产 |
| `langchain-azure-cosmosdb` | Azure Cosmos DB |

---

## 获取和更新状态

### 获取最新状态

```python
config = {"configurable": {"thread_id": "1"}}
graph.get_state(config)
```

### 获取特定检查点

```python
config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"
    }
}
graph.get_state(config)
```

### StateSnapshot 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `values` | dict | 此时的状态通道值 |
| `next` | tuple[str, ...] | 下一个执行的节点 |
| `config` | dict | 包含 thread_id, checkpoint_ns, checkpoint_id |
| `metadata` | dict | 执行元数据 |
| `created_at` | str | ISO 8601 时间戳 |
| `parent_config` | dict \| None | 前一个检查点的配置 |
| `tasks` | tuple[PregelTask, ...] | 此时执行的任务 |

### 获取状态历史

```python
config = {"configurable": {"thread_id": "1"}}
history = list(graph.get_state_history(config))
```

### 查找特定检查点

```python
# 找到特定节点执行前的检查点
before_node_b = next(s for s in history if s.next == ("node_b",))

# 按步骤号查找
step_2 = next(s for s in history if s.metadata["step"] == 2)

# 找到中断发生的检查点
interrupted = next(s for s in history if s.tasks and any(t.interrupts for t in s.tasks))
```

### 重放 (Replay)

```python
# 使用 prior checkpoint_id 重放
config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "1ef663ba-28f9-6ec4-8001-31981c2c39f8"
    }
}
graph.invoke(None, config)  # 从该检查点后重放
```

### 更新状态

```python
graph.update_state(config, {"foo": "updated_value", "bar": ["new"]})
```

---

## Memory Store

State Schema 定义图执行时填充的键集合。Store 接口支持跨线程共享信息。

### 基本用法

```python
from langgraph.store.memory import InMemoryStore
import uuid

store = InMemoryStore()

user_id = "1"
namespace = (user_id, "memories")

# 存储
memory_id = str(uuid.uuid4())
memory = {"food_preference": "I like pizza"}
store.put(namespace, memory_id, memory)

# 搜索
memories = store.search(namespace)
memories[-1].dict()
```

### 语义搜索

```python
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
        "fields": ["food_preference", "$"]
    }
)

# 自然语言查询
memories = store.search(
    namespace,
    query="What does the user like to eat?",
    limit=3
)
```

### 在 LangGraph 中使用

```python
from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

@dataclass
class Context:
    user_id: str

checkpointer = InMemorySaver()

builder = StateGraph(MessagesState, context_schema=Context)
# ... 添加节点和边 ...
graph = builder.compile(
    checkpointer=checkpointer,
    store=store
)

# 调用时指定 context
config = {"configurable": {"thread_id": "1"}}
for update in graph.stream(
    {"messages": [{"role": "user", "content": "hi"}]},
    config,
    context=Context(user_id="1"),
):
    print(update)
```

### 在节点中访问 Store

```python
async def update_memory(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")

    memory_id = str(uuid.uuid4())
    await runtime.store.aput(namespace, memory_id, {"memory": memory})

async def call_model(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")

    memories = await runtime.store.asearch(
        namespace,
        query=state["messages"][-1].content,
        limit=3
    )
    info = "\n".join([d.value["memory"] for d in memories])
```

---

## Checkpointer 库

| 库 | 类 | 说明 |
|-----|-----|------|
| `langgraph-checkpoint` | `InMemorySaver` | 内存，开发测试 |
| `langgraph-checkpoint-sqlite` | `SqliteSaver` | SQLite，本地开发 |
| `langgraph-checkpoint-postgres` | `PostgresSaver` | PostgreSQL，生产环境 |
| `langchain-azure-cosmosdb` | `CosmosDBSaver` | Azure Cosmos DB |

### Checkpointer 接口方法

| 方法 | 说明 |
|------|------|
| `.put` | 存储检查点及其配置和元数据 |
| `.put_writes` | 存储链接到检查点的中间写入 |
| `.get_tuple` | 获取给定配置的检查点元组 |
| `.list` | 列出匹配给定配置和过滤条件的检查点 |

异步执行时使用：`.aput`, `.aput_writes`, `.aget_tuple`, `.alist`

---

## 序列化与加密

### 默认序列化

`JsonPlusSerializer` 使用 ormsgpack 和 JSON，支持 LangChain/LangGraph 原语、datetime、enum 等。

### Pickle 回退

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

graph = workflow.compile(
    checkpointer=InMemorySaver(
        serde=JsonPlusSerializer(pickle_fallback=True)
    )
)
```

### 加密

```python
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

serde = EncryptedSerializer.from_pycryptodome_aes()  # 读取 LANGGRAPH_AES_KEY
checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db"), serde=serde)
```

### PostgreSQL 加密

```python
from langgraph.checkpoint.postgres import PostgresSaver

serde = EncryptedSerializer.from_pycryptodome_aes()
checkpointer = PostgresSaver.from_conn_string("postgresql://...", serde=serde)
checkpointer.setup()
```

---

## 完整示例

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from langchain.agents import create_agent
from dataclasses import dataclass
from typing import Annotated
import uuid

# 1. 配置 Store
store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
        "fields": ["memory"]
    }
)

# 2. 创建 Checkpointer
checkpointer = InMemorySaver()

# 3. 定义 Context
@dataclass
class Context:
    user_id: str

# 4. 创建 Agent
agent = create_agent(
    model="gpt-5.4",
    tools=[get_weather],
    checkpointer=checkpointer,
    store=store,
    context_schema=Context,
)

# 5. 调用 - 多线程
# 线程 1
config1 = {"configurable": {"thread_id": "1"}}
agent.invoke(
    {"messages": [{"role": "user", "content": "Hi, I'm Bob"}]},
    config1,
    context=Context(user_id="user_123")
)

# 线程 2 - 同一用户，不同会话
config2 = {"configurable": {"thread_id": "2"}}
agent.invoke(
    {"messages": [{"role": "user", "content": "What do you remember about me?"}]},
    config2,
    context=Context(user_id="user_123")  # 跨线程记忆
)

# 6. 时间旅行 - 重放
history = list(agent.get_state_history(config1))
checkpoint_to_replay = history[1]  # 某个历史检查点
agent.invoke(
    None,
    {"configurable": {"thread_id": "1", "checkpoint_id": checkpoint_to_replay.config["configurable"]["checkpoint_id"]}}
)
```

---

## 关键要点

1. **Checkpointer** 是持久化检查点的关键
2. **thread_id** 用于区分不同会话/线程
3. **StateSnapshot** 包含完整的状态快照和元数据
4. **Store** 支持跨线程共享信息（如用户偏好）
5. **Checkpoint Namespace** 区分父图和子图状态
6. 支持多种数据库后端：SQLite、PostgreSQL、Azure Cosmos DB
7. 可选加密所有持久化状态

---

## 参考链接

- 官方文档：https://docs.langchain.com/oss/python/langgraph/persistence
- Checkpointer 库列表：Persistence 文档
- Agent Server：自动处理 checkpointing
- DeltaChannel：优化检查点存储
