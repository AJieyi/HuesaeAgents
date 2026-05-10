# LangGraph Workflows and Agents 关键代码写法总结

## 目录

- [基础设置](#基础设置)
- [LLM 增强](#llm-增强)
- [Prompt Chaining](#prompt-chaining)
- [Parallelization (并行化)](#parallelization-并行化)
- [Routing (路由)](#routing-路由)
- [Orchestrator-Worker](#orchestrator-worker)
- [Evaluator-Optimizer](#evaluator-optimizer)
- [Agents](#agents)

---

## 基础设置

### 安装依赖

```bash
pip install langchain_core langchain-anthropic langgraph
```

### 初始化 LLM

```python
import os
import getpass
from langchain_anthropic import ChatAnthropic

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}:")

_set_env("ANTHROPIC_API_KEY")

llm = ChatAnthropic(model="claude-sonnet-4-6")
```

---

## LLM 增强

### 结构化输出

```python
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query optimized for web search.")
    justification: str = Field(None, description="Why this query is relevant.")

structured_llm = llm.with_structured_output(SearchQuery)
output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")
```

### 工具绑定

```python
def multiply(a: int, b: int) -> int:
    return a * b

llm_with_tools = llm.bind_tools([multiply])
msg = llm_with_tools.invoke("What is 2 times 3?")
print(msg.tool_calls)
```

---

## Prompt Chaining

每个 LLM 调用处理前一个调用的输出，用于将任务分解为可验证的步骤。

### 典型场景

- 翻译文档
- 验证生成内容的一致性

### 代码示例

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str

def generate_joke(state: State):
    """First LLM call to generate initial joke"""
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}

def check_punchline(state: State):
    """Gate function to check if the joke has a punchline"""
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass"
    return "Fail"

def improve_joke(state: State):
    """Second LLM call to improve the joke"""
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}

def polish_joke(state: State):
    """Third LLM call for final polish"""
    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": msg.content}

# Build workflow
workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)

workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges("generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END})
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)

chain = workflow.compile()

# Invoke
state = chain.invoke({"topic": "cats"})
```

---

## Parallelization (并行化)

多个 LLM 同时处理任务，用于加速或提高置信度。

### 典型场景

- 并行处理文档的不同方面
- 多次运行检查不同标准

### 代码示例

```python
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str

def call_llm_1(state: State):
    msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}

def call_llm_2(state: State):
    msg = llm.invoke(f"Write a story about {state['topic']}")
    return {"story": msg.content}

def call_llm_3(state: State):
    msg = llm.invoke(f"Write a poem about {state['topic']}")
    return {"poem": msg.content}

def aggregator(state: State):
    combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}

parallel_builder = StateGraph(State)
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# 从 START 并行触发多个节点
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")

# 汇聚到 aggregator
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)

parallel_workflow = parallel_builder.compile()
```

---

## Routing (路由)

根据输入内容将任务路由到不同的专门处理流程。

### 典型场景

- 客服系统根据类型路由到不同处理流程（定价、退款、退货等）
- 问答系统路由到不同专业领域

### 代码示例

```python
from typing_extensions import Literal
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(None, description="The next step in the routing process")

router = llm.with_structured_output(Route)

class State(TypedDict):
    input: str
    decision: str
    output: str

def llm_call_1(state: State):
    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_2(state: State):
    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_3(state: State):
    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_router(state: State):
    decision = router.invoke([
        SystemMessage(content="Route the input to story, joke, or poem based on the user's request."),
        HumanMessage(content=state["input"]),
    ])
    return {"decision": decision.step}

def route_decision(state: State):
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"

router_builder = StateGraph(State)
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)

router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {"llm_call_1": "llm_call_1", "llm_call_2": "llm_call_2", "llm_call_3": "llm_call_3"}
)
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

router_workflow = router_builder.compile()
```

---

## Orchestrator-Worker

编排器分解任务、委派给 worker、汇总 worker 输出。

### 典型场景

- 代码生成需要更新多个文件
- 报告生成需要处理多个章节

### Send API

使用 `Send` 动态创建 worker 节点并发送特定输入。

```python
from typing import Annotated, List
import operator
from langgraph.types import Send

class Section(BaseModel):
    name: str = Field(description="Name for this section of the report.")
    description: str = Field(description="Brief overview of the main topics.")

class Sections(BaseModel):
    sections: List[Section] = Field(description="Sections of the report.")

planner = llm.with_structured_output(Sections)

class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]  # 所有 worker 并行写入
    final_report: str

class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]

def orchestrator(state: State):
    """Orchestrator that generates a plan for the report"""
    report_sections = planner.invoke([
        SystemMessage(content="Generate a plan for the report."),
        HumanMessage(content=f"Here is the report topic: {state['topic']}"),
    ])
    return {"sections": report_sections.sections}

def llm_call(state: WorkerState):
    """Worker writes a section of the report"""
    section = llm.invoke([
        SystemMessage(content="Write a report section following the provided name and description."),
        HumanMessage(content=f"Section name: {state['section'].name}, Description: {state['section'].description}"),
    ])
    return {"completed_sections": [section.content]}

def synthesizer(state: State):
    """Synthesize full report from sections"""
    completed = "\n\n ---\n\n".join(state["completed_sections"])
    return {"final_report": completed}

def assign_workers(state: State):
    """Assign a worker to each section"""
    return [Send("llm_call", {"section": s}) for s in state["sections"]]

orchestrator_worker_builder = StateGraph(State)
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges("orchestrator", assign_workers, ["llm_call"])
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

orchestrator_worker = orchestrator_worker_builder.compile()
```

---

## Evaluator-Optimizer

一个 LLM 生成响应，另一个评估。如果需要改进则反馈给生成器循环迭代。

### 典型场景

- 翻译（需要多次迭代达到意义对等）
- 任何有明确成功标准的任务

### 代码示例

```python
from typing import Literal
from pydantic import BaseModel, Field

class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str

class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(description="Decide if the joke is funny.")
    feedback: str = Field(description="If not funny, provide improvement feedback.")

evaluator = llm.with_structured_output(Feedback)

def llm_call_generator(state: State):
    """LLM generates a joke"""
    if state.get("feedback"):
        msg = llm.invoke(f"Write a joke about {state['topic']} but take into account: {state['feedback']}")
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}

def llm_call_evaluator(state: State):
    """LLM evaluates the joke"""
    grade = evaluator.invoke(f"Grade the joke: {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}

def route_joke(state: State):
    if state["funny_or_not"] == "funny":
        return "Accepted"
    return "Rejected + Feedback"

optimizer_builder = StateGraph(State)
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {"Accepted": END, "Rejected + Feedback": "llm_call_generator"}
)

optimizer_workflow = optimizer_builder.compile()
```

---

## Agents

Agent 是动态的，可以自主决定使用哪些工具和如何解决问题。

### 代码示例

```python
from langgraph.graph import MessagesState
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from typing import Literal

# 定义工具
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`."""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`."""
    return a / b

tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            llm_with_tools.invoke(
                [SystemMessage(content="You are a helpful assistant.")] + state["messages"]
            )
        ]
    }

def tool_node(state: dict):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue or stop"""
    if state["messages"][-1].tool_calls:
        return "tool_node"
    return END

agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

# Invoke
messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
```

---

## 模式对比总结

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| **Prompt Chaining** | 顺序处理，每个步骤依赖前一个 | 翻译、验证、文档处理 |
| **Parallelization** | 并行处理独立任务 | 加速、多角度评估 |
| **Routing** | 根据输入类型路由到专门流程 | 客服、分类系统 |
| **Orchestrator-Worker** | 动态分解任务，worker 并行执行 | 代码生成、报告生成 |
| **Evaluator-Optimizer** | 生成-评估循环迭代 | 翻译、优化任务 |
| **Agents** | 自主决策，动态工具调用 | 开放性任务 |

---

## 参考链接

- 官方文档：https://docs.langchain.com/oss/python/langgraph/workflows-agents
- Quickstart：快速开始
- Thinking in LangGraph：设计思想
