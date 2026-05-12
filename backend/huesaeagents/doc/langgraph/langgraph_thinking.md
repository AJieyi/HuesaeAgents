# Thinking in LangGraph 关键代码写法总结

## 目录

- [核心概念](#核心概念)
- [设计流程五步法](#设计流程五步法)
- [State 设计](#state-设计)
- [节点构建](#节点构建)
- [错误处理](#错误处理)
- [条件路由](#条件路由)
- [人机交互](#人机交互)
- [图构建与编译](#图构建与编译)
- [测试与执行](#测试与执行)

---

## 核心概念

LangGraph 构建 Agent 的核心思想：

1. **节点 (Nodes)** - 将工作流分解为离散步骤
2. **边 (Edges)** - 描述节点之间的决策和转换
3. **共享 State** - 节点可以读写共享状态

---

## 设计流程五步法

### Step 1: 将工作流映射为离散步骤

识别流程中的关键节点：

```
Read Email → Classify Intent → Doc Search / Bug Track / Human Review → Draft Reply → Send Reply
```

### Step 2: 识别每个步骤需要做什么

| 步骤类型 | 适用场景 |
|----------|----------|
| LLM steps | 理解、分析、生成文本、推理决策 |
| Data steps | 从外部源检索信息 |
| Action steps | 执行外部操作 |
| User input steps | 需要人工干预 |

### Step 3: 设计 State

### Step 4: 构建节点

### Step 5: 连接节点

---

## State 设计

### 设计原则

- **存储原始数据**，不要存储格式化文本
- **按需格式化提示词**
- 这样不同节点可以用不同方式格式化同一数据

### State 定义示例

```python
from typing import TypedDict, Literal

class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

class EmailAgentState(TypedDict):
    # 原始邮件数据
    email_content: str
    sender_email: str
    email_id: str

    # 分类结果
    classification: EmailClassification | None

    # 原始搜索/API结果
    search_results: list[str] | None
    customer_history: dict | None

    # 生成的内容
    draft_response: str | None
    messages: list[str] | None
```

---

## 节点构建

节点是 Python 函数，接收当前状态并返回状态更新。

### 基本节点模式

```python
def read_email(state: EmailAgentState) -> dict:
    """Extract and parse email content"""
    return {
        "messages": [HumanMessage(content=f"Processing email: {state['email_content']}")]
    }
```

### 返回 Command 进行路由

```python
from typing import Literal
from langgraph.types import Command

def classify_intent(state: EmailAgentState) -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]:
    """Use LLM to classify email intent and urgency, then route accordingly"""

    # 创建结构化 LLM
    structured_llm = llm.with_structured_output(EmailClassification)

    # 按需格式化提示词
    classification_prompt = f"""
    Analyze this customer email and classify it:
    Email: {state['email_content']}
    From: {state['sender_email']}
    """

    # 获取结构化响应
    classification = structured_llm.invoke(classification_prompt)

    # 根据分类决定下一个节点
    if classification['intent'] == 'billing' or classification['urgency'] == 'critical':
        goto = "human_review"
    elif classification['intent'] in ['question', 'feature']:
        goto = "search_documentation"
    elif classification['intent'] == 'bug':
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    return Command(update={"classification": classification}, goto=goto)
```

### 数据节点

```python
def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Search knowledge base for relevant information"""
    classification = state.get('classification', {})
    query = f"{classification.get('intent', '')} {classification.get('topic', '')}"

    try:
        search_results = ["Reset password via Settings > Security > Change Password"]
    except SearchAPIError as e:
        search_results = [f"Search temporarily unavailable: {str(e)}"]

    return Command(update={"search_results": search_results}, goto="draft_response")
```

### LLM 节点

```python
def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """Generate response using context and route based on quality"""
    classification = state.get('classification', {})

    # 按需格式化上下文
    context_sections = []
    if state.get('search_results'):
        formatted_docs = "\n".join([f"- {doc}" for doc in state['search_results']])
        context_sections.append(f"Relevant documentation:\n{formatted_docs}")

    if state.get('customer_history'):
        context_sections.append(f"Customer tier: {state['customer_history'].get('tier', 'standard')}")

    draft_prompt = f"""
    Draft a response to this customer email: {state['email_content']}
    Email intent: {classification.get('intent', 'unknown')}
    Urgency level: {classification.get('urgency', 'medium')}
    {chr(10).join(context_sections)}
    """

    response = llm.invoke(draft_prompt)

    # 根据紧急程度决定是否需要人工审核
    needs_review = (
        classification.get('urgency') in ['high', 'critical'] or
        classification.get('intent') == 'complex'
    )

    goto = "human_review" if needs_review else "send_reply"

    return Command(update={"draft_response": response.content}, goto=goto)
```

### 动作节点

```python
def send_reply(state: EmailAgentState) -> dict:
    """Send the email response"""
    email_service.send(state["draft_response"])
    return {}  # 同步操作不需要更新状态
```

---

## 错误处理

### 错误类型与处理策略

| 错误类型 | 修复者 | 策略 |
|----------|--------|------|
| 瞬态错误 (网络、限流) | 系统自动 | 重试策略 |
| LLM 可恢复错误 (工具失败) | LLM | 将错误存入状态并循环 |
| 用户可修复错误 (缺少信息) | 用户 | interrupt() 暂停 |
| 意外错误 | 开发者 | 让其冒泡 |

### 重试策略

```python
from langgraph.types import RetryPolicy

workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0)
)
```

### LLM 可恢复错误

```python
from langgraph.types import Command

def execute_tool(state: State) -> Command[Literal["agent", "execute_tool"]]:
    try:
        result = run_tool(state['tool_call'])
        return Command(update={"tool_result": result}, goto="agent")
    except ToolError as e:
        # 让 LLM 看到错误并重试
        return Command(update={"tool_result": f"Tool error: {str(e)}"}, goto="agent")
```

### 用户可修复错误 (interrupt)

```python
def lookup_customer_history(state: State) -> Command[Literal["draft_response"]]:
    if not state.get('customer_id'):
        user_input = interrupt({
            "message": "Customer ID needed",
            "request": "Please provide the customer's account ID"
        })
        return Command(update={"customer_id": user_input['customer_id']}, goto="lookup_customer_history")

    customer_data = fetch_customer_history(state['customer_id'])
    return Command(update={"customer_history": customer_data}, goto="draft_response")
```

### 错误处理器 (Saga/Compensation)

```python
from langgraph.errors import NodeError
from langgraph.types import Command, RetryPolicy

def payment_error_handler(state: State, error: NodeError) -> Command:
    return Command(
        update={"status": f"compensated: {error.error}"},
        goto="finalize",
    )

workflow.add_node(
    "charge_payment",
    charge_payment,
    retry_policy=RetryPolicy(max_attempts=3, retry_on=ConnectionError),
    error_handler=payment_error_handler,
)
```

---

## 条件路由

节点内通过 `Command` 返回 `goto` 进行路由决策：

```python
def classify_intent(state: EmailAgentState) -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]:
    # ... 分类逻辑
    return Command(update={"classification": classification}, goto=goto)
```

### tools_condition

```python
from langgraph.prebuilt import ToolNode, tools_condition

builder.add_conditional_edges("llm", tools_condition)  # 路由到 "tools" 或 END
```

---

## 人机交互

### interrupt() 用法

```python
def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """Pause for human review using interrupt and route based on decision"""

    # interrupt() 必须放在最前面
    human_decision = interrupt({
        "email_id": state.get('email_id', ''),
        "original_email": state.get('email_content', ''),
        "draft_response": state.get('draft_response', ''),
        "urgency": classification.get('urgency'),
        "intent": classification.get('intent'),
        "action": "Please review and approve/edit this response"
    })

    # 处理人工决策
    if human_decision.get("approved"):
        return Command(
            update={"draft_response": human_decision.get("edited_response", state.get('draft_response', ''))},
            goto="send_reply"
        )
    else:
        return Command(update={}, goto=END)
```

---

## 图构建与编译

### 基本图构建

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 创建图
workflow = StateGraph(EmailAgentState)

# 添加节点
workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("search_documentation", search_documentation, retry_policy=RetryPolicy(max_attempts=3))
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

# 添加边
workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)

# 编译 (需要 checkpointer 实现 interrupt 持久化)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

---

## 测试与执行

### 运行 Agent

```python
# 初始状态
initial_state = {
    "email_content": "I was charged twice for my subscription! This is urgent!",
    "sender_email": "customer@example.com",
    "email_id": "email_123",
    "messages": []
}

# 使用 thread_id 实现持久化
config = {"configurable": {"thread_id": "customer_123"}}

# 运行
result = app.invoke(initial_state, config)

# 图会在 human_review 处暂停
print(f"human review interrupt: {result['__interrupt__']}")
```

### 恢复执行

```python
from langgraph.types import Command

human_response = Command(resume={
    "approved": True,
    "edited_response": "We sincerely apologize for the double charge. I've initiated an immediate refund..."
})

# 恢复执行
final_result = app.invoke(human_response, config)
print(f"Email sent successfully!")
```

---

## 关键设计原则

1. **分解为离散步骤** - 每个节点做一件事，便于流式更新、持久化、调试
2. **State 是共享内存** - 存储原始数据，按需格式化提示词
3. **节点是函数** - 接收状态、执行工作、返回更新
4. **错误是流程的一部分** - 瞬态错误重试、LLM 可恢复错误循环、用户修复错误暂停
5. **人工输入是一等的** - `interrupt()` 暂停执行，保存状态，恢复后精确继续
6. **图结构自然涌现** - 定义必要的边，节点处理自己的路由逻辑

---

## 完整示例结构

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, RetryPolicy, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage

llm = ChatOpenAI(model="gpt-5-nano")

# 1. 定义 State
class EmailAgentState(TypedDict):
    email_content: str
    sender_email: str
    email_id: str
    classification: dict | None
    search_results: list[str] | None
    customer_history: dict | None
    draft_response: str | None
    messages: list[str] | None

# 2. 定义节点
def read_email(state: EmailAgentState) -> dict: ...
def classify_intent(state: EmailAgentState) -> Command[...]: ...
def search_documentation(state: EmailAgentState) -> Command[...]: ...
def bug_tracking(state: EmailAgentState) -> Command[...]: ...
def draft_response(state: EmailAgentState) -> Command[...]: ...
def human_review(state: EmailAgentState) -> Command[...]: ...
def send_reply(state: EmailAgentState) -> dict: ...

# 3. 构建图
workflow = StateGraph(EmailAgentState)
workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("search_documentation", search_documentation, retry_policy=RetryPolicy(max_attempts=3))
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)

# 4. 编译
app = workflow.compile(checkpointer=MemorySaver())

# 5. 运行
result = app.invoke(initial_state, config)
```

---

## 参考链接

- 官方文档：https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph
- Human-in-the-loop patterns
- Subgraphs
- Streaming
- Observability (LangSmith)
