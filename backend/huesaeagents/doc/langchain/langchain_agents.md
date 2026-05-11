# LangChain Agents 关键代码写法总结

## 目录

- [基础创建 Agent](#基础创建-agent)
- [Model 配置](#model-配置)
- [Tools 工具](#tools-工具)
- [ReAct 循环](#react-循环)
- [System Prompt](#system-prompt)
- [结构化输出](#结构化输出)
- [状态与记忆](#状态与记忆)
- [流式输出](#流式输出)
- [中间件](#中间件)

---

## 基础创建 Agent

```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-5.4",
    tools=[get_weather, search],
    system_prompt="You are a helpful assistant.",
)
```

### 调用 Agent

```python
# 普通调用
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]
})

# 流式调用
for chunk in agent.stream({"messages": [{"role": "user", "content": "..."}]}):
    print(chunk)
```

---

## Model 配置

### 静态模型（模型标识符）

```python
from langchain.agents import create_agent

agent = create_agent("openai:gpt-5.4", tools=tools)
```

### 静态模型（直接初始化）

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5.4",
    temperature=0.1,
    max_tokens=1000,
    timeout=30,
)
agent = create_agent(model, tools=tools)
```

### 动态模型选择

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

basic_model = ChatOpenAI(model="gpt-5.4-mini")
advanced_model = ChatOpenAI(model="gpt-5.4")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """根据对话复杂度选择模型"""
    message_count = len(request.state["messages"])
    if message_count > 10:
        model = advanced_model
    else:
        model = basic_model
    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,
    tools=tools,
    middleware=[dynamic_model_selection]
)
```

---

## Tools 工具

### 定义静态工具

```python
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(model, tools=[search, get_weather])
```

### 动态工具过滤（基于状态）

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def state_based_tools(request: ModelRequest, handler: Callable) -> ModelResponse:
    """根据对话状态过滤工具"""
    state = request.state
    is_authenticated = state.get("authenticated", False)
    message_count = len(state["messages"])

    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
    elif message_count < 5:
        tools = [t for t in request.tools if t.name != "advanced_search"]
    else:
        tools = request.tools

    return handler(request.override(tools=tools))

agent = create_agent(
    model="gpt-5.4",
    tools=[public_search, private_search, advanced_search],
    middleware=[state_based_tools]
)
```

### 动态工具过滤（基于 Store）

```python
from dataclasses import dataclass
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

@wrap_model_call
def store_based_tools(request: ModelRequest, handler: Callable) -> ModelResponse:
    """根据用户特性标志过滤工具"""
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    feature_flags = store.get(("features",), user_id)

    if feature_flags:
        enabled_features = feature_flags.value.get("enabled_tools", [])
        tools = [t for t in request.tools if t.name in enabled_features]
        return handler(request.override(tools=tools))
    return handler(request)

agent = create_agent(
    model="gpt-5.4",
    tools=[search_tool, analysis_tool, export_tool],
    middleware=[store_based_tools],
    context_schema=Context,
    store=InMemoryStore()
)
```

### 动态工具过滤（基于 Runtime Context 权限）

```python
from dataclasses import dataclass
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

@dataclass
class Context:
    user_role: str

@wrap_model_call
def context_based_tools(request: ModelRequest, handler: Callable) -> ModelResponse:
    """根据用户角色过滤工具"""
    user_role = request.runtime.context.user_role if request.runtime and request.runtime.context else "viewer"

    if user_role == "admin":
        pass  # 管理员获取所有工具
    elif user_role == "editor":
        tools = [t for t in request.tools if t.name != "delete_data"]
        return handler(request.override(tools=tools))
    else:
        tools = [t for t in request.tools if t.name.startswith("read_")]
        return handler(request.override(tools=tools))
    return handler(request)

agent = create_agent(
    model="gpt-5.4",
    tools=[read_data, write_data, delete_data],
    middleware=[context_based_tools],
    context_schema=Context
)
```

### 运行时注册动态工具

```python
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest

@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 20.0) -> str:
    """Calculate the tip amount for a bill."""
    tip = bill_amount * (tip_percentage / 100)
    return f"Tip: ${tip:.2f}, Total: ${bill_amount + tip:.2f}"

class DynamicToolMiddleware(AgentMiddleware):
    """动态工具中间件"""

    def wrap_model_call(self, request: ModelRequest, handler):
        # 添加动态工具
        updated = request.override(tools=[*request.tools, calculate_tip])
        return handler(updated)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        if request.tool_call["name"] == "calculate_tip":
            return handler(request.override(tool=calculate_tip))
        return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[get_weather],
    middleware=[DynamicToolMiddleware()],
)
```

### 工具错误处理

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """自定义工具错误处理"""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="gpt-5.4",
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)
```

---

## ReAct 循环

Agent 遵循 ReAct（Reasoning + Acting）模式：

```
Prompt: 找出当前最流行的无线耳机并验证库存

推理: "需要使用搜索工具查询流行产品"
行动: 调用 search_products("wireless headphones")

推理: "需要确认销量第一的产品的库存状态"
行动: 调用 check_inventory("WH-1000XM5")

推理: "已获取最流行型号及其库存状态，可以回答用户"
行动: 生成最终答案
```

---

## System Prompt

### 简单字符串

```python
agent = create_agent(
    model,
    tools,
    system_prompt="You are a helpful assistant. Be concise and accurate."
)
```

### 使用 SystemMessage（支持提示缓存）

```python
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage

literary_agent = create_agent(
    model="google_genai:gemini-3.1-pro-preview",
    system_prompt=SystemMessage(content=[
        {"type": "text", "text": "You are an AI assistant tasked with analyzing literary works."},
        {"type": "text", "text": "<the entire contents of 'Pride and Prejudice'>", "cache_control": {"type": "ephemeral"}}
    ])
)
```

### 动态系统提示词

```python
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest

class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据用户角色生成系统提示词"""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."
    return base_prompt

agent = create_agent(
    model="gpt-5.4",
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)
```

---

## 结构化输出

### ToolStrategy（通用方式）

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-5.4-mini",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})
# result["structured_response"] -> ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

### ProviderStrategy（原生方式）

```python
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-5.4",
    response_format=ProviderStrategy(ContactInfo)
)

# 或直接传 schema（LangChain 1.0 默认行为）
agent = create_agent(
    model="gpt-5.4",
    response_format=ContactInfo  # 自动选择可用策略
)
```

---

## 状态与记忆

### 通过中间件定义状态（推荐）

```python
from langchain.agents import AgentState, AgentMiddleware
from typing import Any

class CustomState(AgentState):
    user_preferences: dict

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        ...

agent = create_agent(
    model,
    tools=[tool1, tool2],
    middleware=[CustomMiddleware()]
)
```

### 通过 state_schema 定义状态

```python
from langchain.agents import AgentState

class CustomState(AgentState):
    user_preferences: dict

agent = create_agent(
    model,
    tools=[tool1, tool2],
    state_schema=CustomState
)
```

---

## 流式输出

```python
from langchain.messages import AIMessage, HumanMessage

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Search for AI news and summarize"}]},
    stream_mode="values"
):
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
```

---

## 中间件

### 常用中间件装饰器

- `@before_model` - 模型调用前处理状态
- `@after_model` - 模型调用后处理响应
- `@wrap_model_call` - 包装模型调用
- `@wrap_tool_call` - 包装工具调用
- `@dynamic_prompt` - 动态生成系统提示词

### 中间件常见用途

- 根据状态/上下文修改工具集
- 动态选择模型
- 工具执行错误处理
- 消息修剪/上下文注入
- guardrails/内容过滤
- 自定义日志/监控

---

## Agent 命名

```python
agent = create_agent(
    model,
    tools,
    name="research_assistant"  # 多 Agent 系统中用作子图节点标识
)
```

> 注意：推荐使用 snake_case（如 `research_assistant`），某些模型提供者可能拒绝包含空格或特殊字符的名称。

---

## 完整示例

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

# 1. 定义工具
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# 2. 错误处理中间件
@wrap_tool_call
def handle_errors(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: {str(e)}",
            tool_call_id=request.tool_call["id"]
        )

# 3. 创建 Agent
agent = create_agent(
    model="openai:gpt-5.4",
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
    middleware=[handle_errors]
)

# 4. 调用 Agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]
})
print(result["messages"][-1].content_blocks)
```

---

## 参考链接

- 官方文档：https://docs.langchain.com/oss/python/langchain/agents
- LangChain vs LangGraph vs Deep Agents：
  - Deep Agents：开箱即用方案
  - LangChain：直接使用预建 Agent
  - LangGraph：高级编排框架
- LangSmith：追踪、调试、评估 Agent
