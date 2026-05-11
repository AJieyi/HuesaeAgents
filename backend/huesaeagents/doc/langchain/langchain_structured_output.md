# LangChain Structured Output 关键代码写法总结

## 目录

- [概述](#概述)
- [响应格式](#响应格式)
- [Provider Strategy](#provider-strategy)
- [Tool Strategy](#tool-strategy)
- [自定义工具消息内容](#自定义工具消息内容)
- [错误处理](#错误处理)
- [完整示例](#完整示例)

---

## 概述

结构化输出允许 Agent 以特定格式返回数据（JSON、Pydantic 模型、dataclass），无需解析自然语言响应。

使用 `create_agent` 的 `response_format` 参数控制结构化输出。

---

## 响应格式

```python
def create_agent(
    ...,
    response_format: Union[
        ToolStrategy[StructuredResponseT],
        ProviderStrategy[StructuredResponseT],
        type[StructuredResponseT],
        None,
    ]
)
```

| 格式 | 说明 |
|------|------|
| `ToolStrategy[Schema]` | 使用工具调用生成结构化输出 |
| `ProviderStrategy[Schema]` | 使用提供者原生结构化输出 |
| `type[Schema]` | 自动选择最佳策略（Provider > Tool） |
| `None` | 不请求结构化输出 |

**自动选择逻辑：**
- 如果模型支持原生结构化输出（OpenAI、Anthropic、Gemini、xAI），自动使用 ProviderStrategy
- 否则使用 ToolStrategy
- 需要 langchain>=1.1 且模型 profile 数据可用

---

## Provider Strategy

提供者原生结构化输出，最可靠的方法。

### ProviderStrategy 定义

```python
class ProviderStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    strict: bool | None = None  # 需要 langchain>=1.2
```

### 支持的 Schema 类型

- **Pydantic models** - 返回验证后的 Pydantic 实例
- **Dataclasses** - 返回字典
- **TypedDict** - 返回字典
- **JSON Schema** - 返回字典

### Pydantic Model 示例

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent

class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

agent = create_agent(
    model="gpt-5.4",
    response_format=ContactInfo  # 自动选择 ProviderStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

print(result["structured_response"])
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

---

## Tool Strategy

使用工具调用生成结构化输出，适用于不支持原生结构化输出的模型。

### ToolStrategy 定义

```python
class ToolStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    tool_message_content: str | None = None
    handle_errors: Union[
        bool,
        str,
        type[Exception],
        tuple[type[Exception], ...],
        Callable[[Exception], str],
    ] = True
```

### Pydantic Model 示例

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ProductReview(BaseModel):
    """Analysis of a product review."""
    rating: int | None = Field(description="The rating of the product", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
    key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")

agent = create_agent(
    model="gpt-5.4",
    tools=tools,
    response_format=ToolStrategy(ProductReview)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})

print(result["structured_response"])
# ProductReview(rating=5, sentiment='positive', key_points=['fast shipping', 'expensive'])
```

### Union Types（多 Schema 选项）

```python
from pydantic import BaseModel, Field
from typing import Union

class ContactInfo(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")

class EventDetails(BaseModel):
    event_name: str = Field(description="Name of the event")
    date: str = Field(description="Event date")

agent = create_agent(
    model="gpt-5.4",
    tools=[],
    response_format=ToolStrategy(Union[ContactInfo, EventDetails])
)
```

---

## 自定义工具消息内容

使用 `tool_message_content` 自定义结构化输出生成时出现在对话历史中的消息：

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class MeetingAction(BaseModel):
    """Action items extracted from a meeting transcript."""
    task: str = Field(description="The specific task to be completed")
    assignee: str = Field(description="Person responsible for the task")
    priority: Literal["low", "medium", "high"] = Field(description="Priority level")

agent = create_agent(
    model="gpt-5.4",
    tools=[],
    response_format=ToolStrategy(
        schema=MeetingAction,
        tool_message_content="Action item captured and added to meeting notes!"
    )
)
```

---

## 错误处理

模型生成结构化输出时可能犯错，LangChain 提供智能重试机制。

### 错误类型

1. **MultipleStructuredOutputsError** - 模型错误地调用了多个结构化输出工具
2. **StructuredOutputValidationError** - 结构化输出不符合预期 schema

### 多结构化输出错误

当模型错误地返回多个结构化响应时，Agent 会提供错误反馈并提示重试：

```python
# 模型先错误地返回了两个结构化输出
# 然后被提示修正，只返回一个
```

### Schema 验证错误

当输出不符合 schema 时（如 rating 超出范围），Agent 会提供具体错误信息并重试：

```python
# rating: 10 超出 1-5 范围
# Agent 收到错误: "1 validation error for ProductRating.rating Input should be less than or equal to 5"
# 然后修正为 rating: 5
```

### 错误处理策略

| handle_errors 值 | 行为 |
|------------------|------|
| `True` | 捕获所有错误，使用默认错误模板 |
| `str` | 捕获所有错误，使用自定义消息 |
| `type[Exception]` | 只捕获指定异常类型 |
| `tuple[type[Exception], ...]` | 捕获多种指定异常类型 |
| `Callable[[Exception], str]` | 自定义错误处理函数 |
| `False` | 不捕获，让异常冒泡 |

### 自定义错误消息

```python
ToolStrategy(
    schema=ProductRating,
    handle_errors="Please provide a valid rating between 1-5 and include a comment."
)
```

### 只处理特定异常

```python
# 只在 ValueError 时重试，其他异常抛出
ToolStrategy(
    schema=ProductRating,
    handle_errors=ValueError
)

# 在 ValueError 和 TypeError 时重试
ToolStrategy(
    schema=ProductRating,
    handle_errors=(ValueError, TypeError)
)
```

### 自定义错误处理函数

```python
from langchain.agents.structured_output import (
    StructuredOutputValidationError,
    MultipleStructuredOutputsError
)

def custom_error_handler(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "There was an issue with the format. Try again."
    elif isinstance(error, MultipleStructuredOutputsError):
        return "Multiple structured outputs were returned. Pick the most relevant one."
    else:
        return f"Error: {str(error)}"

agent = create_agent(
    model="gpt-5.4",
    tools=[],
    response_format=ToolStrategy(
        schema=Union[ContactInfo, EventDetails],
        handle_errors=custom_error_handler
    )
)
```

### 禁用错误处理

```python
ToolStrategy(
    schema=ProductRating,
    handle_errors=False  # 所有错误都会抛出
)
```

---

## 完整示例

### Provider Strategy（自动选择）

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent

class ContactInfo(BaseModel):
    """Contact information."""
    name: str = Field(description="The person's name")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")

agent = create_agent(
    model="gpt-5.4",
    response_format=ContactInfo
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "John Doe, john@email.com, 555-1234"}]
})

print(result["structured_response"])
```

### Tool Strategy（带错误处理）

```python
from pydantic import BaseModel, Field
from typing import Literal, Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, StructuredOutputValidationError

class ProductRating(BaseModel):
    rating: int = Field(description="Rating from 1-5", ge=1, le=5)
    comment: str = Field(description="Review comment")

class UserFeedback(BaseModel):
    feedback: str = Field(description="User feedback text")
    category: Literal["complaint", "praise", "question"]

def custom_handler(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "Format error. Please check constraints and retry."
    return str(error)

agent = create_agent(
    model="gpt-5.4",
    tools=some_tools,
    response_format=ToolStrategy(
        schema=Union[ProductRating, UserFeedback],
        tool_message_content="Feedback captured!",
        handle_errors=custom_handler
    )
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "The product is okay, maybe 3 stars. When will the new version arrive?"}]
})
```

### 带验证约束的 Pydantic Model

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class Order(BaseModel):
    order_id: str = Field(description="Order identifier", pattern=r"^ORD-\d+$")
    items: list[str] = Field(description="List of item names", min_length=1)
    total: float = Field(description="Total amount", gt=0)
    status: Literal["pending", "shipped", "delivered", "cancelled"]
    priority: Literal["low", "medium", "high"] = "medium"

    @field_validator("order_id")
    @classmethod
    def validate_order_id(cls, v: str) -> str:
        if not v.startswith("ORD-"):
            raise ValueError("Order ID must start with 'ORD-'")
        return v

agent = create_agent(
    model="gpt-5.4",
    response_format=Order
)
```

---

## 关键要点

1. **response_format** 参数控制结构化输出方式
2. **自动选择**：传递 schema 类型时自动选择 Provider 或 Tool 策略
3. **Pydantic** 提供最丰富的功能：字段验证、描述、嵌套结构
4. **handle_errors** 控制验证失败时的行为
5. **tool_message_content** 自定义对话历史中的消息
6. **structured_response** 位于 agent 最终状态的 `structured_response` 键中

---

## 参考链接

- 官方文档：https://docs.langchain.com/oss/python/langchain/structured-output
- Models - Structured output：直接对模型使用结构化输出
