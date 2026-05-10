# Huesae Models系统搭建计划

## 一、目录结构

```
backend/huesaeagents/huesae/models/
├── __init__.py
├── factory.py                  # 【核心】模型工厂 create_chat_model()
├── providers/                  # 模型提供商
│   ├── __init__.py
│   ├── base.py               # Provider基类
│   └── deepseek.py           # DeepSeek Provider
└── test/
    └── test_models.py        # 测试文件
```

---

## 二、核心文件说明

### 2.1 模型工厂 (factory.py)
```python
def create_chat_model(
    provider: str = "deepseek",
    model: str = "deepseek-v4-flash",
    **kwargs,
) -> BaseChatModel
```
从环境变量 `DEEPSEEK_API_KEY` 读取API Key。

### 2.2 Provider基类 (providers/base.py)
```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def get_model(self) -> BaseChatModel
```

### 2.3 DeepSeek Provider (providers/deepseek.py)
```python
class DeepSeekProvider(BaseLLMProvider):
    def get_model(self) -> ChatDeepSeek
    async def ainvoke(messages, **kwargs) -> BaseMessage
    def invoke(input, **kwargs) -> AIMessage  # 同步方法
```

---

## 三、已实现功能

| 文件 | 状态 | 说明 |
|------|------|------|
| providers/base.py | ✅ | Provider抽象基类 |
| providers/deepseek.py | ✅ | DeepSeek Provider（含同步/异步调用） |
| factory.py | ✅ | create_chat_model 工厂函数 |
| test/test_models.py | ✅ | 测试文件 |

---

## 四、使用示例

```python
from huesae.models.factory import create_chat_model

# 创建模型
model = create_chat_model(provider="deepseek")

# 同步调用
result = model.invoke("你好")

# 异步调用
result = await model.ainvoke([HumanMessage(content="你好")])
```

---

## 五、扩展多Provider

如需添加其他模型（如Qwen），在 `providers/` 下新建文件：

```
providers/
├── base.py
├── deepseek.py
└── qwen.py      # 新增
```

工厂函数根据provider名称选择对应的Provider。
