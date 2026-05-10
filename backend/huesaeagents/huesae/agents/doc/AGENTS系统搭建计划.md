# Agents系统搭建计划（简化版）

## 一、目录结构

### 初期只需3个核心文件
```
backend/huesaeagents/huesae/agents/
├── __init__.py
├── state.py       # 【核心】ThreadState状态定义
├── factory.py     # 【核心】Agent工厂 create_huesae_agent()
└── graph.py      # 【核心】LangGraph工作流（5节点）
```

### 后续按需添加
```
├── prompts.py     # 系统提示词模板（后续）
├── character/    # 角色管理（后续）
├── middleware/   # 中间件（后续）
└── subagent/    # 子Agent（后续）
```

---

## 二、核心文件说明

### 2.1 状态定义 (state.py)
```python
class ThreadState(TypedDict):
    messages: list              # 对话消息列表
    character_id: str | None   # 当前角色ID
    user_id: str | None       # 用户ID
    thread_id: str | None     # 会话线程ID
```

### 2.2 Agent工厂 (factory.py)
```python
def create_huesae_agent(model, **kwargs) -> CompiledStateGraph
```
- 创建LangGraph StateGraph
- 返回可执行Agent

### 2.3 LangGraph工作流 (graph.py) - 简化5节点
```
input → emotion_detect → reasoner → output
```
- `input`: 解析用户输入
- `emotion_detect`: 情绪检测（可选，初期可省略）
- `reasoner`: LLM推理
- `output`: 输出格式化

---

## 三、实现顺序

| 顺序 | 文件 | 说明 |
|------|------|------|
| 1 | state.py | 状态定义，最基础 |
| 2 | graph.py | 工作流定义 |
| 3 | factory.py | Agent工厂 |

---

## 四、验证方案

```python
from huesae.models.factory import create_chat_model
from huesaeagents.huesae.agents.factory import create_huesae_agent

# 创建Agent
model = create_chat_model("deepseek")
agent = create_huesae_agent(model)

# 对话测试
result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
print(result)
```

---

## 五、后续扩展方向

| 扩展 | 说明 |
|------|------|
| prompts.py | 添加系统提示词模板 |
| middleware/ | 添加emotion、guardrail、character中间件 |
| character/ | 角色管理系统 |
| subagent/ | 子Agent执行器（最多3个） |

---

## 六、与deerflow2对比

| 功能 | deerflow2 | huesae（简化） |
|------|-----------|---------------|
| 核心文件 | 大量 | 仅3个 |
| 工作流节点 | 14+ | 5个 |
| 中间件 | 16个 | 0个（初期） |
| 角色管理 | 无 | 后续添加 |
