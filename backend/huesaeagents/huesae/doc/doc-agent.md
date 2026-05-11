# HuesaeAgents - 二次元多智能体陪伴系统架构规划

## Context

用户希望重新规划二次元多智能体陪伴系统：
- 已删除旧的 LangGraph 工作流文件
- 参考 TradingAgents.md 架构风格新建 `graph/` 文件夹
- 使用 LangChain + LangGraph 技术栈
- 虚拟环境：`conda activate HuesaeAgents`

**当前项目状态**：
- 已有 `tools/` - 图片生成工具（即梦/豆包）
- 已有 `models/` - 模型工厂（DeepSeek）
- 已有 `subagents/` - 空目录，待填充
- 待新建 `graph/` - 工作流编排

---

## 一、系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HuesaeAgents                                 │
├──────────────────┬──────────────────┬───────────────────────────────┤
│   CLI 入口        │   API 入口        │        前端（未来）            │
│   (main.py)      │   (app/)         │        (frontend/)             │
├──────────────────┴──────────────────┴───────────────────────────────┤
│                     核心框架 (huesae/)                               │
│  ┌─────────────┬──────────────┬────────────┬─────────────────┐      │
│  │   Agents    │    Graph      │   Models   │     Tools       │      │
│  │  智能体      │   工作流编排   │   模型工厂   │     工具集      │      │
│  └─────────────┴──────────────┴────────────┴─────────────────┘      │
│  ┌─────────────┬──────────────┬────────────┬─────────────────┐      │
│  │  Character  │    Memory     │    Voice   │     Search     │      │
│  │   角色管理   │    记忆系统    │    语音合成  │     搜索工具    │      │
│  └─────────────┴──────────────┴────────────┴─────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、目录结构

```
backend/huesaeagents/huesae/
├── __init__.py
│
├── graph/                          # 【新建】LangGraph 工作流
│   ├── __init__.py
│   ├── huesae_graph.py            # 主工作流（核心）
│   ├── conditional_logic.py        # 条件路由逻辑
│   ├── state.py                   # 图状态定义
│   └── checkpointer/              # 持久化检查点
│       └── memory.py              # 内存检查点
│
├── agents/                         # 【已有】智能体
│   ├── __init__.py
│   ├── lead_agent/                # 主智能体（对话中枢）
│   │   ├── __init__.py
│   │   ├── lead.py                # 主智能体实现
│   │   └── prompts.py             # 提示词模板
│   │
│   ├── subagents/                 # 【重建】子智能体
│   │   ├── __init__.py
│   │   ├── image_agent.py         # 生图智能体
│   │   ├── voice_agent.py        # 语音智能体
│   │   ├── memory_agent.py       # 记忆智能体
│   │   ├── search_agent.py       # 搜索智能体
│   │   ├── remind_agent.py       # 提醒智能体
│   │   └── safe_agent.py         # 安全智能体
│   │
│   └── character/                 # 角色管理
│       ├── __init__.py
│       ├── manager.py             # 角色管理器
│       ├── loader.py              # 角色配置加载
│       └── configs/               # 角色配置
│           ├── gentle_sister.py
│           ├── tsundere.py
│           └── furry_fox.py
│
├── models/                         # 【已有】模型模块
│   ├── __init__.py
│   ├── models_factory.py
│   └── providers/
│       └── deepseek.py
│
├── memory/                         # 记忆系统
│   ├── __init__.py
│   ├── short_term.py              # 短期记忆
│   ├── long_term.py               # 长期记忆
│   └── memory_manager.py          # 记忆管理器
│
├── voice/                          # 语音模块
│   ├── __init__.py
│   └── minimax.py                 # MiniMax 语音合成
│
└── tools/                          # 【已有】工具模块
    ├── __init__.py
    ├── image.py                   # 图片生成统一接口
    ├── jimeng/                    # 即梦AI
    │   ├── client.py
    │   └── sign.py
    └── doubao/                    # 豆包Seedream
        └── client.py
```

---

## 三、Graph 模块设计

### 3.1 主工作流 (`huesae_graph.py`)

```python
# 工作流结构
"""
输入 → 意图分类 → [条件路由]
                  ├─ 对话类 → 情绪处理 → 主智能体 → 输出 → END
                  ├─ 生图类 → 生图智能体 → END
                  ├─ 语音类 → 语音智能体 → END
                  ├─ 记忆类 → 记忆智能体 → END
                  ├─ 搜索类 → 搜索智能体 → END
                  └─ 安全类 → 安全智能体 → END
"""
```

### 3.2 状态定义 (`state.py`)

```python
class HuesaeState(TypedDict):
    """主图状态"""
    messages: Annotated[list, add_messages]     # 对话历史
    intent: str | None                          # 意图分类结果
    character_id: str                           # 当前角色ID
    emotion_state: str                          # 情绪状态
    emotion_score: float                        # 情绪强度 0-1

    # 子智能体结果
    subagent_result: dict | None

    # 用户信息
    user_id: str | None
    thread_id: str | None

    # 安全标记
    safety_flag: bool
    high_risk_flag: bool
```

### 3.3 条件路由 (`conditional_logic.py`)

```python
def classify_intent(state: HuesaeState) -> str:
    """意图分类"""
    # 对话/生图/语音/记忆/搜索/提醒/安全

def route_by_intent(state: HuesaeState) -> str:
    """根据意图路由到对应节点"""
```

---

## 四、子智能体职责

| 智能体 | 优先级 | 职责 | 依赖 |
|-------|--------|------|------|
| **Image Agent** | P0 | 自然语言→Danbooru标签→ComfyUI生图、Pixiv爬取、反推标签 | ComfyUI API |
| **Voice Agent** | P1 | MiniMax TTS（3种风格：温柔/活泼/沉稳） | MiniMax API |
| **Safe Agent** | P0 | 安全边界、高风险检测、关怀话术 | 关键词检测 |
| **Memory Agent** | P1 | 日记存储、时间线、目标记忆 | SQLite/文件 |
| **Search Agent** | P2 | 网页搜索、PPT生成 | 搜索API |
| **Remind Agent** | P2 | 目标提醒、番茄钟 | APScheduler |

---

## 五、实现顺序

### Phase 1: Graph 核心（P0）

| 顺序 | 文件 | 说明 |
|------|------|------|
| 1 | `graph/state.py` | 图状态定义 |
| 2 | `graph/conditional_logic.py` | 意图分类与路由 |
| 3 | `graph/huesae_graph.py` | 主工作流 |
| 4 | `graph/checkpointer/memory.py` | 持久化检查点 |

### Phase 2: 主智能体（P0）

| 顺序 | 文件 | 说明 |
|------|------|------|
| 5 | `agents/lead_agent/lead.py` | 主智能体实现 |
| 6 | `agents/lead_agent/prompts.py` | 提示词模板 |
| 7 | `agents/character/manager.py` | 角色管理器 |
| 8 | `agents/character/configs/*.py` | 角色配置 |

### Phase 3: 核心子智能体（P0-P1）

| 顺序 | 文件 | 说明 |
|------|------|------|
| 9 | `agents/subagents/safe_agent.py` | 安全智能体 |
| 10 | `agents/subagents/image_agent.py` | 生图智能体 |
| 11 | `voice/minimax.py` + `agents/subagents/voice_agent.py` | 语音智能体 |

### Phase 4: 扩展功能（P1-P2）

| 顺序 | 文件 | 说明 |
|------|------|------|
| 12 | `memory/memory_manager.py` | 记忆管理器 |
| 13 | `agents/subagents/memory_agent.py` | 记忆智能体 |
| 14 | `agents/subagents/search_agent.py` | 搜索智能体 |
| 15 | `agents/subagents/remind_agent.py` | 提醒智能体 |

---

## 六、关键文件清单

| 文件路径 | 优先级 | 说明 |
|---------|--------|------|
| `graph/state.py` | P0 | 图状态定义 |
| `graph/huesae_graph.py` | P0 | 主工作流 |
| `graph/conditional_logic.py` | P0 | 条件路由 |
| `agents/lead_agent/lead.py` | P0 | 主智能体 |
| `agents/character/manager.py` | P0 | 角色管理 |
| `agents/subagents/safe_agent.py` | P0 | 安全智能体 |
| `agents/subagents/image_agent.py` | P0 | 生图智能体 |
| `voice/minimax.py` | P1 | MiniMax语音 |
| `agents/subagents/voice_agent.py` | P1 | 语音智能体 |
| `memory/memory_manager.py` | P1 | 记忆管理 |

---

## 七、环境配置

```bash
# 虚拟环境
conda activate HuesaeAgents

# 现有依赖
langchain==1.2.17
langgraph==1.1.10
langchain-deepseek

# 新增依赖
minimax-python      # MiniMax 语音合成
httpx               # HTTP 客户端
apscheduler         # 定时任务
pillow              # 图片处理
```

---

## 八、验证方案

```bash
# 1. 激活环境
conda activate HuesaeAgents

# 2. 运行测试
cd backend/huesaeagents
python -m huesae.agents.test.test_agents

# 3. 手动测试流程
# - 对话测试：发送消息，验证情绪检测和角色回复
# - 生图测试：发送"画一个银发红瞳少女"，验证标签生成和图片
# - 语音测试：发送"用语音回复"，验证音频生成
```

---

## 九、技术规范

1. **代码风格**：参考 TradingAgents.md 模块划分
2. **异步支持**：IO密集型操作使用 `async/await`
3. **状态持久化**：LangGraph checkpointer
4. **类型提示**：所有函数添加类型注解
5. **安全第一**：Safe Agent 作为始终运行的中间件
