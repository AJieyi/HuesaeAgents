# 二次元多智能体系统 - 框架规划

## Context
开发一个以二次元为主题的多智能体陪伴系统。
- 技术栈：LangChain + LangGraph
- 架构：1个主智能体 + 1个子智能体（最多3个）
- 简化设计，先规划框架，后续按需扩展

---

## 一、项目模块规划

```
anicloud/
├── backend/                    # 后端服务
│   ├── app/                   # FastAPI应用
│   ├── packages/
│   │   └── harness/
│   │       └── aniverse/      # 核心智能体框架
│   └── config/                # 配置文件
│
├── frontend/                  # 前端（后续按需开发）
│
├── database/                  # 数据库（可选，初始可用文件存储）
│
└── data/                      # 数据存储（记忆、角色配置）
```

---

## 二、核心模块（Backend）

### 1. 智能体模块（agents）
```
packages/harness/aniverse/agents/
├── lead_agent/                # 主智能体
│   ├── factory.py            # Agent工厂（create_aniverse_agent）
│   ├── graph.py              # LangGraph工作流定义
│   └── prompts.py            # 系统提示词模板
│
├── subagent/                  # 子智能体（1个通用，可扩展到3个）
│   ├── executor.py            # 子智能体执行器
│   └── general_purpose.py    # 通用子智能体定义
│
├── state.py                   # 状态定义（ThreadState）
├── character/                 # 角色管理
│   ├── manager.py             # 角色管理器
│   └── loader.py              # 角色配置加载
│
└── middleware/                # 中间件（精简设计）
    ├── base.py               # 中间件基类
    ├── emotion.py            # 情绪感知
    ├── guardrail.py          # 安全护栏
    └── character.py          # 角色切换
```

### 2. 工具模块（tools）
```
tools/
├── registry.py               # 工具注册表
├── builtins/                 # 内置工具
│   ├── ask.py               # 询问用户
│   ├── task.py             # 任务委托
│   └── ...
└── external/                # 外部工具（搜索、语音等）
```

### 3. 记忆模块（memory）
```
memory/
├── storage.py               # 记忆存储（文件或数据库）
├── manager.py               # 记忆管理器
└── types.py                 # 记忆数据结构
```

### 4. API模块（app/api）
```
app/api/
├── chat.py                  # 对话API
├── character.py             # 角色API
└── memory.py                # 记忆API
```

---

## 三、数据存储

### 初始方案：文件存储
- 角色配置：JSON文件
- 用户记忆：JSON文件
- 后续可迁移到SQLite/PostgreSQL

---

## 四、部署架构

```
用户请求 → FastAPI(Gateway) → 主智能体 → 子智能体(1-3个)
                                      ↓
                                   工具/记忆/LLM
```

---

## 五、后续扩展方向

| 模块 | 扩展内容 |
|------|----------|
| 前端 | Web界面/APP（按需开发） |
| 数据库 | SQLite → PostgreSQL |
| 子智能体 | 1个 → 3个（图片、搜索、记忆专项） |
| 工具 | 搜索、语音、图片生成等 |

---

## 六、验证计划（框架完成后）

```bash
cd backend
uvicorn app.main:app --reload

# 测试对话
curl -X POST http://localhost:8000/api/chat \
  -d '{"message": "你好", "character_id": "gentle_sister"}'
```