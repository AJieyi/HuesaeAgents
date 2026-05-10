# DeerFlow 系统架构设计文档

> 本文档详细分析 `backend/packages/harness/deerflow/` 目录下的系统架构、功能模块划分、服务集成方式以及模块间依赖关系。

---

## 一、项目概述

DeerFlow 是一个基于 LangGraph 的 AI Super Agent 系统，采用模块化架构设计。核心定位是一个**可嵌入的 Python 客户端和 LangGraph Agent 运行时**，支持沙箱执行、持久化记忆、子 Agent 委托和可扩展工具集成，所有操作都在**线程级别隔离的环境**中运行。

### 1.1 核心特性

- **多模型支持**: Claude、OpenAI、vLLM、MiniMax、DeepSeek 等
- **工具扩展**: MCP 协议、社区工具、内置工具、自定义工具
- **沙箱隔离**: 本地文件系统或 Docker 容器隔离执行环境
- **记忆系统**: 持久化上下文记忆，支持事实提取和去重
- **子 Agent**: 支持任务委托和并发执行控制
- **技能系统**: 可插拔的技能模块，支持热更新

---

## 二、顶层目录结构与功能模块划分

```
backend/packages/harness/deerflow/
├── agents/                    # ★ 核心 Agent 系统
│   ├── lead_agent/           # 主 Agent（工厂 + 系统提示词）
│   ├── middlewares/          # 14+ 个中间件组件
│   ├── memory/               # 记忆提取、队列、提示词
│   ├── checkpointer/         # 状态持久化
│   ├── factory.py            # SDK 级 Agent 工厂
│   ├── features.py           # 运行时特性标志
│   └── thread_state.py       # ThreadState 状态模式
├── client.py                 # ★ DeerFlowClient 嵌入式客户端入口
├── config/                    # ★ 配置系统
│   ├── app_config.py        # 主配置（config.yaml）
│   ├── extensions_config.py # MCP 和 Skills 配置
│   ├── memory_config.py     # 记忆系统配置
│   ├── model_config.py      # 模型配置
│   ├── sandbox_config.py    # 沙箱配置
│   └── paths.py             # 路径解析
├── models/                   # ★ 模型工厂
│   ├── factory.py           # create_chat_model()
│   ├── claude_provider.py   # Claude provider
│   ├── vllm_provider.py     # vLLM provider
│   └── openai_codex_provider.py
├── sandbox/                  # ★ 沙箱执行系统
│   ├── sandbox.py           # 抽象 Sandbox 接口
│   ├── sandbox_provider.py  # SandboxProvider 提供者模式
│   ├── local/               # 本地文件系统提供者
│   ├── middleware.py         # 沙箱生命周期管理
│   └── tools.py             # bash/ls/read/write/str_replace 工具
├── tools/                    # ★ 工具系统
│   ├── tools.py             # get_available_tools() 入口
│   ├── builtins/            # 内置工具
│   │   ├── present_file_tool.py
│   │   ├── ask_clarification_tool.py
│   │   ├── view_image_tool.py
│   │   ├── task_tool.py     # 子 Agent 委托
│   │   ├── tool_search.py   # 工具搜索
│   │   └── invoke_acp_agent_tool.py
│   └── skill_manage_tool.py # 技能管理工具
├── subagents/               # ★ 子 Agent 委托系统
│   ├── executor.py          # 后台执行引擎
│   ├── registry.py          # Agent 注册表
│   └── builtins/            # 内置子 Agent
├── community/               # ★ 社区工具集成
│   ├── aio_sandbox/         # Docker 隔离沙箱
│   ├── tavily/              # 网页搜索
│   ├── jina_ai/            # 内容提取
│   ├── firecrawl/          # 网页爬取
│   ├── ddg_search/         # DuckDuckGo 搜索
│   ├── image_search/        # 图片搜索
│   └── infoquest/
├── mcp/                      # ★ MCP (Model Context Protocol) 集成
│   ├── client.py            # MCP 客户端
│   ├── cache.py             # 工具缓存
│   ├── oauth.py             # OAuth 支持
│   └── tools.py             # MCP 工具
├── runtime/                  # ★ Agent 运行时（Gateway 模式）
│   ├── runs/                # 运行管理
│   │   ├── manager.py       # RunManager 运行记录
│   │   ├── worker.py        # run_agent() 后台执行
│   │   └── schemas.py       # 运行状态定义
│   ├── stream_bridge/       # SSE 流式桥接
│   └── store/               # 状态存储
├── skills/                  # ★ 技能系统
│   ├── loader.py           # 技能加载
│   ├── manager.py          # 技能管理
│   └── installer.py        # 技能安装
├── reflection/              # 动态模块加载
├── guardrails/              # 安全护栏
├── tracing/                 # 追踪
└── uploads/                 # 文件上传管理
```

---

## 三、核心模块详解

### 3.1 Agent 系统 (`agents/`)

Agent 系统是 DeerFlow 的核心，负责管理对话流程、消息处理和工具调用。

#### 3.1.1 入口点

| 文件 | 函数/类 | 用途 |
|------|---------|------|
| `factory.py` | `create_deerflow_agent()` | SDK 级工厂，接受纯 Python 参数 |
| `lead_agent/agent.py` | `make_lead_agent()` | 应用级工厂，读取配置文件 |

#### 3.1.2 ThreadState 状态模式

位于 `agents/thread_state.py`，扩展 `AgentState`：

```python
@dataclass
class ThreadState(AgentState):
    messages: list[BaseMessage]              # 对话消息
    sandbox: dict[str, Any]                  # 沙箱信息
    thread_data: dict[str, Any]               # 线程级数据
    title: str | None                        # 自动生成的标题
    artifacts: list[dict]                     # 输出产物
    todos: list[dict]                        # 任务列表
    uploaded_files: list[str]                 # 上传文件
    viewed_images: list[str]                  # 查看的图片
```

#### 3.1.3 中间件链 (14+ 个)

按执行顺序排列：

| 序号 | 中间件 | 文件 | 职责 |
|------|--------|------|------|
| 1 | `ThreadDataMiddleware` | `thread_data_middleware.py` | 创建线程目录结构 |
| 2 | `UploadsMiddleware` | `uploads_middleware.py` | 注入上传文件到上下文 |
| 3 | `SandboxMiddleware` | `sandbox/middleware.py` | 获取和管理沙箱 |
| 4 | `DanglingToolCallMiddleware` | `dangling_tool_call_middleware.py` | 修补缺失的 ToolMessages |
| 5 | `LLMErrorHandlingMiddleware` | `llm_error_handling_middleware.py` | LLM 错误规范化 |
| 6 | `GuardrailMiddleware` | `guardrails/middleware.py` | 工具调用授权 |
| 7 | `SandboxAuditMiddleware` | `sandbox_audit_middleware.py` | 安全审计 |
| 8 | `ToolErrorHandlingMiddleware` | `tool_error_handling_middleware.py` | 工具异常转换 |
| 9 | `SummarizationMiddleware` | `summarization_middleware.py` | 上下文摘要 |
| 10 | `TodoListMiddleware` | `todo_middleware.py` | 任务追踪 |
| 11 | `TokenUsageMiddleware` | `token_usage_middleware.py` | Token 统计 |
| 12 | `TitleMiddleware` | `title_middleware.py` | 自动生成标题 |
| 13 | `MemoryMiddleware` | `memory_middleware.py` | 记忆更新队列 |
| 14 | `ViewImageMiddleware` | `view_image_middleware.py` | 图片注入 |
| 15 | `DeferredToolFilterMiddleware` | `deferred_tool_filter_middleware.py` | 延迟工具过滤 |
| 16 | `SubagentLimitMiddleware` | `subagent_limit_middleware.py` | 并发限制 |
| 17 | `LoopDetectionMiddleware` | `loop_detection_middleware.py` | 循环检测 |
| 18 | `ClarificationMiddleware` | `clarification_middleware.py` | 澄清请求拦截（最后） |

**中间件组装位置**: `agents/middlewares/tool_error_handling_middleware.py` (`build_lead_runtime_middlewares`) 和 `agents/lead_agent/agent.py` (`_build_middlewares`)

#### 3.1.4 记忆系统 (`agents/memory/`)

| 文件 | 职责 |
|------|------|
| `updater.py` | LLM 记忆更新、事实提取、去重、原子文件写入 |
| `queue.py` | 防抖更新队列、线程级去重 |
| `prompt.py` | 记忆更新提示词模板 |

**数据存储结构** (`backend/.deer-flow/memory.json`):

```json
{
  "userContext": {
    "workContext": "string",
    "personalContext": "string",
    "topOfMind": "string"
  },
  "history": {
    "recentMonths": "string",
    "earlierContext": "string",
    "longTermBackground": "string"
  },
  "facts": [
    {
      "id": "uuid",
      "content": "string",
      "category": "preference|knowledge|context|behavior|goal",
      "confidence": 0.0-1.0,
      "createdAt": "ISO timestamp",
      "source": "string"
    }
  ]
}
```

**更新流程**:
```
MemoryMiddleware → 过滤消息 → Queue.debounce(30s) → LLM 提取事实 → 原子写入 memory.json
```

---

### 3.2 配置系统 (`config/`)

#### 3.2.1 配置文件

| 文件 | 配置内容 |
|------|----------|
| `config.yaml` | 主配置：模型、工具、沙箱、skills、记忆、摘要等 |
| `extensions_config.json` | MCP 服务器和 Skills 启用状态 |

#### 3.2.2 配置类

| 类 | 文件 | 职责 |
|----|------|------|
| `AppConfig` | `app_config.py` | 主配置，统一管理所有配置段 |
| `ExtensionsConfig` | `extensions_config.py` | MCP 服务器和 Skills 配置 |
| `MemoryConfig` | `memory_config.py` | 记忆系统配置 |
| `ModelConfig` | `model_config.py` | 单个模型配置 |
| `SandboxConfig` | `sandbox_config.py` | 沙箱配置 |
| `Paths` | `paths.py` | 虚拟路径解析 |

#### 3.2.3 配置解析链

```
config.yaml ──▶ AppConfig.from_file() ──▶ get_app_config()
                                         │
                                         ▼
                           ExtensionsConfig.from_file() ──▶ get_extensions_config()
```

#### 3.2.4 配置热重载

- `AppConfig.from_file()` 缓存配置，通过 mtime 检测文件变更自动重载
- `get_extensions_config()` 支持运行时更新 MCP 和 Skills 配置

---

### 3.3 模型工厂 (`models/`)

#### 3.3.1 入口

**文件**: `models/factory.py`
**函数**: `create_chat_model(name, thinking_enabled)`

#### 3.3.2 支持的模型类型

| Provider | 文件 | 特性 |
|----------|------|------|
| Claude | `claude_provider.py` | thinking、vision |
| OpenAI / Codex | `openai_codex_provider.py` | thinking、vision |
| vLLM | `vllm_provider.py` | Qwen 推理模型、chat_template_kwargs |
| MiniMax | `patched_minimax.py` | 厂商定制 |
| DeepSeek | `patched_deepseek.py` | 厂商定制 |
| MindIE | `mindie_provider.py` | 厂商定制 |

#### 3.3.3 模型特性标志

```python
@dataclass
class ModelConfig:
    name: str
    model: str                    # 模型标识
    supports_thinking: bool       # 支持 extended thinking
    supports_vision: bool         # 支持视觉理解
    use_responses_api: bool        # 使用 OpenAI Responses API
    # ... 其他 provider 特定字段
```

---

### 3.4 工具系统 (`tools/`)

#### 3.4.1 入口

**文件**: `tools/tools.py`
**函数**: `get_available_tools(groups, include_mcp, model_name, subagent_enabled)`

#### 3.4.2 工具来源

| 来源 | 加载方式 | 优先级 |
|------|----------|--------|
| Config 定义工具 | `resolve_variable(cfg.use, BaseTool)` | 最高 |
| 内置工具 | `BUILTIN_TOOLS` 列表 | 其次 |
| MCP 工具 | `get_cached_mcp_tools()` | 其次 |
| ACP 工具 | `build_invoke_acp_agent_tool()` | 最低 |

#### 3.4.3 内置工具 (`tools/builtins/`)

| 工具 | 文件 | 功能 |
|------|------|------|
| `present_files` | `present_file_tool.py` | 展示输出文件 |
| `ask_clarification` | `clarification_tool.py` | 请求用户澄清（被 ClarificationMiddleware 拦截） |
| `view_image` | `view_image_tool.py` | 图片查看（需要模型支持 vision） |
| `task` | `task_tool.py` | 子 Agent 委托 |
| `tool_search` | `tool_search.py` | 动态搜索可用工具 |
| `skill_manage` | `skill_manage_tool.py` | 技能管理（CRUD） |
| `invoke_acp_agent` | `invoke_acp_agent_tool.py` | 调用外部 ACP Agent |

#### 3.4.4 工具注册流程

```
config.yaml tools[]
    │
    ▼
resolve_variable(cfg.use, BaseTool)  // 通过 reflection 动态加载
    │
    ▼
loaded_tools + builtin_tools + mcp_tools + acp_tools
    │
    ▼
去重（按工具名称）→ get_available_tools()
```

---

### 3.5 沙箱系统 (`sandbox/`)

#### 3.5.1 架构

```
Sandbox (抽象接口)
    │
    ├── execute_command()  // 执行命令
    ├── read_file()       // 读文件
    ├── write_file()      // 写文件
    └── list_dir()        // 列目录
    │
    ▼
SandboxProvider (提供者模式)
    │
    ├── LocalSandboxProvider (本地文件系统)
    └── AioSandboxProvider (Docker 隔离)
```

#### 3.5.2 虚拟路径系统

| 虚拟路径 | 物理路径 |
|----------|----------|
| `/mnt/user-data/workspace` | `backend/.deer-flow/threads/{thread_id}/user-data/workspace` |
| `/mnt/user-data/uploads` | `backend/.deer-flow/threads/{thread_id}/user-data/uploads` |
| `/mnt/user-data/outputs` | `backend/.deer-flow/threads/{thread_id}/user-data/outputs` |
| `/mnt/skills` | `deer-flow/skills/` |

**转换函数**:
- `replace_virtual_path()` - 虚拟路径转物理路径
- `replace_virtual_paths_in_command()` - 命令中的路径转换

#### 3.5.3 沙箱工具 (`sandbox/tools.py`)

| 工具 | 功能 |
|------|------|
| `bash` | 执行 shell 命令 |
| `ls` | 目录列表（树形，最多 2 层） |
| `read_file` | 读文件（支持行范围） |
| `write_file` | 写/追加文件 |
| `str_replace` | 字符串替换（单处或全部） |

---

### 3.6 子 Agent 系统 (`subagents/`)

#### 3.6.1 内置子 Agent

| 名称 | 工具范围 | 用途 |
|------|----------|------|
| `general-purpose` | 所有工具（除 `task` 外） | 通用任务 |
| `bash` | 仅 bash 工具 | 命令执行专家 |

#### 3.6.2 执行架构

```
SubagentExecutor
    │
    ├── _scheduler_pool (3 workers)  // 调度线程池
    └── _execution_pool (3 workers)  // 执行线程池
```

**并发控制**:
- `MAX_CONCURRENT_SUBAGENTS = 3`
- `SubagentLimitMiddleware` 截断超限的 `task` 工具调用
- 超时: 15 分钟

#### 3.6.3 执行流程

```
task_tool(input)
    │
    ▼
SubagentExecutor.submit()
    │
    ▼
后台线程 → 创建子图 → astream()
    │
    ├── task_started    // 事件
    ├── task_running    // 事件
    ├── task_completed  // 最终结果
    ├── task_failed     // 错误
    └── task_timed_out  // 超时
```

---

### 3.7 MCP 系统 (`mcp/`)

基于 `langchain-mcp-adapters` 的 `MultiServerMCPClient`。

#### 3.7.1 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| `MultiServerMCPClient` | `langchain-mcp-adapters` | 多服务器管理 |
| `build_server_params()` | `client.py` | 构建服务器参数 |
| `get_cached_mcp_tools()` | `cache.py` | 获取缓存的工具 |
| `reset_mcp_tools_cache()` | `cache.py` | 重置缓存 |

#### 3.7.2 传输类型

| 类型 | 用途 | 配置字段 |
|------|------|----------|
| stdio | 命令行 MCP 服务器 | `command`, `args`, `env` |
| SSE | Server-Sent Events | `url`, `headers` |
| HTTP | HTTP 请求 | `url`, `headers` |

#### 3.7.3 OAuth 支持

位于 `mcp/oauth.py`，支持:
- `client_credentials`
- `refresh_token` 自动刷新

#### 3.7.4 生命周期

```
初始化 ──▶ 首次使用 ──▶ 缓存工具
              │
              ▼
         配置变更检测 (mtime)
              │
              ▼
         重新加载工具
```

---

### 3.8 运行时系统 (`runtime/`)

Gateway 模式下嵌入 Agent 运行时，无需独立的 LangGraph Server。

#### 3.8.1 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| `RunManager` | `runs/manager.py` | 运行记录管理 |
| `RunRecord` | `runs/manager.py` | 单个运行记录 |
| `run_agent()` | `runs/worker.py` | 后台 Agent 执行 |
| `StreamBridge` | `stream_bridge/` | SSE 流式桥接 |
| `Store` | `store/` | 状态存储 |

#### 3.8.2 运行状态机

```
pending ──▶ running ──▶ success
              │           │
              │           ▼
              │         error
              │
              ▼
        interrupted
```

#### 3.8.3 StreamBridge

- 支持多订阅者 fan-out
- `Last-Event-ID` replay
- 心跳保活
- SSE 序列化

---

### 3.9 技能系统 (`skills/`)

#### 3.9.1 技能位置

| 路径 | 用途 | Git |
|------|------|-----|
| `skills/public/` | 内置技能 | 跟踪 |
| `skills/custom/` | 自定义技能 | 忽略 |

#### 3.9.2 技能格式

**目录结构**:
```
skill-name/
└── SKILL.md          # 必需，YAML frontmatter + Markdown 内容
scripts/              # 可选，支持文件
└── *.sh
```

**SKILL.md frontmatter**:
```yaml
---
name: skill-name
description: 技能描述
license: MIT
category: developer
allowed-tools:
  - bash
  - read_file
---
```

#### 3.9.3 生命周期

```
load_skills()
    │
    ▼
递归扫描 skills/{public,custom}/
    │
    ▼
解析 SKILL.md frontmatter
    │
    ▼
读取 extensions_config.json 获取启用状态
    │
    ▼
apply_prompt_template() 注入系统提示词
```

#### 3.9.4 安装流程

```
POST /api/skills/install (或 install_skill_from_archive)
    │
    ▼
验证 .skill 归档格式
    │
    ▼
安全扫描 (scan_skill_content)
    │
    ▼
解压到 skills/custom/
    │
    ▼
reload_extensions_config()
```

---

### 3.10 社区工具集成 (`community/`)

| 集成 | 目录 | 用途 |
|------|------|------|
| aio_sandbox | `aio_sandbox/` | Docker 容器化沙箱 |
| Tavily | `tavily/` | 网页搜索和内容获取 |
| Jina AI | `jina_ai/` | 网页内容提取和可读性处理 |
| Firecrawl | `firecrawl/` | 网页爬取 |
| DuckDuckGo | `ddg_search/` | 搜索 |
| Image Search | `image_search/` | 图片搜索 |
| Infoquest | `infoquest/` | 信息查询 |

---

## 四、服务入口点与集成方式

### 4.1 入口文件

| 入口 | 文件 | 用途 |
|------|------|------|
| **Python 客户端** | `client.py` | `DeerFlowClient` 嵌入式调用 |
| **LangGraph Agent** | `agents/lead_agent/agent.py` | `make_lead_agent()` |
| **Gateway API** | `app/gateway/app.py` | FastAPI 应用 |
| **运行时 Worker** | `runtime/runs/worker.py` | `run_agent()` |

### 4.2 系统部署架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           外部客户端                                     │
│                         (浏览器 / API)                                   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ HTTP/SSE
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              Nginx                                      │
│                         (Port 2026)                                     │
│                                                                         │
│   /api/langgraph/* ──────────▶ LangGraph Server (Port 2024)            │
│   /api/* (others) ───────────▶ Gateway API (Port 8001)                  │
│   / ─────────────────────────▶ Frontend (Port 3000)                     │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
┌───────────────────────────┐   ┌─────────────────────────────────────────┐
│    LangGraph Server       │   │            Gateway API                    │
│      (Port 2024)          │   │            (Port 8001)                   │
│                           │   │                                         │
│  make_lead_agent()       │   │  FastAPI + run_agent()                  │
│  Agent Runtime            │   │  (Gateway 模式)                           │
│                           │   │                                         │
│  - Standalone 模式        │   │  - Gateway 模式 (嵌入 runtime)            │
│  - 独立进程                │   │  - 单一进程                              │
└───────────────────────────┘   └─────────────────────────────────────────┘
                                  │                    │
                                  │                    │
                                  ▼                    ▼
                    ┌─────────────────────────┐  ┌──────────────────────┐
                    │     Frontend           │  │   Provisioner        │
                    │    (Next.js)          │  │   (Port 8002)        │
                    │    (Port 3000)         │  │   (Optional)         │
                    └─────────────────────────┘  └──────────────────────┘
```

### 4.3 前后端集成

| 端 | 通信方式 | 端口 |
|----|----------|------|
| Frontend → LangGraph | HTTP + SSE | 2024 (直连) 或 2026 (经 Nginx) |
| Frontend → Gateway | REST API | 8001 (直连) 或 2026 (经 Nginx) |
| Gateway → LangGraph | langchain-sdk | 2024 |

### 4.4 外部服务集成

| 服务 | 集成模块 | 用途 |
|------|----------|------|
| Claude API | `models/claude_provider.py` | LLM 推理 |
| OpenAI API | `models/openai_codex_provider.py` | LLM 推理 |
| vLLM Server | `models/vllm_provider.py` | 自托管 LLM |
| Tavily | `community/tavily/` | 网页搜索 |
| Jina AI | `community/jina_ai/` | 内容提取 |
| Firecrawl | `community/firecrawl/` | 网页爬取 |
| Docker | `community/aio_sandbox/` | 沙箱隔离 |
| MCP Servers | `mcp/` | 扩展工具协议 |
| ACP Agents | `tools/invoke_acp_agent_tool.py` | 外部 Agent |

---

## 五、模块依赖关系图

```
                            ┌───────────────────────────────────────┐
                            │             client.py                  │
                            │         DeerFlowClient                 │
                            │   (嵌入式 Python 客户端入口)            │
                            └───────────────────┬───────────────────┘
                                                │ 使用
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        agents/lead_agent/agent.py                            │
│                          make_lead_agent()                                   │
│                    (应用级 Agent 工厂，读取配置)                               │
└────┬────────┬────────┬────────┬────────┬────────┬────────┬─────────┬──────────┘
     │        │        │        │        │        │        │         │
     ▼        ▼        ▼        ▼        ▼        ▼        ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│ models │ │ tools/ │ │sandbox/│ │ config/│ │  mcp/  │ │subagents│ │runtime/ │
│        │ │        │ │        │ │        │ │        │ │        │ │         │
│factory │ │tools.py│ │        │ │        │ │        │ │        │ │         │
└────┬───┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └────┬────┘
     │          │          │          │          │          │           │
     ▼          ▼          ▼          ▼          ▼          ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│claude   │ │sandbox  │ │sandbox  │ │app_config│ │cache   │ │executor │ │runs/    │
│openai   │ │tools    │ │provider │ │extensions│ │client  │ │registry │ │worker.py│
│vllm     │ │builtins │ │local/   │ │config   │ │oauth   │ │builtins │ │         │
│minimax  │ │mcp      │ │aio_sbox │ │paths    │ │tools   │ │         │ │         │
└─────────┘ │community│ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
            │tavily   │
            │jina_ai  │
            │firecrawl│
            │ddg_search│
            └─────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          community/                                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐              │
│  │aio_sandbox │ │  tavily   │ │  jina_ai  │ │ firecrawl  │ ...          │
│  │ (Docker)   │ │ (搜索)    │ │  (提取)   │ │  (爬取)   │              │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 依赖方向说明

```
client.py
    │
    ├──▶ agents/lead_agent/agent.py
    │        │
    │        ├──▶ models/factory.py
    │        ├──▶ tools/tools.py
    │        ├──▶ config/ (多模块)
    │        └──▶ sandbox/
    │
    ├──▶ agents/memory/updater.py
    │
    └──▶ skills/loader.py

runtime/runs/worker.py (Gateway 模式)
    │
    ├──▶ agents/lead_agent/agent.py
    │
    └──▶ runtime/stream_bridge/
```

**重要约束**: `deerflow.*` 模块**禁止**导入 `app.*` 模块（由 `tests/test_harness_boundary.py` 强制执行）

---

## 六、数据流

### 6.1 Agent 对话流程

```
用户消息 (HumanMessage)
    │
    ▼
DeerFlowClient.stream(message, thread_id)
    │
    ├── _get_runnable_config()  // 构建 RunnableConfig
    │
    ├── _ensure_agent()          // 确保 Agent 已创建
    │        │
    │        ├──▶ create_chat_model()
    │        ├──▶ get_available_tools()
    │        ├──▶ _build_middlewares()
    │        └──▶ apply_prompt_template()
    │
    ▼
agent.stream(state, config)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph 执行                           │
│                      │                                     │
│  ┌──────────────────┼──────────────────┐                   │
│  ▼                  ▼                  ▼                   │
│ ThreadData ──▶ Uploads ──▶ Sandbox ──▶ LLM ──▶ [Tools]    │
│  Middleware  Middleware  Middleware  Error                   │
│                                                        ▼   │
│                    ┌─────────────────────────────────────┐  │
│                    │         中间件链继续                 │  │
│                    │  Guardrail → Audit → ToolError ──▶   │  │
│                    │  Summarization → Memory → Loop ──▶  │  │
│                    │  Clarification (最后)                 │  │
│                    └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
工具调用
    │
    ├──▶ 沙箱工具 (bash, ls, read, write)
    ├──▶ 内置工具 (present_files, ask_clarification)
    ├──▶ MCP 工具 (来自外部服务器)
    ├──▶ 社区工具 (tavily, jina_ai, firecrawl)
    └──▶ 子 Agent (task 委托)
    │
    ▼
响应序列化 ──▶ StreamEvent
    │
    ├── type="values"        // 完整状态快照
    ├── type="messages-tuple" // 消息元组（AI 文本增量、工具调用、工具结果）
    ├── type="custom"        // 自定义事件
    └── type="end"           // 流结束
```

### 6.2 文件上传流程

```
DeerFlowClient.upload_files(thread_id, files)
    │
    ▼
验证文件列表
    │
    ├── 检查文件存在
    ├── 检查是普通文件（非目录）
    └── 生成唯一文件名
    │
    ▼
复制到 uploads 目录
    │
    ▼
可转换文件处理 (PDF, PPT, Excel, Word)
    │
    ▼
markitdown 转换为 Markdown
    │
    ▼
enrich_file_listing() ──▶ 返回文件元信息
    │
    ├── filename
    ├── size
    ├── path
    ├── virtual_path
    ├── artifact_url
    └── markdown_file (如已转换)
```

### 6.3 MCP 工具加载流程

```
initialize_mcp_tools()
    │
    ▼
ExtensionsConfig.from_file()
    │
    ▼
获取启用的 MCP 服务器列表
    │
    ▼
build_servers_config() ──▶ 构建 MultiServerMCPClient 配置
    │
    ▼
MultiServerMCPClient() ──▶ 连接所有 MCP 服务器
    │
    ▼
get_mcp_tools() ──▶ 获取所有工具
    │
    ▼
缓存到 get_cached_mcp_tools()
```

---

## 七、关键设计模式

### 7.1 提供者模式 (Provider Pattern)

```python
class SandboxProvider(ABC):
    @abstractmethod
    def acquire(self) -> Sandbox: ...

    @abstractmethod
    def get(self, sandbox_id: str) -> Sandbox | None: ...

    @abstractmethod
    def release(self, sandbox_id: str) -> None: ...
```

**应用场景**:
- `SandboxProvider` - 沙箱抽象
- `CheckpointProvider` - 状态持久化
- `SandboxBackend` - Docker/本地后端

### 7.2 中间件模式 (Middleware Pattern)

```python
class AgentMiddleware(ABC):
    async def before_model(self, ...): ...

    async def after_model(self, ...): ...

    async def on_tool_start(self, ...): ...

    async def on_tool_end(self, ...): ...
```

**特点**:
- 14+ 中间件组成处理链
- 支持 `@Next`/`@Prev` 锚定插入

### 7.3 工厂模式 (Factory Pattern)

| 工厂 | 函数 | 用途 |
|------|------|------|
| Agent 工厂 | `create_deerflow_agent()` | SDK 工厂，纯 Python 参数 |
| Agent 工厂 | `make_lead_agent()` | 应用工厂，读取配置 |
| 模型工厂 | `create_chat_model()` | 创建 LLM 实例 |

### 7.4 缓存模式 (Cache Pattern)

| 缓存项 | 模块 | 失效策略 |
|--------|------|----------|
| MCP 工具 | `mcp/cache.py` | mtime 检测 |
| AppConfig | `config/app_config.py` | 文件变更检测 |
| Skills | `skills/loader.py` | 重载时刷新 |
| 系统提示词 | `agents/lead_agent/prompt.py` | 配置变更时刷新 |

### 7.5 反射模式 (Reflection Pattern)

位于 `reflection/`:

```python
resolve_variable("deerflow.sandbox.tools:bash_tool")  # 导入模块变量
resolve_class("deerflow.models.claude_provider:ClaudeChatModel", BaseChatModel)  # 导入并验证
```

---

## 八、配置驱动设计

### 8.1 config.yaml 结构

```yaml
config_version: "1.0"

models:
  - name: claude
    model: anthropic/claude-sonnet-4-20250514
    supports_thinking: true
    supports_vision: true

tools:
  - name: bash
    use: deerflow.sandbox.tools:bash_tool
    group: bash

tool_groups:
  - name: bash
    tools: [bash]

sandbox:
  use: deerflow.sandbox.local:LocalSandboxProvider

skills:
  path: ./skills
  container_path: /mnt/skills

title:
  enabled: true
  max_words: 20

summarization:
  enabled: false

subagents:
  enabled: false

memory:
  enabled: false
  storage_path: .deer-flow/memory.json
  debounce_seconds: 30
  max_facts: 100
```

### 8.2 extensions_config.json 结构

```json
{
  "mcpServers": {
    "filesystem": {
      "enabled": true,
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "./workspace"]
    }
  },
  "skills": {
    "my-skill": {
      "enabled": true
    }
  }
}
```

---

## 九、线程隔离机制

每个对话线程 (`thread_id`) 有完全独立的运行环境：

```
.deer-flow/
└── threads/
    └── {thread_id}/
        ├── user-data/
        │   ├── workspace/     # 沙箱工作目录
        │   ├── uploads/       # 上传的文件
        │   └── outputs/       # 生成的产物
        └── checkpoint/        # LangGraph 检查点
```

**隔离内容**:
- 文件系统访问
- 内存状态 (通过 checkpointer)
- 上传文件
- 生成的产物

---

## 十、总结

### 10.1 架构设计要点

1. **模块化分层**: agents / tools / sandbox / models / config / runtime 各司其职
2. **配置驱动**: 通过 `config.yaml` 和 `extensions_config.json` 灵活配置
3. **中间件链**: 14+ 中间件提供日志、安全、记忆、循环检测等功能
4. **提供者模式**: 沙箱、checkpoint 等核心组件支持多种实现
5. **双重入口**: 支持独立 LangGraph Server 模式或嵌入式 Gateway 模式
6. **线程隔离**: 每个对话线程有独立的文件系统和状态
7. **反射加载**: 工具和模型通过字符串路径动态加载

### 10.2 扩展点

| 扩展点 | 方式 | 示例 |
|--------|------|------|
| 新模型 | 实现 `BaseChatModel` | MiniMax、DeepSeek providers |
| 新工具 | 实现 `BaseTool` | 社区工具 |
| 新沙箱 | 实现 `SandboxProvider` | Kubernetes provider |
| 新中间件 | 继承 `AgentMiddleware` | 自定义安全审计 |
| 新记忆后端 | 实现存储接口 | Vector store |

### 10.3 技术栈

- **框架**: LangGraph, LangChain
- **Web**: FastAPI, SSE
- **协议**: MCP (Model Context Protocol)
- **隔离**: Docker, Local filesystem
- **异步**: asyncio
