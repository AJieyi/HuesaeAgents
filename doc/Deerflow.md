# DeerFlow 系统架构说明

> 本文档基于 DeerFlow v0.1.0 源码分析生成，详细描述系统的整体设计、模块划分和服务集成方式。

## 目录

- [一、项目概述](#一项目概述)
- [二、顶层目录结构](#二顶层目录结构)
- [三、后端架构](#三后端架构)
  - [3.1 模块划分](#31-模块划分)
  - [3.2 Agent 系统](#32-agent-系统)
  - [3.3 沙箱系统](#33-沙箱系统)
  - [3.4 工具系统](#34-工具系统)
  - [3.5 子Agent系统](#35-子agent系统)
  - [3.6 MCP 系统](#36-mcp-系统)
  - [3.7 模型工厂](#37-模型工厂)
  - [3.8 Gateway API](#38-gateway-api)
  - [3.9 IM 频道集成](#39-im-频道集成)
- [四、前端架构](#四前端架构)
  - [4.1 技术栈](#41-技术栈)
  - [4.2 目录结构](#42-目录结构)
  - [4.3 核心模块](#43-核心模块)
- [五、服务间依赖关系](#五服务间依赖关系)
- [六、数据流](#六数据流)
  - [6.1 Web Chat 流程](#61-web-chat-流程)
  - [6.2 IM 频道流程](#62-im-频道流程)
- [七、配置系统](#七配置系统)
- [八、关键入口文件](#八关键入口文件)
- [九、架构设计要点](#九架构设计要点)

---

## 一、项目概述

DeerFlow 是一个基于 **LangGraph** 的 AI Super Agent 系统，采用前后端分离架构：

- **后端**: Python 实现，提供基于 LangGraph 的 Agent 运行时，支持沙箱执行、持久化记忆、子 Agent 委托和可扩展工具集成
- **前端**: Next.js 16 Web 界面，提供基于线程的 AI 对话、实时流式响应、工件展示和技能管理系统
- **通信协议**: LangGraph SDK + REST API

### 运行时架构

| 服务 | 端口 | 技术栈 | 职责 |
|------|------|--------|------|
| LangGraph Server | 2024 | LangGraph | Agent 运行时和工作流执行 |
| Gateway API | 8001 | FastAPI | REST API (模型/MCP/技能/记忆/工件/上传) |
| Frontend | 3000 | Next.js | Web 用户界面 |
| Nginx | 2026 | Nginx | 统一反向代理入口 |
| Provisioner | 8002 | Kubernetes | 可选，仅在沙箱配置为 Kubernetes 模式时启动 |

### 运行时模式

| 模式 | 说明 | 进程数 |
|------|------|--------|
| **标准模式** (`make dev`) | LangGraph Server 作为独立进程处理 Agent 执行 | 4 |
| **Gateway 模式** (`make dev-pro`) | Agent 运行时嵌入 Gateway via `RunManager` + `run_agent()` + `StreamBridge` | 3 |

---

## 二、顶层目录结构

```
deer-flow/
├── backend/                     # Python 后端 (LangGraph + FastAPI)
│   ├── packages/
│   │   └── harness/            # deerflow-harness 包 (可发布)
│   │       └── deerflow/
│   │           ├── agents/     # LangGraph Agent 系统
│   │           ├── sandbox/   # 沙箱执行系统
│   │           ├── subagents/ # 子Agent委托系统
│   │           ├── tools/     # 工具系统
│   │           ├── mcp/       # MCP 集成
│   │           ├── models/    # 模型工厂
│   │           ├── skills/    # 技能系统
│   │           ├── config/    # 配置系统
│   │           ├── community/  # 社区工具
│   │           ├── reflection/# 动态模块加载
│   │           ├── guardrails/# 安全防护
│   │           └── client.py  # 嵌入式 Python Client
│   ├── app/
│   │   ├── gateway/           # FastAPI 网关 API
│   │   └── channels/         # IM 平台集成
│   ├── tests/                 # 测试套件
│   └── docs/                  # 后端文档
│
├── frontend/                   # Next.js 16 前端
│   ├── src/
│   │   ├── app/              # Next.js App Router
│   │   ├── components/       # React 组件
│   │   ├── core/            # 核心业务逻辑
│   │   ├── hooks/          # React Hooks
│   │   ├── lib/            # 工具函数
│   │   └── server/         # 服务端代码
│   └── tests/              # 测试套件
│
├── skills/                     # Agent 技能目录
│   ├── public/              # 内置技能 (已提交)
│   └── custom/              # 自定义技能 (gitignored)
│
├── docker/                    # Docker 配置
├── docs/                     # 项目文档
├── scripts/                  # 脚本
├── config.example.yaml       # 主配置文件模板
└── extensions_config.json    # MCP 和技能配置模板
```

---

## 三、后端架构

### 3.1 模块划分

后端分为两个严格解耦的层次，遵循单向依赖原则：

```
┌─────────────────────────────────────────────────────────────┐
│                    App Layer (应用层)                        │
│            Import: app.* — 不可导入 harness                   │
├─────────────────────────────────────────────────────────────┤
│  app/gateway/         — FastAPI 网关 API                      │
│  app/channels/       — IM 平台集成 (飞书/Slack/Telegram)      │
└─────────────────────────────────────────────────────────────┘
                              ▲ 依赖
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Harness Layer (框架层)                      │
│         Import: deerflow.* — 可独立发布为 pip 包              │
├─────────────────────────────────────────────────────────────┤
│  packages/harness/deerflow/                                 │
│    ├── agents/      — Agent 系统 + 中间件                     │
│    ├── sandbox/     — 沙箱执行                               │
│    ├── subagents/   — 子 Agent 委托                          │
│    ├── tools/       — 工具                                   │
│    ├── mcp/         — MCP 协议                               │
│    ├── models/      — 模型工厂                               │
│    ├── skills/      — 技能                                   │
│    ├── config/      — 配置                                   │
│    ├── community/   — 社区工具集成                           │
│    ├── reflection/  — 动态加载                               │
│    ├── guardrails/  — 安全防护                               │
│    └── client.py    — 嵌入式客户端                           │
└─────────────────────────────────────────────────────────────┘
```

**依赖规则**: App 可以导入 deerflow，但 deerflow 禁止导入 app。这一边界由 `tests/test_harness_boundary.py` 在 CI 中强制执行。

### 3.2 Agent 系统

**路径**: `packages/harness/deerflow/agents/`

#### 3.2.1 Lead Agent (主 Agent)

| 文件 | 功能 |
|------|------|
| `lead_agent/agent.py` | Lead Agent 实现，`make_lead_agent()` 入口 |
| `lead_agent/prompt.py` | 系统 Prompt 模板生成 |
| `factory.py` | Agent 工厂，创建 Lead Agent |

**Lead Agent 职责**:
- 接收用户消息
- 通过中间件链处理消息
- 调用工具执行任务
- 生成 AI 响应
- 管理线程状态

#### 3.2.2 中间件链

Lead Agent 配置了 **18 个中间件**，按顺序执行：

| 序号 | 中间件 | 文件 | 职责 |
|------|--------|------|------|
| 1 | ThreadDataMiddleware | `middlewares/` | 创建线程目录结构 |
| 2 | UploadsMiddleware | `middlewares/` | 注入上传文件信息 |
| 3 | SandboxMiddleware | `middlewares/` | 获取沙箱实例 |
| 4 | DanglingToolCallMiddleware | `middlewares/` | 注入占位 ToolMessage |
| 5 | LLMErrorHandlingMiddleware | `middlewares/` | 规范化 LLM 错误 |
| 6 | GuardrailMiddleware | `middlewares/` | 工具调用授权 |
| 7 | SandboxAuditMiddleware | `middlewares/` | 沙箱操作审计 |
| 8 | ToolErrorHandlingMiddleware | `middlewares/` | 工具异常处理 |
| 9 | SummarizationMiddleware | `middlewares/` | 上下文压缩 |
| 10 | TodoListMiddleware | `middlewares/` | 任务跟踪 (plan_mode) |
| 11 | TokenUsageMiddleware | `middlewares/` | Token 使用记录 |
| 12 | TitleMiddleware | `middlewares/` | 自动生成线程标题 |
| 13 | MemoryMiddleware | `middlewares/` | 记忆更新队列 |
| 14 | ViewImageMiddleware | `middlewares/` | 注入图像数据 |
| 15 | DeferredToolFilterMiddleware | `middlewares/` | 隐藏延迟工具 |
| 16 | SubagentLimitMiddleware | `middlewares/` | 限制并发子 Agent |
| 17 | LoopDetectionMiddleware | `middlewares/` | 检测循环调用 |
| 18 | ClarificationMiddleware | `middlewares/` | 处理澄清请求 |

#### 3.2.3 线程状态

**文件**: `agents/thread_state.py`

```python
class ThreadState(AgentState):
    sandbox: Optional[str]              # 沙箱 ID
    thread_data: ThreadData             # 线程元数据
    title: Optional[str]                 # 线程标题
    artifacts: list[Artifact]           # 工件列表
    todos: list[Todo]                    # 任务列表
    uploaded_files: list[str]           # 上传文件
    viewed_images: list[str]           # 查看的图片
```

#### 3.2.4 记忆系统

**路径**: `agents/memory/`

| 文件 | 功能 |
|------|------|
| `updater.py` | LLM 记忆更新，事实提取，去重 |
| `queue.py` | 消息队列，防抖处理 |
| `prompt.py` | 记忆更新 Prompt 模板 |

**记忆数据结构** (存储在 `backend/.deer-flow/memory.json`):
- **User Context**: workContext, personalContext, topOfMind
- **History**: recentMonths, earlierContext, longTermBackground
- **Facts**: 离散事实 (id, content, category, confidence, createdAt, source)

### 3.3 沙箱系统

**路径**: `packages/harness/deerflow/sandbox/`

#### 3.3.1 架构

```
┌─────────────────────────────────────────────┐
│              SandboxProvider                 │
│         (acquire / get / release)           │
└─────────────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌─────────────────┐     ┌─────────────────────────┐
│ LocalSandbox    │     │ AioSandboxProvider      │
│ (本地文件系统)   │     │ (Docker 隔离)           │
└─────────────────┘     └─────────────────────────┘
```

#### 3.3.2 核心文件

| 文件 | 功能 |
|------|------|
| `sandbox.py` | 抽象 `Sandbox` 接口 |
| `sandbox_provider.py` | `SandboxProvider` 生命周期管理 |
| `local/` | 本地沙箱实现 |
| `tools.py` | 沙箱工具 (bash/ls/read_file/write_file/str_replace) |
| `middleware.py` | 沙箱生命周期中间件 |

#### 3.3.3 虚拟路径系统

Agent 看到的路径与物理路径解耦：

| 虚拟路径 | 物理路径 |
|----------|----------|
| `/mnt/user-data/workspace` | `backend/.deer-flow/threads/{thread_id}/user-data/workspace` |
| `/mnt/user-data/uploads` | `backend/.deer-flow/threads/{thread_id}/user-data/uploads` |
| `/mnt/user-data/outputs` | `backend/.deer-flow/threads/{thread_id}/user-data/outputs` |
| `/mnt/skills` | `deer-flow/skills/` |

#### 3.3.4 沙箱工具

| 工具 | 功能 |
|------|------|
| `bash` | 执行命令，带路径转换 |
| `ls` | 目录列表 (树状，最多 2 层) |
| `read_file` | 读取文件内容，支持行范围 |
| `write_file` | 写入/追加文件，自动创建目录 |
| `str_replace` | 字符串替换 (单次或全部) |

### 3.4 工具系统

**路径**: `packages/harness/deerflow/tools/` 和 `packages/harness/deerflow/community/`

#### 3.4.1 工具分类

| 类型 | 来源 | 示例 |
|------|------|------|
| 内置工具 | `tools/builtins/` | present_files, ask_clarification, view_image |
| 沙箱工具 | `sandbox/tools.py` | bash, ls, read_file, write_file, str_replace |
| MCP 工具 | `mcp/tools.py` | 来自启用的 MCP 服务器 |
| 社区工具 | `community/` | tavily, firecrawl, jina_ai, image_search |
| 子 Agent 工具 | `subagents/` | task (委托任务) |

#### 3.4.2 工具加载

`get_available_tools(groups, include_mcp, model_name, subagent_enabled)` 组合：
1. `config.yaml` 中定义的工具
2. MCP 服务器提供的工具 (惰性初始化)
3. 内置工具
4. 子 Agent 工具 (如果启用)

#### 3.4.3 社区工具

| 工具 | 路径 | 功能 |
|------|------|------|
| Tavily Search | `community/tavily/` | 网络搜索 (默认 5 条结果) |
| Tavily Fetch | `community/tavily/` | 网页内容获取 (4KB 限制) |
| Jina AI | `community/jina_ai/` | Jina Reader API 网页抓取 |
| Firecrawl | `community/firecrawl/` | Firecrawl API 爬虫 |
| Image Search | `community/image_search/` | DuckDuckGo 图片搜索 |

### 3.5 子Agent系统

**路径**: `packages/harness/deerflow/subagents/`

#### 3.5.1 架构

```
              task() 工具调用
                    │
                    ▼
         ┌──────────────────────┐
         │  SubagentExecutor    │
         │  (后台线程池执行)    │
         └──────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
┌────────────┐ ┌──────────┐ ┌──────────┐
│general-    │ │  bash    │ │ (自定义   │
│purpose     │ │ agent    │ │  agents) │
└────────────┘ └──────────┘ └──────────┘
```

#### 3.5.2 核心文件

| 文件 | 功能 |
|------|------|
| `registry.py` | Agent 注册表 |
| `executor.py` | 后台执行引擎 (线程池) |
| `builtins/general_purpose/` | 通用 Agent (所有工具) |
| `builtins/bash/` | Bash 专用 Agent |

#### 3.5.3 执行参数

- 最大并发: `MAX_CONCURRENT_SUBAGENTS = 3`
- 超时时间: 15 分钟
- 线程池: `_scheduler_pool` (3 workers) + `_execution_pool` (3 workers)

### 3.6 MCP 系统

**路径**: `packages/harness/deerflow/mcp/`

#### 3.6.1 架构

使用 `langchain-mcp-adapters` 的 `MultiServerMCPClient` 管理多服务器 MCP 连接。

#### 3.6.2 核心文件

| 文件 | 功能 |
|------|------|
| `client.py` | MCP 客户端封装 |
| `tools.py` | MCP 工具加载 (惰性初始化，mtime 缓存失效) |
| `cache.py` | 缓存管理 |
| `oauth.py` | OAuth 令牌刷新支持 |

#### 3.6.3 传输协议

| 协议 | 说明 |
|------|------|
| stdio | 基于命令的传输 |
| SSE | Server-Sent Events |
| HTTP | HTTP 传输 |
| OAuth | 支持 token endpoint 流程 (client_credentials, refresh_token) |

### 3.7 模型工厂

**路径**: `packages/harness/deerflow/models/`

#### 3.7.1 核心文件

| 文件 | 功能 |
|------|------|
| `factory.py` | `create_chat_model()` 动态创建 LLM |
| `vllm_provider.py` | vLLM 兼容推理模型支持 |
| `openai_codex_provider.py` | OpenAI Codex 支持 |
| `claude_provider.py` | Claude 模型支持 |
| `mindie_provider.py` | Mindie 模型支持 |
| `patched_*.py` | 各模型的补丁实现 |

#### 3.7.2 模型配置

在 `config.yaml` 中配置：

```yaml
models:
  - name: gpt-4o
    use: langchain_openai:ChatOpenAI
    provider: openai
    supports_thinking: true
    supports_vision: true
```

### 3.8 Gateway API

**路径**: `app/gateway/`

FastAPI 应用 (端口 8001)，提供 REST API。

#### 3.8.1 入口文件

| 文件 | 功能 |
|------|------|
| `app.py` | FastAPI 应用入口 |
| `services.py` | 服务依赖注入 |
| `deps.py` | 依赖项定义 |
| `routers/` | 路由模块 |

#### 3.8.2 路由模块

| 路由 | 文件 | 功能 |
|------|------|------|
| `/api/models` | `models.py` | 列出/获取模型信息 |
| `/api/mcp` | `mcp.py` | MCP 配置管理 |
| `/api/skills` | `skills.py` | 技能列表/安装/更新 |
| `/api/memory` | `memory.py` | 记忆数据管理 |
| `/api/threads/{id}/uploads` | `uploads.py` | 文件上传/列表/删除 |
| `/api/threads/{id}/artifacts` | `artifacts.py` | 工件访问 |
| `/api/suggestions` | `suggestions.py` | 后续问题建议 |
| `/api/runs` | `runs.py` | 运行管理 |
| `/api/channels` | `channels.py` | 频道管理 |

### 3.9 IM 频道集成

**路径**: `app/channels/`

#### 3.9.1 支持的平台

| 平台 | 文件 | 说明 |
|------|------|------|
| 飞书 | `feishu.py` | 企业级 IM，支持增量更新 |
| Slack | `slack.py` | Bot 集成 |
| Telegram | `telegram.py` | Bot 集成 |
| 企业微信 | `wecom.py` | 企业微信集成 |
| Discord | `discord.py` | Discord 集成 |

#### 3.9.2 核心组件

| 文件 | 功能 |
|------|------|
| `base.py` | 抽象 `Channel` 基类 |
| `manager.py` | 核心分发器，线程管理 |
| `message_bus.py` | 异步发布/订阅消息总线 |
| `service.py` | 频道生命周期管理 |
| `store.py` | JSON 持久化 (channel:chat → thread_id) |

#### 3.9.3 消息流

```
外部平台 → Channel 实现 → MessageBus.publish_inbound()
    → ChannelManager._dispatch_loop()
    → LangGraph Server (创建线程/发送消息)
    → AI 响应 → OutboundMessage → 平台回复
```

---

## 四、前端架构

### 4.1 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Next.js | 16 | React 框架 |
| React | 19 | UI 库 |
| TypeScript | 5.8 | 类型系统 |
| Tailwind CSS | 4 | 样式 |
| TanStack Query | 5 | 服务端状态管理 |
| Radix UI | - | UI 组件原语 |
| Vercel AI SDK | 6 | AI 流式响应 |
| Better Auth | 1.3 | 认证 |
| pnpm | 10.26.2 | 包管理器 |

### 4.2 目录结构

```
frontend/src/
├── app/                          # Next.js App Router
│   ├── layout.tsx               # 根布局
│   ├── page.tsx                 # 落地页
│   ├── [lang]/                  # 多语言支持
│   ├── workspace/               # 工作区
│   │   ├── layout.tsx          # 工作区布局
│   │   ├── page.tsx           # 工作区首页
│   │   └── agents/             # Agent 列表
│   ├── api/                    # API 路由
│   │   └── memory/            # 记忆 API
│   └── blog/                   # 博客
│
├── components/                  # React 组件
│   ├── ui/                     # Shadcn UI 组件 (自动生成)
│   ├── ai-elements/            # Vercel AI SDK 元素 (自动生成)
│   ├── workspace/               # 工作区组件
│   │   ├── chats/              # 聊天组件
│   │   ├── settings/           # 设置组件
│   │   └── ...
│   └── landing/                # 落地页组件
│
├── core/                        # 核心业务逻辑
│   ├── threads/                # 线程管理
│   │   ├── hooks.ts           # 线程操作 hooks
│   │   ├── types.ts           # 类型定义
│   │   └── ...
│   ├── api/                   # API 客户端
│   │   └── index.ts          # LangGraph SDK 封装
│   ├── artifacts/             # 工件管理
│   ├── memory/                # 记忆管理
│   ├── skills/                # 技能管理
│   ├── mcp/                   # MCP 管理
│   ├── models/                # TypeScript 类型
│   ├── messages/              # 消息处理
│   ├── tasks/                 # 任务管理
│   ├── todos/                 # TodoList 管理
│   └── ...
│
├── hooks/                       # 共享 React hooks
├── lib/                        # 工具函数 (cn, etc.)
├── server/                    # 服务端代码
│   └── better-auth/           # 认证实现
├── styles/                    # 全局样式
└── env.js                     # 环境变量验证
```

### 4.3 核心模块

#### 4.3.1 线程系统 (`core/threads/`)

| 文件 | 功能 |
|------|------|
| `hooks.ts` | `useThreadStream`, `useSubmitThread`, `useThreads` |
| `types.ts` | 线程相关 TypeScript 类型 |

#### 4.3.2 API 客户端 (`core/api/`)

| 文件 | 功能 |
|------|------|
| `index.ts` | LangGraph SDK 客户端单例 |
| `stream-mode.ts` | 流式响应处理 |

#### 4.3.3 组件结构

| 组件 | 路径 | 功能 |
|------|------|------|
| Chat Input | `components/workspace/chats/` | 消息输入 |
| Chat Messages | `components/workspace/chats/messages/` | 消息列表 |
| Artifact Viewer | `components/workspace/` | 工件展示 |
| Settings | `components/workspace/settings/` | 设置面板 |

---

## 五、服务间依赖关系

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Nginx (端口 2026)                            │
│                   (统一反向代理入口)                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   /api/langgraph/* ──────────▶ LangGraph Server (2024)               │
│                              ┌──────────────────────────────────┐    │
│                              │         Agent Runtime            │    │
│                              │  ┌────────────────────────────┐  │    │
│                              │  │      Lead Agent            │  │    │
│                              │  │   (18 Middlewares)         │  │    │
│                              │  └────────────────────────────┘  │    │
│                              │              │                   │    │
│                              │  ┌───────────┼───────────┐     │    │
│                              │  ▼           ▼           ▼     │    │
│                              │ Sandbox   Tools      Subagents │    │
│                              │              │                   │    │
│                              │  ┌──────────┼──────────┐      │    │
│                              │  ▼          ▼           ▼      │    │
│                              │ MCP       Memory      Skills   │    │
│                              └──────────────────────────────────┘    │
│                                                                      │
│   /api/* ────────────────▶ Gateway API (8001)                        │
│   (非 langgraph)         ┌──────────────────────────────────┐    │
│                          │  REST API                         │    │
│                          │  • Models                         │    │
│                          │  • Skills                         │    │
│                          │  • Memory                         │    │
│                          │  • Uploads/Artifacts              │    │
│                          │  • MCP Config                     │    │
│                          └──────────────────────────────────┘    │
│                                       │                              │
│                                       ▼                              │
│                              ┌──────────────────────────────────┐   │
│                              │  IM Channels                      │   │
│                              │  • Feishu (飞书)                  │   │
│                              │  • Slack                         │   │
│                              │  • Telegram                      │   │
│                              │  • WeCom (企业微信)               │   │
│                              │  • Discord                       │   │
│                              └──────────────────────────────────┘   │
│                                                                      │
│   / ──────────────────────▶ Frontend (3000)                          │
│   (非 API)                  ┌──────────────────────────────────┐    │
│                              │  Next.js Web UI                  │    │
│                              │  • Landing Page                 │    │
│                              │  • Workspace                    │    │
│                              │  • Chat Interface                │    │
│                              └──────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 六、数据流

### 6.1 Web Chat 流程

```
用户输入消息
      │
      ▼
useSubmitThread()  ──────────────────────────────────────────────┐
      │                                                             │
      ▼                                                             │
LangGraph SDK client.runs.stream()                                 │
      │                                                             │
      ▼                                                             ▼
SSE 流事件 ────────────────▶ StreamBridge ◀───────────────▶ Thread 状态更新
      │                           │                                 │
      │                           ▼                                 │
      │                     ┌─────────────┐                        │
      │                     │  AI 响应    │                        │
      │                     │  • 文本     │                        │
      │                     │  • 工具调用 │                        │
      │                     │  • 工件     │                        │
      │                     │  • Todo    │                        │
      │                     └─────────────┘                        │
      │                                                             │
      └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                         React 组件渲染
                        (消息/工件/Todo)
```

### 6.2 IM 频道流程

```
外部消息 (飞书/Slack/Telegram/...)
      │
      ▼
Channel 实现 (feishu.py/slack.py/telegram.py/...)
      │
      ▼
MessageBus.publish_inbound()
      │
      ▼
ChannelManager._dispatch_loop()
      │
      ├─▶ /new 等命令 ──▶ 本地处理或 Gateway API 查询
      │
      └─▶ 聊天消息
              │
              ▼
         client.threads.create()  ──▶ LangGraph Server
                      │
                      ▼
         client.runs.stream() / client.runs.wait()
                      │
                      ▼
         AI 响应
              │
              ▼
         OutboundMessage
              │
              ▼
         Channel 回调 ──▶ 平台回复
```

### 6.3 飞书特殊流程 (增量更新)

```
用户消息
   │
   ▼
client.runs.stream(["messages-tuple", "values"])
   │
   ▼
累积 AI 文本
   │
   ├─▶ 发布增量更新 (is_final=False) ──▶ 飞书卡片 Patch
   │
   └─▶ 发布最终更新 (is_final=True) ──▶ 飞书卡片 Patch 完成
```

---

## 七、配置系统

### 7.1 主配置文件 (`config.yaml`)

**位置**: 项目根目录

**优先级**:
1. 显式 `config_path` 参数
2. `DEER_FLOW_CONFIG_PATH` 环境变量
3. `config.yaml` (当前目录或项目根目录)

#### 主要配置项

| 章节 | 配置项 | 说明 |
|------|--------|------|
| `models[]` | - | LLM 模型列表 |
| `tools[]` | - | 工具配置 |
| `tool_groups[]` | - | 工具分组 |
| `sandbox` | use | 沙箱提供者 |
| `skills` | path, container_path | 技能路径 |
| `title` | - | 标题生成配置 |
| `summarization` | - | 上下文压缩配置 |
| `subagents` | enabled | 子 Agent 开关 |
| `memory` | - | 记忆系统配置 |
| `guardrails` | - | 安全防护配置 |
| `channels` | - | IM 频道配置 |

### 7.2 扩展配置 (`extensions_config.json`)

**位置**: 项目根目录

| 章节 | 配置项 | 说明 |
|------|--------|------|
| `mcpServers` | - | MCP 服务器配置 |
| `skills` | - | 技能启用状态 |

### 7.3 配置版本管理

`config.example.yaml` 包含 `config_version` 字段。启动时 `AppConfig.from_file()` 比较用户版本与示例版本，不一致时发出警告。

---

## 八、关键入口文件

### 8.1 后端入口

| 功能 | 入口文件 | 说明 |
|------|----------|------|
| **Gateway API 主入口** | `app/gateway/app.py` | FastAPI 应用 |
| **Gateway 路由** | `app/gateway/routers/` | REST API 路由定义 |
| **Agent 创建工厂** | `packages/harness/deerflow/agents/factory.py` | `make_lead_agent()` |
| **Lead Agent 实现** | `packages/harness/deerflow/agents/lead_agent/agent.py` | 主 Agent 逻辑 |
| **LangGraph Server** | `backend/langgraph.json` | 配置，通过 `langgraph dev` 启动 |
| **沙箱系统入口** | `packages/harness/deerflow/sandbox/sandbox.py` | 抽象接口 |
| **工具系统入口** | `packages/harness/deerflow/tools/tools.py` | `get_available_tools()` |
| **子 Agent 执行器** | `packages/harness/deerflow/subagents/executor.py` | 后台执行 |
| **IM 频道服务** | `app/channels/service.py` | `start_channel_service()` |
| **嵌入式客户端** | `packages/harness/deerflow/client.py` | `DeerFlowClient` |

### 8.2 前端入口

| 功能 | 入口文件 | 说明 |
|------|----------|------|
| **根布局** | `frontend/src/app/layout.tsx` | Next.js 根布局 |
| **落地页** | `frontend/src/app/page.tsx` | 首页 |
| **工作区** | `frontend/src/app/workspace/page.tsx` | 工作区页面 |
| **聊天页面** | `frontend/src/app/workspace/agents/[thread_id]/page.tsx` | 线程详情 |
| **API 客户端** | `frontend/src/core/api/index.ts` | LangGraph SDK 封装 |
| **线程 Hooks** | `frontend/src/core/threads/hooks.ts` | `useThreadStream` 等 |
| **认证** | `frontend/src/server/better-auth/` | Better Auth 实现 |

### 8.3 部署脚本

| 文件 | 功能 |
|------|------|
| `scripts/serve.sh` | 本地服务管理 |
| `scripts/docker.sh` | Docker 开发环境 |
| `scripts/deploy.sh` | Docker 生产部署 |
| `Makefile` | 项目根目录命令 |

---

## 九、架构设计要点

### 9.1 Harness/App 分离

```
packages/harness/deerflow/  ← 可发布为 pip 包
        │
        ▼ 依赖
app/  ← 应用代码，导入 harness 但不反向依赖
```

**边界强制**: `tests/test_harness_boundary.py` 确保 harness 模块不导入 app.*。

### 9.2 中间件链模式

Agent 执行经过 18 个中间件，每个职责单一：
- 前置处理: 线程数据、上传文件、沙箱获取
- 错误处理: LLM 错误、工具错误、循环检测
- 后置处理: 记忆更新、Token 统计、标题生成
- 特殊拦截: 澄清请求、Guardrail 检查

### 9.3 虚拟路径隔离

沙箱中的路径与物理路径解耦，提供：
- 安全性: Agent 无法直接访问宿主机路径
- 隔离性: 每个线程有独立的目录空间
- 一致性: 不同沙箱类型提供统一的路径视图

### 9.4 惰性初始化

| 组件 | 初始化时机 |
|------|-----------|
| MCP 工具 | 首次使用时加载，基于 mtime 缓存失效 |
| 模型 | 首次创建 Agent 时实例化 |
| 沙箱 | 首次工具调用时获取 |
| 技能 | Agent 初始化时加载 |

### 9.5 线程隔离

每个对话线程拥有：
- 独立的沙箱实例
- 独立的目录结构 (`backend/.deer-flow/threads/{thread_id}/`)
- 独立的记忆数据
- 独立的检查点状态

### 9.6 流式响应架构

```
LangGraph stream_mode
       │
       ▼
┌─────────────────────────────────────────────────┐
│  values    — 完整状态快照                        │
│  messages-tuple — 逐块更新 (AI 文本增量)         │
│  custom    — 自定义事件                          │
│  end       — 流结束                             │
└─────────────────────────────────────────────────┘
```

### 9.7 社区工具集成

社区工具通过标准接口集成：
- 搜索工具: Tavily, Jina AI, DuckDuckGo
- 爬虫工具: Firecrawl
- 所有工具遵循 `BaseTool` 接口

---

## 附录：技术栈汇总

### 后端依赖

| 包 | 版本 | 用途 |
|---|------|------|
| langgraph | 1.0.6-1.0.10 | Agent 框架 |
| langchain | 1.2.3+ | LLM 集成 |
| fastapi | - | REST API |
| langchain-openai | 1.1.7+ | OpenAI 模型 |
| langchain-anthropic | 1.3.4+ | Anthropic 模型 |
| langchain-mcp-adapters | 0.1.0+ | MCP 集成 |
| langfuse | 3.4.1+ | 可观测性 |

### 前端依赖

| 包 | 版本 | 用途 |
|---|------|------|
| next | 16.1.7 | React 框架 |
| react | 19.0.0 | UI 库 |
| typescript | 5.8.2 | 类型系统 |
| @langchain/langgraph-sdk | 1.5.3+ | 后端通信 |
| @tanstack/react-query | 5.90.17+ | 状态管理 |
| ai | 6.0.33+ | AI 流式响应 |
| better-auth | 1.3+ | 认证 |

---

*本文档由源码分析生成，如有不准确之处请以实际代码为准。*
