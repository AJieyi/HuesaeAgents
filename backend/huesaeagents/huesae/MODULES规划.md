# Huesae 模块规划

## 一、顶层目录结构

```
backend/huesaeagents/huesae/
├── __init__.py
├── models/                  # LLM模型（简化版，支持Anthropic/OpenAI）
├── agents/                 # Agent系统（核心）
├── config/                 # 配置系统（简化版）
├── tools/                  # 工具系统（简化版）
├── subagents/              # 子Agent系统（简化版，最多3个）
├── memory/                 # 记忆系统（简化版）
└── runtime/                # 运行时（简化版，仅SSE）
```

---

## 二、各模块说明

### 2.1 models/
**功能**：LLM模型封装
**参考**：deerflow `models/`（简化）

| 文件 | 说明 |
|------|------|
| `factory.py` | 模型工厂 `create_chat_model()` |
| `providers/deepseek.py` |  |
|  |  |

---

### 2.2 agents/
**功能**：Agent核心系统
**参考**：deerflow `agents/`（简化）

| 文件/目录 | 说明 |
|------|------|
| `lead_agent/` | 主Agent（factory + graph + prompts） |
| `state.py` | ThreadState状态定义 |
| `character/` | 角色管理（manager + loader） |
| `middleware/` | 中间件（精简到3个） |

**中间件精简**：deerflow有14+个 → 我们只需3个
- `emotion.py` - 情绪感知
- `guardrail.py` - 安全护栏
- `character.py` - 角色切换

**去除**：checkpointer, DanglingToolCall, LoopDetection, SubagentLimit 等

---

### 2.3 config/
**功能**：配置管理
**参考**：deerflow `config/`（简化）

| 文件 | 说明 |
|------|------|
| `app_config.py` | 主配置类 |
| `paths.py` | 路径解析 |

**去除**：sandbox_config, extensions_config, memory_config（这些功能简化或后续添加）

---

### 2.4 tools/
**功能**：工具系统
**参考**：deerflow `tools/`（简化）

| 文件/目录 | 说明 |
|------|------|
| `registry.py` | 工具注册表 |
| `builtins/` | 内置工具 |
| `present_files.py` | 文件展示 |
| `ask.py` | 询问用户 |
| `task.py` | 任务委托 |

**去除**：tool_search, skill_manage, invoke_acp_agent, MCP集成

---

### 2.5 subagents/
**功能**：子Agent系统
**参考**：deerflow `subagents/`（简化）

| 文件 | 说明 |
|------|------|
| `executor.py` | 子Agent执行器 |
| `registry.py` | Agent注册表 |

**限制**：最多3个子Agent并发

**去除**：复杂的隔离执行机制

---

### 2.6 memory/
**功能**：记忆系统
**参考**：deerflow `agents/memory/`（简化）

| 文件 | 说明 |
|------|------|
| `storage.py` | 文件存储 |
| `manager.py` | 记忆管理器 |

**去除**：updater.py的复杂去重逻辑，queue.py的防抖机制（简化实现）

---

### 2.7 runtime/
**功能**：运行时
**参考**：deerflow `runtime/`（简化）

| 文件 | 说明 |
|------|------|
| `stream_bridge.py` | SSE流式桥接 |
| `runs/manager.py` | 运行记录（简化） |

**去除**：store/, journal/, 复杂的callback机制

---

## 三、去除的deerflow模块

| 模块 | 原因 |
|------|------|
| `sandbox/` | 不需要沙箱隔离 |
| `community/` | 不需要 Tavily/Jina/Firecrawl 等 |
| `mcp/` | 不需要MCP协议支持 |
| `skills/` | 不需要技能系统 |
| `reflection/` | 不需要反射加载 |
| `tracing/` | 不需要分布式追踪 |
| `uploads/` | 文件上传后续添加 |
| `guardrails/` | 简化为middleware内 |

---

## 四、模块依赖关系

```
models/           # 无依赖，最底层
    ↓
config/           # 依赖models
    ↓
agents/           # 依赖models, config
    ↓
tools/            # 依赖agents
    ↓
subagents/        # 依赖agents, tools
    ↓
memory/           # 依赖agents
    ↓
runtime/          # 依赖以上所有
```

---

## 五、搭建顺序

1. **models/** - LLM基础设施
2. **config/** - 配置系统
3. **agents/** - Agent核心（state → middleware → prompts → graph → factory）
4. **tools/** - 工具系统
5. **subagents/** - 子Agent
6. **memory/** - 记忆系统
7. **runtime/** - 运行时