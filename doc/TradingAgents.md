# TradingAgents-CN 系统架构设计文档

## 一、项目整体定位

这是一个**多智能体量化股票分析学习平台**，采用前后端分离架构：
- **开源核心**：`tradingagents/` - Apache 2.0 协议，多智能体框架
- **专有后端**：`app/` - FastAPI 后端（需商业授权）
- **专有前端**：`frontend/` - Vue 3 前端（需商业授权）

---

## 二、系统模块划分

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TradingAgents-CN                              │
├──────────────────┬──────────────────┬───────────────────────────────┤
│   CLI 入口       │   后端 API        │        前端 Vue 3             │
│   (main.py)     │   (app/)         │        (frontend/)            │
├──────────────────┴──────────────────┴───────────────────────────────┤
│                     核心框架 (tradingagents/)                        │
│  ┌─────────────┬──────────────┬────────────┬─────────────────┐    │
│  │ Agents      │ Graph        │ LLM适配器   │ DataFlows       │    │
│  │ 智能体       │ 工作流编排    │ 多LLM支持   │ 数据流处理      │    │
│  └─────────────┴──────────────┴────────────┴─────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                     外部服务依赖                                      │
│  ┌─────────────┬──────────────┬────────────┬─────────────────┐    │
│  │ MongoDB     │ Redis        │ LLM提供商   │ 数据源           │    │
│  │ 数据存储     │ 缓存/会话    │ OpenAI等   │ AKShare/Tushare │    │
│  └─────────────┴──────────────┴────────────┴─────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、功能模块与代码目录映射

### 3.1 核心框架 `tradingagents/` (开源)

| 功能模块 | 代码目录/文件 | 功能说明 |
|---------|--------------|---------|
| **智能体** | `tradingagents/agents/` | 各种AI智能体实现 |
| └ 市场分析师 | `agents/analysts/market_analyst.py` | 技术分析、趋势判断 |
| └ 基本面分析师 | `agents/analysts/fundamentals_analyst.py` | 财务数据分析 |
| └ 新闻分析师 | `agents/analysts/news_analyst.py` | 新闻情感分析 |
| └ 多头研究员 | `agents/researchers/bull_researcher.py` | 看多观点研究 |
| └ 空头研究员 | `agents/researchers/bear_researcher.py` | 看空观点研究 |
| └ 交易员 | `agents/trader/trader.py` | 交易决策 |
| └ 风险管理器 | `agents/risk_mgmt/risk_manager.py` | 风险评估 |
| **工作流编排** | `tradingagents/graph/` | LangGraph 工作流 |
| └ 核心图 | `graph/trading_graph.py` | 主工作流（67KB），编排所有智能体 |
| └ 条件逻辑 | `graph/conditional_logic.py` | 决策分支逻辑 |
| └ 信号处理 | `graph/signal_processing.py` | 信号处理与传播 |
| └ 反思机制 | `graph/reflection.py` | 反思与记忆 |
| **LLM适配器** | `tradingagents/llm_adapters/` | 多LLM统一接口 |
| └ Google适配器 | `llm_adapters/google_openai_adapter.py` | Gemini |
| └ DeepSeek适配器 | `llm_adapters/deepseek_adapter.py` | DeepSeek |
| └ 阿里百炼适配器 | `llm_adapters/dashscope_openai_adapter.py` | 通义千问 |
| └ OpenAI兼容基类 | `llm_adapters/openai_compatible_base.py` | OpenAI兼容接口 |
| **数据流** | `tradingagents/dataflows/` | 数据处理管道 |
| └ 数据源管理 | `dataflows/data_source_manager.py` | 多数据源统一管理 |
| └ 数据接口 | `dataflows/interface.py` | 数据流抽象接口 |
| └ 实时指标 | `dataflows/realtime_metrics.py` | 实时行情指标 |
| └ 数据源 providers | `dataflows/providers/` | 各市场数据源 |
| └ 中国市场 | `providers/china/` | A股数据（akshare/tushare/baostock） |
| └ 港股 | `providers/hk/` | 港股数据 |
| └ 美股 | `providers/us/` | 美股数据 |
| **工具** | `tradingagents/tools/` | 智能体共享工具 |
| └ 统一新闻工具 | `tools/unified_news_tool.py` | 新闻获取与分析 |
| **配置** | `tradingagents/config/` | 配置管理 |
| **常量** | `tradingagents/constants/` | 数据源常量定义 |
| **模型** | `tradingagents/models/` | 数据模型定义 |

---

### 3.2 后端服务 `app/` (专有)

| 功能模块 | 代码目录/文件 | 功能说明 |
|---------|--------------|---------|
| **核心配置** | `app/core/` | 核心基础设施 |
| └ 配置管理 | `core/config.py` | 环境配置加载 |
| └ 数据库连接 | `core/database.py` | MongoDB 连接管理 |
| └ Redis客户端 | `core/redis_client.py` | Redis 缓存客户端 |
| └ 日志配置 | `core/logging_config.py` | 日志系统配置 |
| └ 限流器 | `core/rate_limiter.py` | API 限流 |
| └ 缓存 | `core/cache/` | 多级缓存实现 |
| **API路由** | `app/routers/` | RESTful API 端点 |
| └ 分析路由 | `routers/analysis.py` | 股票分析 API |
| └ 认证路由 | `routers/auth_db.py` | 用户认证 API |
| └ 股票路由 | `routers/stocks.py` | 股票数据 API |
| └ 筛选路由 | `routers/screening.py` | 股票筛选 API |
| └ 收藏路由 | `routers/favorites.py` | 自选股管理 API |
| └ 配置路由 | `routers/config.py` | 系统配置 API |
| └ 报告路由 | `routers/reports.py` | 报告导出 API |
| └ 调度路由 | `routers/scheduler.py` | 任务调度 API |
| └ 健康检查 | `routers/health.py` | 健康检查 |
| └ 通知路由 | `routers/notifications.py` | SSE/WebSocket 通知 |
| └ 数据同步 | `routers/sync.py` | 数据同步 API |
| └ 模拟交易 | `routers/paper.py` | 模拟交易 API |
| **业务服务** | `app/services/` | 业务逻辑层 |
| └ 分析服务 | `services/analysis_service.py` | 分析业务逻辑 |
| └ 配置服务 | `services/config_service.py` | 190KB，配置管理核心 |
| └ 数据服务 | `services/stock_data_service.py` | 股票数据服务 |
| └ 收藏服务 | `services/favorites_service.py` | 收藏业务逻辑 |
| └ 新闻服务 | `services/news_data_service.py` | 新闻数据处理 |
| └ 财务服务 | `services/financial_data_service.py` | 财务数据 |
| └ 同步服务 | `services/basics_sync_service.py` | 基础数据同步 |
| └ 增强筛选 | `services/enhanced_screening_service.py` | 高级筛选 |
| └ 模型能力 | `services/model_capability_service.py` | LLM模型能力管理 |
| **数据模型** | `app/models/` | Pydantic 数据模型 |
| **中间件** | `app/middleware/` | 中间件（日志记录等） |
| **后台 Worker** | `app/worker/` | 异步任务 workers |
| └ Tushare同步 | `worker/tushare_sync_service.py` | Tushare数据同步 |
| └ AKShare同步 | `worker/akshare_sync_service.py` | AKShare数据同步 |
| └ BaoStock同步 | `worker/baostock_sync_service.py` | BaoStock数据同步 |

---

### 3.3 前端 `frontend/` (专有)

| 功能模块 | 代码目录/文件 | 功能说明 |
|---------|--------------|---------|
| **入口** | `frontend/src/main.ts` | Vue 应用入口 |
| **视图** | `frontend/src/views/` | 页面组件 |
| └ 认证视图 | `views/Auth/` | 登录/注册 |
| └ 首页/仪表盘 | `views/Dashboard/` | 首页概览 |
| └ 分析视图 | `views/Analysis/` | 股票分析页面 |
| └ 股票视图 | `views/Stocks/` | 个股详情 |
| └ 自选视图 | `views/Favorites/` | 自选股管理 |
| └ 筛选视图 | `views/Screening/` | 股票筛选 |
| └ 报告视图 | `views/Reports/` | 报告查看/导出 |
| └ 设置视图 | `views/Settings/` | 系统设置 |
| **API客户端** | `frontend/src/api/` | 后端API调用 |
| └ analysis.ts | `api/analysis.ts` | 分析相关API |
| └ auth.ts | `api/auth.ts` | 认证相关API |
| └ config.ts | `api/config.ts` | 配置相关API |
| └ stocks.ts | `api/stocks.ts` | 股票数据API |
| └ screening.ts | `api/screening.ts` | 筛选API |
| └ request.ts | `api/request.ts` | HTTP 请求封装 |
| **状态管理** | `frontend/src/stores/` | Pinia 状态管理 |
| └ auth store | `stores/auth.ts` | 认证状态 |
| └ app store | `stores/app.ts` | 全局应用状态 |
| **路由** | `frontend/src/router/` | Vue Router 配置 |
| **组件库** | `frontend/src/components/` | 可复用组件 |
| **类型定义** | `frontend/src/types/` | TypeScript 类型 |
| **样式** | `frontend/src/styles/` | SCSS 样式文件 |

---

### 3.4 CLI 工具 `cli/`

| 模块 | 文件 | 功能 |
|-----|------|------|
| CLI主入口 | `cli/main.py` | 84KB，命令行接口 |
| AKShare初始化 | `cli/akshare_init.py` | AKShare数据初始化 |
| BaoStock初始化 | `cli/baostock_init.py` | BaoStock初始化 |
| Tushare初始化 | `cli/tushare_init.py` | Tushare初始化 |
| 工具函数 | `cli/utils.py` | CLI通用工具 |

---

## 四、入口文件与集成方式

### 4.1 系统入口点

```
┌─────────────────────────────────────────────────────────────────┐
│                         入口点架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │  main.py     │     │ app/__main__.py│   │ frontend/    │   │
│  │  (Legacy CLI)│     │  (Backend)    │   │ src/main.ts  │   │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘   │
│         │                    │                     │            │
│         ▼                    ▼                     ▼            │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │TradingAgents │     │  FastAPI     │     │  Vue 3 App   │   │
│  │  Graph       │     │  Application │     │  (SPA)       │   │
│  └──────────────┘     └──────┬───────┘     └──────┬───────┘   │
│                              │                     │            │
│                              ▼                     │            │
│                     ┌──────────────┐              │            │
│                     │   Routers    │              │            │
│                     │   (API层)    │              │            │
│                     └──────┬───────┘              │            │
│                            │                      │            │
│                            ▼                      │            │
│                     ┌──────────────┐              │            │
│                     │  Services    │              │            │
│                     │  (业务逻辑)   │◄─────────────┘            │
│                     └──────┬───────┘     (HTTP/REST)           │
│                            │                                │
│         ┌──────────────────┼──────────────────┐              │
│         ▼                  ▼                  ▼              │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐         │
│  │  MongoDB   │    │   Redis    │    │  trading   │         │
│  │  (主存储)  │    │  (缓存)    │    │  agents/   │         │
│  └────────────┘    └────────────┘    └────────────┘         │
│                                             (Core Framework)  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 核心框架调用链

```
tradingagents/trading_graph.py (核心工作流)
         │
         ├── agents/analysts/     (分析师智能体)
         │       ├── market_analyst.py
         │       ├── fundamentals_analyst.py
         │       ├── news_analyst.py
         │       └── china_market_analyst.py
         │
         ├── agents/researchers/  (研究员智能体)
         │       ├── bull_researcher.py
         │       └── bear_researcher.py
         │
         ├── agents/trader/       (交易员智能体)
         │       └── trader.py
         │
         ├── agents/risk_mgmt/    (风险管理器)
         │       └── risk_manager.py
         │
         ├── agents/managers/     (管理智能体)
         │       ├── research_manager.py
         │       └── risk_manager.py
         │
         ├── llm_adapters/        (LLM适配)
         │       ├── google_openai_adapter.py
         │       ├── deepseek_adapter.py
         │       └── dashscope_openai_adapter.py
         │
         └── dataflows/           (数据流)
                 ├── data_source_manager.py
                 ├── interface.py
                 └── providers/
                         ├── china/ (AKShare/Tushare/BaoStock)
                         ├── hk/
                         └── us/
```

---

## 五、服务依赖关系

### 5.1 外部服务依赖

```
┌──────────────────────────────────────────────────────────────────┐
│                      外部服务集成                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  MongoDB    │  │   Redis     │  │  LLM APIs   │              │
│  │  localhost  │  │  localhost  │  │  (多提供商) │              │
│  │  :27017     │  │  :6379      │  │             │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                     app/core/                           │     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │     │
│  │  │ database.py│  │redis_client │  │config.py    │     │     │
│  │  │ (MongoDB)  │  │  (Redis)   │  │ (配置加载)  │     │     │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │     │
│  └─────────────────────────────────────────────────────────┘     │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                   app/services/                          │     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │     │
│  │  │ analysis_   │  │stock_data_ │  │ news_data_  │     │     │
│  │  │ service    │  │ service    │  │ service     │     │     │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │     │
│  └─────────────────────────────────────────────────────────┘     │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │               tradingagents/ (Core Framework)           │     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │     │
│  │  │ agents/    │  │ llm_       │  │ dataflows/  │     │     │
│  │  │ (智能体)   │  │adapters/   │  │ (数据流)    │     │     │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                   │
│  数据源:                                                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ AKShare  │ │ Tushare  │ │ BaoStock │ │  yfinance│            │
│  │ (免费)   │ │ (专业)   │ │ (免费)   │ │ (美股)   │            │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
│                                                                   │
│  LLM提供商:                                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ OpenAI   │ │ Google   │ │ DeepSeek │ │ 阿里百炼  │            │
│  │ GPT-4    │ │ Gemini   │ │ DeepSeek │ │ 通义千问  │            │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 模块间调用关系

```
┌─────────────────────────────────────────────────────────────────┐
│                     模块调用层次关系                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  【表现层】frontend/src/views/                                    │
│       │                                                          │
│       ▼ (HTTP API)                                               │
│  【API层】app/routers/                                           │
│       │                                                          │
│       ▼ (Service调用)                                            │
│  【业务层】app/services/                                         │
│       │                                                          │
│       ├──► tradingagents/ (可选，核心框架集成)                     │
│       │     ├── agents/                                          │
│       │     ├── graph/trading_graph.py                           │
│       │     └── llm_adapters/                                    │
│       │                                                          │
│       ├──► app/core/ (基础设施)                                   │
│       │     ├── database.py (MongoDB)                            │
│       │     ├── redis_client.py (Redis)                         │
│       │     └── config.py                                       │
│       │                                                          │
│       └──► dataflows/providers/ (数据源)                         │
│             ├── china/ (AKShare/Tushare/BaoStock)               │
│             ├── hk/                                              │
│             └── us/                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 六、配置管理体系

### 6.1 配置来源

| 配置来源 | 文件 | 说明 |
|---------|------|------|
| **环境变量** | `.env` | MongoDB/Redis/JWT/API密钥等 |
| **配置文件** | `config/*.json` | models.json, settings.json, pricing.json |
| **代码默认** | `tradingagents/default_config.py` | 框架默认配置 |
| **后端配置** | `app/core/config.py` | 后端配置加载 |

### 6.2 环境变量分类

```
# 数据库
MONGODB_HOST/PORT/USERNAME/PASSWORD/DATABASE
REDIS_HOST/PORT/PASSWORD/DB

# 安全
JWT_SECRET/JWT_ALGORITHM
CSRF_SECRET

# LLM API
DEEPSEEK_API_KEY / DASHSCOPE_API_KEY
OPENAI_API_KEY / GOOGLE_AI_API_KEY

# 数据源
DEFAULT_CHINA_DATA_SOURCE (akshare/tushare/baostock)
TUSHARE_TOKEN
```

---

## 七、Docker 部署架构

```
┌─────────────────────────────────────────────────────────────────┐
│                   Docker Compose 架构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              tradingagents-network (bridge)              │    │
│  │                                                          │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │    │
│  │  │  backend    │  │  frontend   │  │   mongodb   │      │    │
│  │  │  :8000      │  │   :3000     │  │   :27017    │      │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │    │
│  │         │                │                │              │    │
│  │         │                │                │              │    │
│  │         └────────────────┴────────────────┘              │    │
│  │                          │                               │    │
│  │                          ▼                               │    │
│  │                 ┌─────────────┐                          │    │
│  │                 │    redis    │                          │    │
│  │                 │    :6379    │                          │    │
│  │                 └─────────────┘                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  端口映射:                                                        │
│  - backend: 8000:8000                                           │
│  - frontend: 3000:80                                             │
│  - mongodb: 27017:27017                                         │
│  - redis: 6379:6379                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 八、系统架构总结

### 8.1 技术栈

| 层级 | 技术选型 |
|-----|---------|
| **前端** | Vue 3 + Vite + Element Plus + TypeScript + Pinia |
| **后端** | FastAPI + Uvicorn + Pydantic |
| **数据库** | MongoDB (主存储) + Redis (缓存) |
| **任务调度** | APScheduler |
| **核心框架** | LangGraph + LangChain |
| **LLM集成** | OpenAI / Google / DeepSeek / DashScope |
| **数据源** | AKShare / Tushare / BaoStock / yfinance |
| **部署** | Docker + Docker Compose |

### 8.2 核心设计模式

1. **前后端分离**：RESTful API + SPA
2. **多智能体协作**：LangGraph 工作流编排
3. **多LLM适配器**：统一接口支持多提供商
4. **多数据源统一管理**：抽象数据接口，灵活切换
5. **多层缓存**：Redis + MongoDB + 文件缓存
6. **异步任务队列**：APScheduler 后台任务

### 8.3 许可证架构

```
┌────────────────────────────────────────┐
│         Apache 2.0 (开源)               │
│  ┌──────────────────────────────────┐  │
│  │  tradingagents/ (核心框架)       │  │
│  │  cli/ (命令行工具)               │  │
│  │  examples/ (示例)                │  │
│  │  docs/ (文档)                    │  │
│  │  web/ (Streamlit旧版)           │  │
│  └──────────────────────────────────┘  │
├────────────────────────────────────────┤
│         专有软件 (需授权)              │
│  ┌──────────────────────────────────┐  │
│  │  app/ (FastAPI后端)              │  │
│  │  frontend/ (Vue 3前端)            │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```
