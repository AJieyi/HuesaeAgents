# AIGC多智能体系统

个人开发中ing

面向用户的**多智能体AIGC**。帮助用户生图，漫画、漫剧脚本生成以及完成通用任务。



##  v0.0.1 版本说明 

### ✨ 核心特性

#### 🏗️ **技术架构**

- **技术路线**: LangChain + LangGraph + MCP + Skills + Agent

## 系统架构
| #    | 功能模块          | 主要目录 / 核心文件                                          | 职责简述                                                     |
| ---- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | **主Agent层**     | [agents/lead_agent/](backend/huesaeagents/huesae/agents/lead_agent/)（`lead_agent.py`、`chat_loop.py`、`prompts.py`） | 用户主入口、ReAct 调度循环、子Agent 委派、安全检查、终端交互 |
| 2    | **子Agent层**     | [subagents/](backend/huesaeagents/huesae/subagents/)（`base.py`、`registry.py`、`image_agent.py`、`general_agent.py`、`image/`） | 专业能力封装：生图多轮对话、通用任务执行；统一注册接口       |
| 3    | **Agent 运行时**  | [agents/middlewares/](backend/huesaeagents/huesae/agents/middlewares/)、[agents/thread_state.py](backend/huesaeagents/huesae/agents/thread_state.py)、[agents/model_adapter.py](backend/huesaeagents/huesae/agents/model_adapter.py) | LangGraph 状态结构、中间件管道（运行时工具/Token 用量）、模型适配 |
| 4    | **工具运行时**    | [tools/](backend/huesaeagents/huesae/tools/)（`runtime.py`、`tools.py`、`doubao/`、`jimeng/`） | 内置工具定义、共享工具池（SharedToolRuntime）、第三方生图 API 客户端 |
| 5    | **MCP 扩展**      | [mcp/](backend/huesaeagents/huesae/mcp/)（`client.py`、`tools.py`、`cache.py`） | 加载/缓存外部 MCP server 暴露的工具（B 站、抖音、视频脚本等） |
| 6    | **Skills 子系统** | [skills/](backend/huesaeagents/huesae/skills/) + 仓库根目录 [skills/](skills/)（comic-create、polecomic、weather） | Skill 资源扫描、注册、按需读取注入提示词                     |
| 7    | **模型层**        | [models/](backend/huesaeagents/huesae/models/)（`models_factory.py`、`providers/deepseek.py`、`providers/doubao_vision.py`） | LLM 与多模态视觉模型工厂                                     |
| 8    | **业务服务**      | [services/](backend/huesaeagents/huesae/services/)（`memory.py`、`vision.py`） | Honcho 长期记忆服务、图片反推（识图）服务                    |
| ·    | **配置中心**      | [config/](backend/huesaeagents/huesae/config/)（`extensions_config.py`、`middleware_config.py`）+ 根 `.env`、`extensions_config.json` | 集中加载 .env、MCP 扩展配置、中间件开关                      |
</div>
