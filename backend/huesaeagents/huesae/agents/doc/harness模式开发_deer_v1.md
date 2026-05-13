# 开发计划：向 DeerFlow Harness Engineering 模式演进

## Context

当前 HuesaeAgents 采用"主Agent硬编码意图分类 + 子Agent委派"模式：
- 主Agent通过 `_classify_intent()` 用LLM判断用户意图（chat/image/voice等）
- 然后硬编码路由到对应子Agent
- 子Agent内部再有自己的决策逻辑

**问题**：意图分类是硬编码的，新增子Agent需要同时修改分类prompt和路由逻辑。与 DeerFlow "LLM自主工具选择"模式相比，扩展性不足。

**DeerFlow 核心模式**：系统只提供工具列表和描述，工具选择完全由 LLM 决定。LLM 看到工具列表后，自主决策调用哪个工具（包括 `task` 子Agent委托工具）。

## 目标

将主Agent从"硬编码意图分类"改造为"LLM 自主工具选择"模式：
1. 主Agent给 LLM 提供工具列表 + 系统提示词
2. LLM 自主决策：直接回复 / 调用工具 / 委托子Agent
3. 新增子Agent = 新增工具，无需修改主Agent分类逻辑
4. 保留已测试成功的生图功能

## 方案选择

### 方案A：手写 ReAct 循环（推荐）

在现有 `HuesaeMainAgent` 基础上改造 `process()` 为 ReAct 循环：
- 给 LLM 提供工具列表（schema描述）
- LLM 输出：直接回复 或 工具调用请求
- 执行工具，结果返回 LLM
- LLM 再次推理，直到输出最终回复

**优点**：与现有架构兼容性好，改动最小，可逐步引入 DeerFlow 特性
**缺点**：不如 LangGraph 声明式 Agent 标准化

### 方案B：LangGraph create_react_agent

使用 LangGraph 原生的 `create_react_agent()` 构建主Agent。

**优点**：标准化，生态丰富
**缺点**：与当前手写架构差异大，需要大幅重构状态管理和交互层

**推荐方案A**，原因：
1. 当前架构已经是手写的，方案A改动最小
2. 可以保留 `ImageSubAgent` 作为被调用的组件
3. 后续可逐步引入中间件、ThreadState 等 DeerFlow 特性

## 实施方案

### 步骤1：工具系统构建

**文件：** `backend/huesaeagents/huesae/agents/tools.py`（新建）

将现有功能封装为 LangChain Tool：

```python
@tool
def chat_tool(user_input: str, state: dict) -> str:
    """与用户进行日常聊天对话。当用户问天气、问候、闲聊、表达感谢时使用此工具。"""
    ...

@tool
def generate_image_tool(prompt: str, size: str = "2K", output_format: str = "jpeg") -> str:
    """生成单张图片。当用户要求生成1张图片、画画、绘图时使用此工具。"""
    ...

@tool
def generate_images_tool(prompt: str, size: str = "2K", output_format: str = "jpeg") -> str:
    """生成一组连贯图片（组图）。当用户明确要求生成多张图片（如'生成4张'）时使用此工具。"""
    ...

@tool
def expand_prompt_tool(prompt: str) -> str:
    """扩写图片提示词。当用户要求扩写、丰富图片描述时使用此工具。"""
    ...

@tool
def convert_tags_tool(prompt: str) -> str:
    """将自然语言描述转换为Danbooru标签。当用户要求生成标签时使用此工具。"""
    ...

@tool
def task_tool(description: str, subagent_type: str) -> str:
    """委托子Agent处理复杂任务。当任务需要多步骤、专业处理时使用此工具。
    支持的子Agent类型：image（生图对话Agent）
    """
    ...
```

### 步骤2：主Agent改造为 ReAct 模式

**文件：** `backend/huesaeagents/huesae/agents/lead_agent/lead_agent.py`

核心改造：`process()` 方法改为 ReAct 循环：

```python
def process(self, state: dict, user_input: str) -> dict:
    # 1. 安全检查
    if self._check_safety(user_input): ...

    # 2. ReAct 循环
    messages = state.get("messages", [])
    tool_results = []

    for step in range(MAX_STEPS):  # 最多3轮工具调用
        # 构建系统提示词（含工具列表描述）
        system_prompt = self._build_system_prompt_with_tools()

        # 调用 LLM，传入历史 + 工具描述
        llm_messages = [SystemMessage(content=system_prompt)] + messages + tool_results
        response = self.llm.invoke(llm_messages)

        # 解析 LLM 输出
        action = self._parse_action(response.content)

        if action.type == "reply":
            # 直接回复
            return {"messages": [AIMessage(content=action.content)]}

        elif action.type == "tool_call":
            # 执行工具
            result = self._execute_tool(action.tool_name, action.args, state)
            tool_results.append(AIMessage(content=f"工具 {action.tool_name} 执行结果：{result}"))
            # 继续循环，让 LLM 基于工具结果再次推理

    # Fallback：超过最大步数，直接回复
    return {"messages": [AIMessage(content="抱歉，处理有点复杂，让我直接帮您...")]}
```

**删除**：`_classify_intent()`、`_handle_sub_agent()` 硬编码路由逻辑

**新增**：
- `_build_system_prompt_with_tools()`：构建含工具描述的系统提示词
- `_parse_action()`：解析 LLM 输出为 action（reply / tool_call）
- `_execute_tool()`：执行指定工具
- `MAX_STEPS = 3`：限制工具调用轮数，防止循环

### 步骤3：保留 ImageSubAgent 作为可委托组件

**文件：** `backend/huesaeagents/huesae/agents/subagents/image_agent.py`（不变）

`ImageSubAgent` 保留不变，作为 `task_tool(subagent_type="image")` 的目标。

当 LLM 调用 `task_tool(description="生成图片", subagent_type="image")` 时：
1. 主Agent创建 `ImageSubAgent` 实例
2. 将 `description` 作为用户输入传给 `ImageSubAgent.process()`
3. `ImageSubAgent` 内部执行多轮对话（追问 → 确认 → 生图）
4. 最终结果返回给主Agent

**注意**：`task_tool` 内部可能需要维护子Agent的对话状态（类似当前 `image_context` 的隔离机制）。

### 步骤4：交互层适配

**文件：** `backend/huesaeagents/huesae/agents/lead_agent/chat_loop.py`

当前 `chat_loop.py` 中，主Agent返回 `pending_generation` 标志，由 chat_loop 异步执行生图。

改造后：
- 工具执行（包括生图）在 `process()` 内部的 ReAct 循环中完成
- `chat_loop` 只需要打印 `process()` 返回的最终 `messages`
- 如果工具执行耗时较长（如生图），可以在 `_execute_tool()` 内部异步执行

**简化方案**：由于生图是异步的，可以在 `_execute_tool()` 中直接调用 `asyncio.run()` 执行，结果返回给 LLM。

### 步骤5：Prompt 调整

**文件：** `backend/huesaeagents/huesae/agents/subagents/image/prompts.py`

新增或调整：
- 主Agent系统提示词（含工具列表描述和 ReAct 格式说明）
- LLM 输出格式：必须遵循特定 JSON/文本格式，便于 `_parse_action()` 解析

## 关键设计决策

1. **工具 vs 子Agent**：
   - 简单功能（单图生图、扩写、标签转换）→ 直接封装为工具
   - 复杂多轮对话（生图流程：追问→确认→生图）→ 通过 `task_tool` 委托 `ImageSubAgent`

2. **LLM 输出格式**：
   - 使用结构化输出（Pydantic model + `with_structured_output`）要求 LLM 输出 action 选择
   - 或者使用文本格式（如 "TOOL: generate_image | prompt: xxx"）然后正则解析
   - **推荐结构化输出**，更可靠

3. **状态管理**：
   - 当前 `HuesaeState` 继续保留（内存版）
   - 工具调用历史作为 `messages` 的一部分传递
   - 后续可演进为 DeerFlow 的 `ThreadState`

## 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `agents/tools.py` | 工具定义（chat/generate_image/generate_images/expand_prompt/convert_tags/task） |
| 大幅改造 | `agents/lead_agent/lead_agent.py` | 去掉意图分类，改为 ReAct 循环 |
| 改造 | `agents/lead_agent/chat_loop.py` | 适配新的 `process()` 接口 |
| 新建/调整 | `agents/subagents/image/prompts.py` | 主Agent系统提示词（含工具列表） |
| 保留 | `agents/subagents/image_agent.py` | 作为 task_tool 的 subagent_type="image" 目标 |
| 保留 | `tools/doubao/client.py` | 豆包客户端不变 |

## 验证方案

运行 `python backend/huesaeagents/huesae/agents/lead_agent/chat_loop.py`：

1. 用户："今天天气如何" → LLM 选择 chat_tool → 直接聊天回复
2. 用户："生成一张夕阳下的大海" → LLM 选择 generate_image_tool → 调用生图 → 返回图片URL
3. 用户："生成4张四季插画" → LLM 选择 generate_images_tool → 调用组图 → 返回多个URL
4. 用户："我想生成图片，帮我推荐一下" → LLM 选择 task_tool(subagent_type="image") → 委托 ImageSubAgent → 多轮对话完成生图
5. 用户："扩写：一个少女在樱花树下" → LLM 选择 expand_prompt_tool → 返回扩写结果

## 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| LLM 工具选择错误 | 系统提示词中详细描述每个工具的适用场景；Fallback 到 chat_tool |
| 工具调用循环 | MAX_STEPS=3 限制；LoopDetectionMiddleware（后续引入） |
| 与现有生图功能冲突 | ImageSubAgent 保留，仅调用方式改变；逐步验证 |
| 结构化输出解析失败 | Fallback 到文本解析或默认 chat 回复 |
