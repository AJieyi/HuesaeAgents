# LangGraph 集成计划 — 生图确认流程优化

## Context

当前项目虽然依赖了 `langgraph==1.1.10`，但源代码中完全没有使用 LangGraph。生图子Agent 的"确认→生图→展示→用户确认→结束"流程完全靠 LLM prompt engineering 驱动（[prompts.py:51-137](backend/huesaeagents/huesae/subagents/image/prompts.py#L51-L137)），存在 LLM 不遵守规则的风险（如在用户确认提示词后错误输出 `finish` 而非 `generate`）。

**目标**：用 LangGraph `StateGraph` 替代 ImageSubAgent 内部的 LLM 驱动状态机，将确认→生图→确认→结束的流程用**显式图节点和条件边**保证，同时保持主Agent委派生图Agent的机制不变。

## 核心原则

1. **主委派机制不动** — `BaseSubAgent.process(state, user_input) -> dict` 接口不变，主Agent 的 ReAct 循环和 `active_subagent` 上下文管理不变
2. **确认→生图由代码保证** — `classify_confirm_response` 节点确认后 + `task_type=generate_image` → **必然**路由到 `generate_image_node`，不依赖 LLM 判断
3. **生图任务结束条件** — 只有 `show_result_node` 展示图片后用户确认满意，才路由到 `finish_node`
4. **生图执行在 LangGraph 内部** — async generate node 在 graph 内通过 `asyncio.run()` 执行，不再返回 `pending_generation=True` 给外层

## 架构变化概览

```
当前（LLM驱动）:
  process() → _decide() → LLM输出action → dispatch → 返回结果给主Agent

改造后（LangGraph图驱动）:
  process() → graph.ainvoke() → 图节点执行 → interrupt()暂停 → 返回结果给主Agent
           → graph.ainvoke(Command(resume=...)) → 图恢复 → ... → END
```

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `subagents/image/graph.py` | **新建** | LangGraph 状态定义、节点、边、图构建 |
| `subagents/image_agent.py` | 重写 | `process()` 改为 graph invoke/resume 包装器 |
| `subagents/image/prompts.py` | 修改 | 替换大 prompt 为 3 个分类 prompt + 1 个语气 prompt |
| `agents/lead_agent/lead_agent.py` | 小改 | 简化 `_format_subagent_result()` |
| `agents/lead_agent/chat_loop.py` | 小改 | 移除子Agent路径的 `pending_generation` 特殊处理 |

## 一、状态定义 (`graph.py`)

```python
class ImageGraphState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史（append-only reducer）
    image_task_type: str      # generate_image | expand_prompt | convert_tags
    user_input: str           # 最新用户输入
    prompt: str               # 当前工作提示词
    expanded_prompt: str      # 扩写后的提示词
    generated_url: str        # 生成的图片URL
    generated_urls: list[str] # 组图URLs
    size: str                 # 图片尺寸，默认 2K
    output_format: str        # 输出格式，默认 jpeg
    is_batch: bool            # 是否组图
    provider: str             # Provider名称，默认 doubao
    intent: str               # 当前意图（路由用）
    response: str             # 给用户的回复文本
    action: str               # ask_for_input | ask_confirm | show_result | finish
    has_image: bool           # 本次会话是否已生成过图片
```

## 二、图节点设计

### 分类节点（LLM调用，替代旧的 `_decide()`）

1. **classify_intent** — 入口分类，判断用户意图：
   - `has_description` — 用户提供了具体描述
   - `needs_description` — 用户想生图但没描述
   - `wants_recommendation` — 用户要求推荐
   - `wants_direct_expand` — 用户明确要求扩写
   - `end_conversation` — 用户想结束

2. **classify_confirm_response** — 提示词确认后的响应分类：
   - `confirmed` / `wants_expand` / `wants_modify` / `end_conversation`

3. **classify_post_image_response** — 展示图片后的响应分类：
   - `satisfied` / `regenerate` / `wants_expand` / `new_prompt` / `modify_prompt` / `end_conversation`

### 中断节点（`interrupt()` 暂停等用户输入）

4. **ask_for_prompt_node** — 追问描述 → `interrupt({"message": "请描述...", "action": "ask_for_input"})`
5. **recommend_node** — 推荐主题 → `interrupt(...)`
6. **offer_expand_node** — 短提示词时询问是否扩写 → `interrupt(...)`
7. **confirm_prompt_node** — 展示提示词，询问确认 → `interrupt({"message": "...", "action": "ask_confirm"})`
8. **show_result_node** — 展示生成图片，询问满意度 → `interrupt({"message": "...", "action": "show_result", "image_url": url})`

### 动作节点

9. **build_prompt_node** — 设置工作提示词，调用 `_ensure_anime_style()`
10. **expand_prompt_node** — 调用 LLM 扩写提示词（复用现有 `expand_prompt()`）
11. **generate_image_node** (async) — 调用 Provider 生图，错误时设置错误信息
12. **finish_node** — 设置 `action="finish"`，路由到 END

## 三、图边路由（核心确定性逻辑）

```
START → classify_intent
  ├── needs_description → ask_for_prompt → build_prompt
  ├── wants_recommendation → recommend → build_prompt
  ├── has_description → build_prompt
  ├── wants_direct_expand → expand_prompt → confirm_prompt
  └── end_conversation → finish → END

build_prompt
  ├── prompt短 + task=generate_image → offer_expand
  │     ├── 用户要扩写 → expand_prompt → confirm_prompt
  │     └── 用户不要 → confirm_prompt
  └── 其他 → confirm_prompt

confirm_prompt → classify_confirm_response
  ├── confirmed + task=generate_image → generate_image  ⬅ 代码保证！
  ├── confirmed + task=expand_prompt → finish → END
  ├── confirmed + task=convert_tags → finish → END
  ├── wants_expand → expand_prompt → confirm_prompt
  ├── wants_modify → build_prompt
  └── end_conversation → finish → END

generate_image → show_result → classify_post_image_response
  ├── satisfied → finish → END  ⬅ 只有这里才算任务完成！
  ├── regenerate → generate_image
  ├── wants_expand → expand_prompt → confirm_prompt → generate_image
  ├── new_prompt → build_prompt → confirm_prompt → generate_image
  ├── modify_prompt → build_prompt → confirm_prompt → generate_image
  └── end_conversation → finish → END
```

## 四、`ImageSubAgent.process()` 改造

```python
def process(self, state: dict, user_input: str) -> dict:
    thread_id = state.get("_image_thread_id")
    is_new = thread_id is None
    if is_new:
        thread_id = str(uuid.uuid4())
        state["_image_thread_id"] = thread_id

    config = {"configurable": {"thread_id": thread_id}}

    if is_new:
        graph_input = {
            "messages": state.get("messages", []),
            "image_task_type": state.get("image_task_type", "generate_image"),
            "user_input": user_input,
            "prompt": state.get("image_prompt", ""),
            "stage": "entry", ...
        }
        result = asyncio.run(self.graph.ainvoke(graph_input, config))
    else:
        result = asyncio.run(self.graph.ainvoke(Command(resume=user_input), config))

    # 检查是否中断
    snapshot = self.graph.get_state(config)
    if snapshot and snapshot.interrupts:
        interrupt_data = snapshot.interrupts[0].value
        return _make_result(
            action=interrupt_data.get("action", "ask_for_input"),
            response=interrupt_data.get("message", ""),
            image_url=interrupt_data.get("image_url"),
            image_urls=interrupt_data.get("image_urls"),
            ...
        )

    # 图已完成
    return _make_result(action="finish", response="任务完成~")
```

关键点：
- `asyncio.run()` 桥接 async graph 到 sync `process()` 接口
- `thread_id` 存储在共享 `state` dict 上，随 `subagent_context["state"]` 跨调用传递
- `interrupt()` 的返回值直接从 `snapshot.interrupts[0].value` 读取，包含 `message`、`action`、`image_url` 等

## 五、主Agent 改动 (`lead_agent.py`)

`_format_subagent_result()` 简化 — 不再需要处理 `pending_generation`：

```python
def _format_subagent_result(self, sub_result, subagent_context):
    action = sub_result.get("action", "")
    response = sub_result.get("response", "")

    if action == "finish":
        return {"messages": [AIMessage(content=response)], "clear_subagent": True}

    result = {
        "messages": [AIMessage(content=response)],
        "active_subagent": subagent_context,
    }
    # 透传图片URL
    if sub_result.get("image_url"):
        result["image_url"] = sub_result["image_url"]
    if sub_result.get("image_urls"):
        result["image_urls"] = sub_result["image_urls"]
    return result
```

## 六、Chat Loop 改动 (`chat_loop.py`)

去除子Agent路径的 `pending_generation` 特殊处理。当前代码（[chat_loop.py:102-148](backend/huesaeagents/huesae/agents/lead_agent/chat_loop.py#L102-L148)）检测 `result.get("pending_generation")` 后异步执行生图——这个分支对子Agent不再触发。改为统一走消息展示分支，并在展示后打印图片URL。

**直接工具调用路径**（ReAct 循环中 `generate_image_tool` 返回 `pending_generation=True`）保持不变。

## 七、Prompt 变更 (`prompts.py`)

删除 `IMAGE_CONVERSATION_PROMPT`（135行），替换为 3 个轻量分类 prompt：
- `ENTRY_INTENT_PROMPT` — 入口意图分类
- `CONFIRM_RESPONSE_PROMPT` — 确认响应分类
- `POST_IMAGE_RESPONSE_PROMPT` — 图片反馈分类

保留 `DANBOORU_TAG_PROMPT` 和 `EXPAND_PROMPT_SYSTEM`（被 expand_prompt_node 复用）。

## 八、用户案例验证

### 案例1（基本生图确认）
```
用户："我需要生图" → classify_intent→needs_description → ask_for_prompt(interrupt)
用户："猫娘在咖啡馆" → build_prompt → confirm_prompt(interrupt)
用户："可以" → classify_confirm→confirmed+generate_image → generate_image → show_result(interrupt)
用户："可以" → classify_post_image→satisfied → finish → END ✅
```

### 案例2（生图后扩写再确认）
```
用户确认提示词 → generate → show_result(interrupt)
用户："扩写一下我的提示词" → classify_post_image→wants_expand → expand → confirm_prompt(interrupt)
用户："可以" → classify_confirm→confirmed+generate_image → generate → show_result(interrupt)
用户："可以" → classify_post_image→satisfied → finish → END ✅
```

### 案例3（换一张）
```
... → show_result(interrupt)
用户："换一张" → classify_post_image→regenerate → generate → show_result(interrupt)
用户："可以" → classify_post_image→satisfied → finish → END ✅
```

### 案例4（重新输入提示词）
```
... → show_result(interrupt)
用户："我重新输入一组提示词" → classify_post_image→new_prompt → build_prompt → confirm_prompt(interrupt)
用户："可以" → classify_confirm→confirmed+generate_image → generate → show_result(interrupt)
用户："可以" → classify_post_image→satisfied → finish → END ✅
```

## 实现顺序

1. **新建** `subagents/image/graph.py` — 状态、节点、边、`build_graph()` 工厂函数
2. **修改** `subagents/image/prompts.py` — 替换为 3 个分类 prompt
3. **重写** `subagents/image_agent.py` — `__init__` 构建 graph，`process()` 改为 invoke/resume 包装，保留 `generate_image()`/`generate_images()`/`register_provider()`
4. **小改** `agents/lead_agent/lead_agent.py` — 简化 `_format_subagent_result()`
5. **小改** `agents/lead_agent/chat_loop.py` — 去除子Agent `pending_generation` 分支，添加图片URL展示

## 验证方式

1. 运行 `python chat_loop.py` 进行端到端交互测试
2. 验证上述 4 个案例的完整流程
3. 验证非生图任务（expand_prompt / convert_tags）确认后直接结束
4. 验证直接工具调用路径（"生成一张夕阳下的大海"）仍然正常工作
