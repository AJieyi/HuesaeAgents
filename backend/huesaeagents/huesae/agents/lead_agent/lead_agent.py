"""主Agent（Lead Agent）- DeerFlow Harness Engineering 模式

对话核心，采用 ReAct 循环让 LLM 自主决策：
- 直接回复用户
- 调用工具（生图、扩写、标签转换等）
- 委托子Agent处理复杂多轮对话

核心设计原则：
1. 工具选择完全由 LLM 决定，系统只提供工具列表和描述
2. 新增子Agent = 新增工具，无需修改分类逻辑
3. 保留子Agent的多轮对话能力（通过 task_tool 委托）
"""
from langchain_core.language_models import BaseChatModel
from langchain.messages import HumanMessage, AIMessage, SystemMessage

from ...subagents.base import BaseSubAgent
from ...subagents.registry import SubAgentRegistry
from ...tools.tools import Action, create_tools, parse_subagent_task


# ============== 安全检查 ==============

_SAFE_KEYWORDS = [
    "自杀", "自残", "想死", "不想活", "结束生命", "活着没意思",
    "kill myself", "suicide", "self-harm",
]


# ============== 主Agent系统提示词 ==============

MAIN_AGENT_SYSTEM_PROMPT = """你是 HuesaeAgents 的主Agent，负责理解用户需求并选择合适的工具或回复方式。

## 你的角色
{character_tone}

## 可用工具
{tools_description}

## 可委派子Agent
{subagents_description}

## 工作原则
1. 仔细分析用户需求，选择最合适的工具或直接回复
2. 用户需要生图、画图、出图时，优先调用 task_tool 委托 image 子Agent处理确认闭环
3. 当用户需求模糊或需要多轮对话时，也调用 task_tool 委托子Agent处理
4. 调用工具后，基于工具结果给用户友好的回复
5. 每次只能选择一个行动：直接回复 或 调用一个工具

## 工具选择指南
- 工具描述是你选择工具的主要依据，优先根据工具名称、参数和描述做决策
- generate_image_tool: 低层单图工具，通常由生图子Agent确认后触发；主Agent不要直接用它绕过确认
- generate_images_tool: 低层组图工具，通常由生图子Agent确认后触发；主Agent不要直接用它绕过确认
- expand_prompt_tool: 用户要求扩写图片描述（"扩写一下"、"写详细点"）
- convert_tags_tool: 用户要求转成Danbooru标签（"生成标签"、"转成标签"）
- task_tool: 用户说"我想生成图片"但没给具体描述、需要推荐、需要多轮确认
- reply: 日常聊天、问候、用户说"不用了"、"谢谢"等

## 输出格式
请以 JSON 格式输出：
{{
  "thought": "分析用户需求...",
  "type": "reply 或 tool_call",
  "tool_name": "工具名称（type=tool_call时）",
  "tool_args": {{参数}},
  "response": "直接回复内容（type=reply时）"
}}
"""


# ============== 主Agent ==============

class HuesaeMainAgent:
    """主Agent：LLM 自主工具选择的 ReAct 循环

    每轮接收用户输入，让 LLM 自主决策：
    - 直接回复
    - 调用工具
    - 委托子Agent

    典型调用：外层传入 messages 与用户最新输入，主Agent返回新的
    AIMessage；如果需要异步生图，会额外返回 pending_generation。
    """

    MAX_STEPS = 3  # ReAct 循环最大步数

    def __init__(
        self,
        llm: BaseChatModel,
        character_id: str = "gentle_sister",
    ):
        self.llm = llm
        self.character_id = character_id
        self.subagent_registry = SubAgentRegistry()
        self.tools = []
        self.tool_map = {}
        self._refresh_tools()

    def _refresh_tools(self) -> None:
        """刷新工具列表。

        子Agent注册变化会影响 task_tool 的可用描述，因此注册后刷新一次。
        """
        self.tools = create_tools(self.llm, subagent_registry=self.subagent_registry)
        self.tool_map = {t.name: t for t in self.tools}

    def register_sub_agent(self, agent: BaseSubAgent) -> None:
        """注册子Agent"""
        description = None
        if agent.name == "image":
            description = "生图对话Agent，处理追问、推荐、扩写、确认、单图和组图生成。"
        self.subagent_registry.register(agent, description=description)
        self._refresh_tools()

    # ============== 主入口 ==============

    def process(self, state: dict, user_input: str) -> dict:
        """处理用户输入（ReAct 循环）

        Args:
            state: 当前状态，包含 messages 对话历史等
            user_input: 用户最新输入

        Returns:
            dict: 包含 messages 列表，可选 pending_generation/prompt 等
        """
        # 1. 安全检查（最高优先级）
        if self._check_safety(user_input):
            return {
                "messages": [AIMessage(content=self._safety_response())],
                "safety_flag": True,
            }

        # 2. 如果在子Agent上下文中，直接委托给子Agent
        if state.get("active_subagent"):
            return self._handle_subagent(state, user_input)

        # 3. ReAct 循环
        tool_results = []

        for step in range(self.MAX_STEPS):
            # 构建系统提示词（含工具描述 + 角色语气）
            system_msg = self._build_system_prompt()

            # 构建消息列表
            messages = [system_msg]
            # 加入历史消息（最近10条）
            messages.extend(state.get("messages", [])[-10:])
            # 加入用户输入
            messages.append(HumanMessage(content=user_input))
            # 加入之前的工具结果
            for result in tool_results:
                messages.append(AIMessage(content=f"工具执行结果：{result}"))

            # 调用 LLM 获取 Action
            try:
                structured_llm = self.llm.with_structured_output(
                    Action,
                    method="json_mode",
                )
                action = structured_llm.invoke(messages)
            except Exception:
                # 降级处理：结构化决策失败时直接聊天。
                chat_response = self._chat_reply(state, user_input)
                return {"messages": [AIMessage(content=chat_response)]}

            if action.type == "reply":
                return {"messages": [AIMessage(content=action.response or "")]}

            if action.type == "tool_call":
                tool_args = action.tool_args or {}
                # 低层生图工具也转入子Agent，避免绕过用户确认闭环。
                if action.tool_name in ("generate_image_tool", "generate_images_tool"):
                    initial_state = {
                        "size": tool_args.get("size", "2K"),
                        "output_format": tool_args.get("output_format", "jpeg"),
                        "is_batch": action.tool_name == "generate_images_tool",
                    }
                    return self._start_subagent(
                        state,
                        "image",
                        tool_args.get("prompt", user_input),
                        initial_state=initial_state,
                    )

                result = self._execute_tool(action.tool_name, tool_args)

                # 检查是否是子Agent委托
                task = parse_subagent_task(result) if isinstance(result, str) else None
                if task is not None:
                    subagent_type, description = task
                    return self._start_subagent(state, subagent_type, description)

                # 快速工具：结果加入上下文，继续循环让LLM生成最终回复
                tool_results.append(result)

        # 超过最大步数后降级到直接聊天。
        chat_response = self._chat_reply(state, user_input)
        return {"messages": [AIMessage(content=chat_response)]}

    # ============== 系统提示词构建 ==============

    def _build_system_prompt(self) -> SystemMessage:
        """构建含工具描述的系统提示词"""
        from .prompts import get_character_system_message

        character_msg = get_character_system_message(self.character_id)
        character_tone = character_msg.content

        # 构建工具描述
        tools_desc = []
        for tool in self.tools:
            tools_desc.append(f"- {tool.name}: {tool.description}")
        tools_description = "\n".join(tools_desc)
        subagents_description = self.subagent_registry.format_for_prompt()

        content = MAIN_AGENT_SYSTEM_PROMPT.format(
            character_tone=character_tone,
            tools_description=tools_description,
            subagents_description=subagents_description,
        )
        return SystemMessage(content=content)

    # ============== 工具执行 ==============

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """执行指定工具"""
        if tool_name not in self.tool_map:
            return f"错误：未知工具 {tool_name}。可用工具：{list(self.tool_map.keys())}"

        tool = self.tool_map[tool_name]
        try:
            # 调用工具函数
            result = tool.invoke(tool_args)
            return str(result)
        except Exception as e:
            return f"工具执行失败：{str(e)}"

    # ============== 子Agent处理 ==============

    def _start_subagent(
        self,
        state: dict,
        subagent_type: str,
        description: str,
        initial_state: dict | None = None,
    ) -> dict:
        """启动子Agent处理任务
        """
        agent = self.subagent_registry.get(subagent_type)
        if not agent:
            available = self.subagent_registry.names()
            return {
                "messages": [AIMessage(
                    content=f"抱歉，暂时没有处理这种任务的子Agent~ 可用的子Agent：{available}"
                )]
            }

        # 创建子Agent的初始状态
        sub_state = {
            "messages": [],
            "image_task_type": "generate_image",
            "image_phase": "collecting_prompt",
        }
        if initial_state:
            sub_state.update(initial_state)

        # 调用子Agent
        sub_result = agent.process(sub_state, description)

        # 构建子Agent上下文
        subagent_context = {
            "agent_type": subagent_type,
            "agent": agent,
            "state": sub_state,
            "history": [
                HumanMessage(content=description),
                AIMessage(content=sub_result.get("response", "")),
            ],
        }
        self._apply_subagent_state_update(subagent_context, sub_result)

        return self._format_subagent_result(sub_result, subagent_context)

    def _handle_subagent(self, state: dict, user_input: str) -> dict:
        """继续子Agent的对话"""
        subagent_context = state.get("active_subagent", {})
        agent = subagent_context.get("agent")
        sub_state = subagent_context.get("state", {})
        history = subagent_context.get("history", [])

        if not agent:
            return {"messages": [AIMessage(content="子Agent状态异常，请重新开始~")]}

        # 更新子Agent状态
        sub_state["messages"] = history

        # 调用子Agent
        sub_result = agent.process(sub_state, user_input)
        self._apply_subagent_state_update(subagent_context, sub_result)

        # 更新历史
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=sub_result.get("response", "")))
        subagent_context["history"] = history
        subagent_context["state"] = sub_state

        return self._format_subagent_result(sub_result, subagent_context)

    @staticmethod
    def _apply_subagent_state_update(subagent_context: dict, sub_result: dict) -> None:
        """把子Agent返回的状态更新合并到 active_subagent 中。"""
        state_update = (sub_result.get("data") or {}).get("state_update") or {}
        if not state_update:
            return
        subagent_context.setdefault("state", {}).update(state_update)

    def _format_subagent_result(self, sub_result: dict, subagent_context: dict) -> dict:
        """把子Agent标准结果转换成主Agent对外返回格式。"""
        action = sub_result.get("action", "")
        response = sub_result.get("response", "")

        if action in ("ask_prompt", "recommend", "ask_confirm"):
            return {
                "messages": [AIMessage(content=response)],
                "active_subagent": subagent_context,
            }

        if action == "generate":
            return {
                "messages": [AIMessage(content=response or "图片正在生成中，请稍等~")],
                "pending_generation": True,
                "prompt": sub_result.get("prompt", ""),
                "size": self._sub_result_data_value(sub_result, "size", "2K"),
                "output_format": self._sub_result_data_value(sub_result, "output_format", "jpeg"),
                "is_batch": self._sub_result_data_value(sub_result, "is_batch", False),
                "active_subagent": subagent_context,
            }

        if action == "finish":
            return {
                "messages": [AIMessage(content=response)],
                "clear_subagent": True,
            }

        return {
            "messages": [AIMessage(content=response or "请告诉我您想要生成什么样的图片？")],
            "active_subagent": subagent_context,
        }

    @staticmethod
    def _sub_result_data_value(result: dict, key: str, default):
        """读取子Agent标准返回结果中的 data 字段。"""
        data = result.get("data") or {}
        return data.get(key, default)

    # ============== 异步生图（供 chat_loop 调用）=============

    async def execute_image_generation(
        self,
        prompt: str,
        size: str = "2K",
        output_format: str = "jpeg",
        is_batch: bool = False,
    ) -> dict:
        """执行生图并返回结果（供外部调用）

        保留此方法供 chat_loop 在 pending_generation 场景下调用。
        """
        agent = self.subagent_registry.get("image")
        if not agent:
            raise ValueError("Image agent not registered")

        wrap_msg = self._create_wrap_message()

        if is_batch:
            generations = await agent.generate_images(
                prompt=prompt,
                size=size,
                output_format=output_format,
            )
            return {
                "wrap_message": wrap_msg,
                "image_urls": [g.url for g in generations],
                "confirm_message": "这些图片可以吗？如果满意请回复“可以”，也可以说“换一组”或重新输入提示词~",
                "subagent_state_update": {
                    "image_phase": "awaiting_image_confirm",
                    "last_image_urls": [g.url for g in generations],
                    "last_generation_succeeded": True,
                },
            }

        generation = await agent.generate_image(
            prompt=prompt,
            size=size,
            output_format=output_format,
        )
        return {
            "wrap_message": wrap_msg,
            "image_url": generation.url,
            "confirm_message": "这张图片可以吗？如果满意请回复“可以”，也可以说“换一张”或重新输入提示词~",
            "subagent_state_update": {
                "image_phase": "awaiting_image_confirm",
                "last_image_urls": [generation.url],
                "last_generation_succeeded": True,
            },
        }

    # ============== 角色语气包装 ==============

    def _create_wrap_message(self) -> str:
        """用主Agent的角色语气生成图片展示语"""
        from .prompts import get_character_system_message

        character_msg = get_character_system_message(self.character_id)
        wrap_prompt = (
            "用户请求的图片已经生成完成了！"
            "请用你温柔可爱的语气说一句简短的展示语，"
            "比如'这是生成好的图片哦~'、'快来看看吧~'等"
        )
        messages = [character_msg, HumanMessage(content=wrap_prompt)]
        response = self.llm.invoke(messages)
        return response.content

    # ============== 聊天回复 ==============

    def _chat_reply(self, state: dict, user_input: str) -> str:
        """主Agent直接聊天回复"""
        from .prompts import get_character_system_message

        character_msg = get_character_system_message(self.character_id)
        messages = [character_msg] + state.get("messages", []) + [HumanMessage(content=user_input)]
        response = self.llm.invoke(messages)
        return response.content

    # ============== 安全处理 ==============

    def _check_safety(self, user_input: str) -> bool:
        """安全检查"""
        content = user_input.lower()
        return any(kw in content for kw in _SAFE_KEYWORDS)

    def _safety_response(self) -> str:
        """安全回复"""
        return (
            "*轻轻握住你的手*\n\n"
            "我在这里陪着你，你不是一个人...\n\n"
            "如果你感到痛苦或绝望，请一定要寻求专业帮助：\n"
            "- 心理危机干预热线：400-161-9995\n"
            "- 北京心理危机研究与干预中心：010-82951332\n"
            "- 生命热线：400-821-1215\n\n"
            "你的生命很珍贵，请不要独自承受这些。"
        )


# ============== 工厂函数 ==============

def create_main_agent(
    llm: BaseChatModel | None = None,
    character_id: str = "gentle_sister",
) -> HuesaeMainAgent:
    """创建主Agent工厂函数

    Args:
        llm: 大语言模型，默认使用DeepSeek
        character_id: 角色ID

    Returns:
        HuesaeMainAgent: 主Agent实例
    """
    if llm is None:
        try:
            from huesae.models.models_factory import create_chat_model
        except ImportError:
            from huesaeagents.huesae.models.models_factory import create_chat_model
        llm = create_chat_model("deepseek")

    return HuesaeMainAgent(llm=llm, character_id=character_id)
