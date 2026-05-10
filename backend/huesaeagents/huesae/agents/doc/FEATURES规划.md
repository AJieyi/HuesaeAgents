# Agent 功能扩展规划

## 一、功能 Phase 划分

### Phase 1: 表情包与语C能力
**优先级**: P0（最简单，先做）

| 功能 | 说明 |
|------|------|
| 颜文字生成 | 根据情绪生成颜文字，如 (≧▽≦)/ |
| 动作描述 | 生成角色动作，如 轻轻歪头 |
| 戳一戳动画 | 返回动画描述，如 *戳了戳你的脸颊* |

**实现方式**: 在 graph.py 的 output_node 中根据情绪添加颜文字/动作

---

### Phase 2: 图片能力
**优先级**: P1

| 功能 | 说明 | 依赖 |
|------|------|------|
| Pixiv爬取 | 爬取Pixiv镜像站图片 | 外部库（httpx/requests） |
| 图片反推标签 | 上传图片返回Danbooru标签 | 图像识别模型 |
| 标签生图 | Danbooru标签 → 生成图片 | ComfyUI/Midjourney API |

**实现方式**: 在 tools/ 模块实现，agent 调用工具

---

### Phase 3: 语音能力
**优先级**: P2

| 功能 | 说明 | 依赖 |
|------|------|------|
| MiniMax语音合成 | 调用MiniMax API生成语音 | MiniMax API |
| 3种语音风格 | 温柔/活泼/沉稳 切换 | 语音种类配置 |
| 附带语音回复 | 对话时附带语音 | Phase 2完成 |

---

## 二、目录结构扩展

```
agents/
├── state.py           # ThreadState状态定义
├── factory.py        # Agent工厂
├── graph.py         # LangGraph工作流
├── prompts.py       # 系统提示词（新增）
├── character/       # 角色管理（扩展）
│   ├── manager.py
│   ├── loader.py
│   └── characters/  # 角色配置
└── tools/          # Agent专用工具（新增）
    ├── emotion.py   # 情绪/颜文字工具
    ├── image.py     # 图片工具（Pixiv/反推/生图）
    └── voice.py     # 语音工具（MiniMax）
```

---

## 三、实现顺序

### Phase 1: 表情包与语C（1-2天）

| 顺序 | 文件 | 说明 |
|------|------|------|
| 1 | prompts.py | 添加角色语气提示词 |
| 2 | graph.py | 修改 output_node 添加颜文字/动作 |
| 3 | 测试 | 验证不同情绪的回复 |

### Phase 2: 图片能力（3-5天）

| 顺序 | 文件 | 说明 |
|------|------|------|
| 1 | tools/image.py | Pixiv爬取 |
| 2 | tools/image.py | 图片反推标签 |
| 3 | tools/image.py | ComfyUI/Midjourney生图 |
| 4 | 测试 | 各功能验证 |

### Phase 3: 语音能力（2-3天）

| 顺序 | 文件 | 说明 |
|------|------|------|
| 1 | tools/voice.py | MiniMax语音合成 |
| 2 | graph.py | 对话附带语音 |
| 3 | 测试 | 语音生成验证 |

---

## 四、关键设计

### 4.1 情绪 → 颜文字映射

```python
EMOTION_KAOMOJI = {
    "开心": ["(≧▽≦)/", "(*^▽^*)", "ヽ(○´∀`)ﾉ♪"],
    "难过": ["(´;ω;`)", "(╥﹏╥)", "QAQ"],
    "害羞": ["(*/ω＼*)", "(〃'▽'〃)"],
    "震惊": ["(°△°|||)", "Σ(°△°|||)"],
    "生气": ["(╬▔皿▔)╯", "(｀Д´)"],
}
```

### 4.2 角色动作描述

```python
CHARACTER_ACTIONS = {
    "gentle_sister": ["轻轻抚摸你的头", "温柔地微笑", "歪头"],
    "tsundere": ["别过脸去", "小声嘟囔", "偷偷瞄你"],
    "furry_fox": ["甩尾巴", "耳朵抖动", "蹭蹭你的手"],
}
```

### 4.3 语音风格配置

```python
VOICE_STYLES = {
    "warm": {"name": "温柔型", "minmax_voice_id": "..."},
    "active": {"name": "活泼型", "minmax_voice_id": "..."},
    "calm": {"name": "沉稳型", "minmax_voice_id": "..."},
}
```

---

## 五、验证方案

### Phase 1 验证
```python
# 测试颜文字生成
result = agent.invoke({
    "messages": [{"role": "user", "content": "今天考试考砸了好难过"}]
})
# 期望回复包含颜文字如 "(´;ω;`)"
```

### Phase 2 验证
```python
# 测试Pixiv爬取
image_url = agent.tools.invoke("search_pixiv", {"query": "银发红瞳"})

# 测试图片反推
tags = agent.tools.invoke("reverse_tags", {"image_url": "..."})

# 测试生图
image = agent.tools.invoke("generate_image", {"tags": "silver_hair, red_eyes"})
```

### Phase 3 验证
```python
# 测试语音合成
voice = agent.tools.invoke("generate_voice", {
    "text": "今天也要加油哦~",
    "voice_type": "warm"
})
```
