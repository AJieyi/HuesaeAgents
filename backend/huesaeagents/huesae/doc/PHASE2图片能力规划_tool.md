# Phase2: 图片能力实现规划

## 一、功能列表

| 功能 | 说明 | 优先级 | 状态 |
|------|------|---------|------|
| **自然语言生图（即梦）** | 自然语言 → 即梦AI → 返回图片 | **已完成** | ✅ |
| **自然语言生图（豆包）** | 自然语言 → 豆包Seedream → 返回图片URL | **已完成** | ✅ |
| **自然语言生一组图（豆包）** | 自然语言 → 豆包Seedream → 返回多张图片（最多4张） | **已完成** | ✅ |
| Pixiv爬取 | 爬取Pixiv镜像站图片，返回图片URL | 后续 | 🔲 |
| 图片反推标签 | 上传图片，返回Danbooru标签 | 后续 | 🔲 |
| 标签生图 | Danbooru标签 → 生成图片（ComfyUI/Midjourney） | 后续 | 🔲 |

---

## 二、目录结构

```
huesae/
├── agents/                  # Agent系统
│   ├── graph.py           # 主Agent工作流
│   └── lead_agent/        # 主Agent（未来扩展）
│
└── tools/                  # 工具系统
    ├── __init__.py        # 【已实现】工具导出
    ├── image.py           # 【已实现】图片生成统一入口
    ├── doubao/            # 【已实现】豆包图片生成
    │   ├── __init__.py
    │   └── client.py
    ├── jimeng/            # 【已实现】即梦图片生成
    │   ├── __init__.py
    │   └── client.py
    └── doc/               # 工具文档
        └── 即梦AI-图片生成4.0-接口文档.pdf
```

---

## 三、已实现功能

### 3.1 即梦AI（Jimeng）

**环境变量：**
```bash
JIMENG_ACCESS_KEY_ID=xxx
JIMENG_SECRET_ACCESS_KEY=xxx
```

**使用方式：**
```python
from tools.image import generate_image_by_jimeng

# 文生一张图
url = await generate_image_by_jimeng(
    prompt="一个银发红瞳的少女在樱花树下",
    width=1024,
    height=1024,
)
print(url)
```

### 3.2 豆包Seedream（Doubao）

**环境变量：**
```bash
DOUBAO_SEEDREAM_API_KEY=xxx
DOUBAO_SEEDREAM_MODEL_NAME=doubao-seedream-5-0-260128
```

**使用方式：**
```python
from tools.image import generate_image_by_doubao, generate_images_by_doubao

# 文生一张图（返回URL）
url = await generate_image_by_doubao(
    prompt="一个银发红瞳的少女在樱花树下，电影感",
    size="2K",
)

# 文生一组图（返回base64列表，最多4张）
images = await generate_images_by_doubao(
    prompt="生成一组共4张连贯插画，核心为同一庭院一角的四季变迁",
    size="2K",
    max_images=4,
)
```

---

## 四、实现顺序

| 顺序 | 任务 | 预计时间 | 状态 |
|------|------|----------|------|
| 1 | **自然语言生图（即梦AI）** | 1天 | ✅ 已完成 |
| 2 | **自然语言生图（豆包）** | 1天 | ✅ 已完成 |
| 3 | **文生一组图（豆包）** | 1天 | ✅ 已完成 |
| 4 | Agent集成图片生成能力 | 1天 | 🔲 待开始 |
| 5 | search_pixiv 实现 | 1天 | 🔲 后续 |
| 6 | reverse_tags 实现 | 1天 | 🔲 后续 |
| 7 | generate_image (ComfyUI) | 1天 | 🔲 后续 |

---

## 五、环境配置清单

```bash
# .env 文件
DEEPSEEK_API_KEY=xxx              # DeepSeek API密钥
JIMENG_ACCESS_KEY_ID=xxx         # 即梦AI Access Key
JIMENG_SECRET_ACCESS_KEY=xxx     # 即梦AI Secret Key
DOUBAO_SEEDREAM_API_KEY=xxx      # 豆包 Seedream API Key
DOUBAO_SEEDREAM_MODEL_NAME=xxx   # 豆包模型名，默认 doubao-seedream-5-0-260128
```

---

## 六、下一步计划

### 6.1 Agent 集成图片生成能力

**目标：** 将图片生成工具接入 Agent 工作流，让 Agent 能够根据用户需求调用图片生成。

**实现思路：**
1. 在 `agents/graph.py` 中注册图片生成工具
2. 定义工具调用节点：`generate_image_by_jimeng`、`generate_image_by_doubao`、`generate_images_by_doubao`
3. 设计 Agent 提示词，使其能够判断何时需要生成图片

**示例流程：**
```
用户: "画一个银发红瞳的少女"
Agent: 分析需求 → 调用 generate_image_by_doubao → 返回图片URL → 展示给用户
```

---

## 七、验证方案

```bash
# 运行测试
cd backend
python huesaeagents/huesae/tools/test_image.py
```

**预期输出：**
```
==================================================
图片生成工具测试
==================================================

测试豆包文生图...
豆包生成成功: https://xxx.jpg

测试豆包文生一组图...
豆包生成一组图成功，共 4 张图片
  第1张图片已保存: output_images/doubao_images_xxx_1.png
  ...
```

---

## 八、后续功能

| 功能 | 说明 |
|------|------|
| Pixiv爬取 | 爬取Pixiv镜像站图片 |
| 图片反推标签 | 上传图片返回Danbooru标签 |
| 标签生图 | ComfyUI/Midjourney生图 |

---

## 九、关键参考

- **即梦AI API文档**：`https://www.volcengine.com/docs/85621/1817045?lang=zh`
- **豆包Seedream API**：使用 OpenAI 兼容格式，`https://ark.cn-beijing.volces.com/api/v3`
- **错误处理**：注意API限流和错误码处理
