# RAG 知识库问答系统 — 项目文档

> 基于 DeepSeek V4 1M 上下文窗口的智能文档问答系统
>
> 适用于：简历项目经历 / 面试准备 / 技术复盘

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [技术方案与决策](#3-技术方案与决策)
4. [核心模块详解](#4-核心模块详解)
5. [关键知识点](#5-关键知识点)
6. [面试问答准备](#6-面试问答准备)
7. [后续演进方向](#7-后续演进方向)

---

## 1. 项目概述

### 一句话描述

一个支持多格式文档上传、基于大模型 1M 上下文窗口的智能问答系统，具备流式输出、来源引用标注、多轮对话和结果导出能力。

### 核心特性

| 特性 | 说明 |
|------|------|
| 多格式文档 | 支持 PDF、TXT、Markdown、DOCX |
| 1M 上下文推理 | 利用 DeepSeek V4 超长窗口，全文直接传入 |
| 流式输出 | SSE 流式生成，打字机效果 |
| 来源标注 | 每个回答附带原文引用片段 |
| 多轮对话 | 对话历史管理，上下文记忆 |
| 结果导出 | 支持 Markdown / JSON 格式导出 |
| 容错重试 | API 调用失败自动指数退避重试 |
| Web UI | Streamlit 构建的类 ChatGPT 交互界面 |

---

## 2. 系统架构

```
                         用户
                          │
                    ┌─────▼──────┐
                    │  Streamlit │  Web UI（app.py）
                    │  浏览器界面  │
                    └─────┬──────┘
                          │
            ┌─────────────┼─────────────┐
            │             │             │
      ┌─────▼─────┐ ┌────▼────┐ ┌─────▼──────┐
      │  文档加载   │ │ RAG引擎 │ │  对话历史   │
      │ loader.py │ │ engine  │ │ history.py │
      └─────┬─────┘ └────┬────┘ └────────────┘
            │             │
      ┌─────▼─────┐ ┌────▼────────┐
      │ PDF/TXT   │ │ DeepSeek V4 │  llm_client.py
      │ MD/DOCX   │ │    API      │  (流式 + 重试)
      └───────────┘ └─────────────┘
```

### 数据流

```
1. 用户上传文档 → loader.py 解析 → 全文文本（~6000 字符）
2. 用户输入问题 → rag_engine.py 构造 prompt（系统提示 + 全文 + 历史 + 问题）
3. llm_client.py 调用 DeepSeek API（stream=True）
4. 流式返回 → app.py 逐 chunk 渲染（打字机效果）
5. 回答完成 → extract_sources() 提取引用 → UI 展示来源卡片
6. 对话保存 → ConversationHistory → 支持导出
```

---

## 3. 技术方案与决策

### 3.1 为什么不用向量检索（传统 RAG）？

| 方案 | 优点 | 缺点 | 本项目选择 |
|------|------|------|-----------|
| 向量检索 RAG | 文档量大时有优势 | 需要嵌入模型 API、向量数据库、chunk 策略调优 | ❌ |
| 1M 上下文全文推理 | 实现简单、无信息丢失、免嵌入模型 | 文档太大时 token 成本高 | ✅ |

**决策依据：**

1. DeepSeek 不提供嵌入模型 API（返回 404），使用 ChromaDB 本地 ONNX 模型需下载 80MB 且网络不稳定
2. 当前文档规模（~6000 字符 ≈ 3000 tokens）远低于 1M 窗口，全文传入无损且成本可控
3. 全文推理避免了 chunk 切分导致的信息断裂问题，回答质量更高

**面试时可以说：** "我评估了两种方案。向量检索需要嵌入模型和向量数据库，在当前场景下引入了不必要的复杂度。DeepSeek V4 的 1M 上下文窗口让我可以直接做全文推理，信息零丢失，实现更简洁。当文档规模增长到百万 token 级别时，可以再引入向量检索作为补充。"

### 3.2 为什么用 Streamlit 而不是 Gradio？

- Streamlit 的 `st.chat_message` + `st.chat_input` 原生支持聊天 UI
- 社区更大，面试官更可能认识
- Python 代码即界面，无前端开发成本

### 3.3 API 容错策略

使用 `tenacity` 库实现指数退避重试：
- 最大重试 3 次
- 等待间隔：2s → 4s → 8s（指数增长，上限 30s）
- 面试时可提及"生产级 API 调用的容错处理"

---

## 4. 核心模块详解

### 4.1 `src/loader.py` — 多格式文档加载器

**设计模式：策略模式**

```python
LOADER_MAP = {
    ".pdf":  load_pdf,      # langchain PyPDFLoader
    ".txt":  load_txt,      # 原生 UTF-8 读取
    ".md":   load_txt,      # Markdown → 纯文本
    ".docx": load_docx,     # python-docx 解析
}
```

**技术要点：**
- Unicode 代理字符清理（`surrogateescape` → `decode('utf-8', errors='replace')` → regex 移除）
- 统一接口：`load_document(file_path) -> (file_name, full_text)`
- 扩展新格式只需添加加载函数和映射

### 4.2 `src/llm_client.py` — API 客户端

**技术要点：**
- `OpenAI` SDK 兼容 DeepSeek API（`base_url` 指向 `https://api.deepseek.com/v1`）
- 流式调用：`stream=True` 返回生成器，逐 chunk yield 增量文本
- 重试装饰器：`@retry(stop=stop_after_attempt(3), wait=wait_exponential(...))`

```python
# 流式调用核心代码
def ask_stream(self, system_prompt, user_message):
    response = self.chat(messages=[...], stream=True)
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

### 4.3 `src/rag_engine.py` — RAG 核心

**Prompt 构造策略：**

```
系统提示（角色定义 + 输出格式要求）
  ↓
历史对话（最近 3 轮，每条截断 500 字符）
  ↓
文档全文（完整知识库内容）
  ↓
用户问题
```

**来源引用提取：**

1. 优先解析 LLM 生成的 `【来源引用】` 标记块
2. 回退方案：从回答中提取关键句，在原文中搜索匹配片段

### 4.4 `src/history.py` — 对话历史管理

**技术要点：**
- 自动截断：保留最近 N 轮（默认 20 轮），超出自动丢弃最早消息
- 双格式导出：`export_markdown()` 和 `export_json()`
- O(1) 消息追加，内存友好

---

## 5. 关键知识点

> 以下是面试中可能被追问的技术概念，按模块归类。

### 5.1 RAG（检索增强生成）

**核心概念：**
- RAG = Retrieval + Augmented + Generation
- 传统 RAG 流程：文档 → Chunk → Embedding → VectorStore → 相似度检索 → LLM 生成
- 本项目变体：1M 上下文全文推理（免检索的 RAG）

**面试常见追问：**
- "chunk_size 怎么选？" → 取决于嵌入模型的 max_seq_length，通常 256-1024 tokens，overlap 10-20%
- "怎么评估检索质量？" → Recall@K、MRR（Mean Reciprocal Rank）
- "向量数据库选型？" → ChromaDB（轻量）、Milvus（生产级）、Pinecone（云服务）

### 5.2 大模型上下文窗口

**概念：**
- Context Window = 模型一次能处理的 token 总数（输入 + 输出）
- DeepSeek V4：1M tokens（约 70 万汉字）
- GPT-4 Turbo：128K tokens
- Claude：200K tokens

**1M 上下文的实际意义：**
- 可以一次输入整本书（《三体》约 20 万字 ≈ 30 万 tokens）
- 免去了 chunk 切分的信息丢失问题
- 但长上下文推理速度较慢，token 成本较高

**面试追问：**
- "长上下文有什么问题？" → "Lost in the Middle"效应（中间信息容易被忽视）、推理延迟增加
- "怎么优化长上下文？" → 仍然可以结合检索，先用向量检索缩小范围再送入长窗口

### 5.3 流式输出（Streaming）

**概念：**
- 非流式：发送请求 → 等待完整响应 → 一次返回
- 流式（SSE）：发送请求 → 逐 token 返回 → 实时渲染

**实现方式：**
```python
# 关键参数：stream=True
response = client.chat.completions.create(..., stream=True)
for chunk in response:
    delta = chunk.choices[0].delta.content  # 增量文本
```

**面试追问：**
- "SSE vs WebSocket？" → SSE 单向（服务器→客户端）、WebSocket 双向；聊天场景 SSE 足够
- "前端怎么消费流式？" → Fetch API + ReadableStream，或 Streamlit 的 `st.write_stream()`

### 5.4 多轮对话管理

**核心问题：**
- 每次请求需要携带历史消息，否则模型"失忆"
- 历史过长会超出上下文窗口 → 需要截断策略

**本项目的策略：**
- 保留最近 20 轮（40 条消息）
- 超出的自动丢弃
- 历史传给 LLM 时截断每条至 500 字符

**面试追问：**
- "更好的截断策略？" → 滑动窗口、摘要压缩（用 LLM 压缩历史）、重要性评分
- "怎么实现持久化？" → 数据库存储 session，每次请求从 DB 加载历史

### 5.5 API 容错与重试

**常见失败场景：**
- 网络超时
- API 限流（429 Too Many Requests）
- 服务临时不可用（503）

**指数退避（Exponential Backoff）：**
```
第1次重试：等待 2 秒
第2次重试：等待 4 秒
第3次重试：等待 8 秒
```

**面试追问：**
- "为什么要加 jitter？" → 防止多个客户端同时重试造成"惊群效应"
- "幂等性怎么保证？" → 对于非幂等操作，需要去重键（idempotency key）

### 5.6 DeepSeek API 兼容性

**技术细节：**
- DeepSeek API 与 OpenAI API 接口兼容
- 通过修改 `base_url` 即可切换：`https://api.deepseek.com/v1`
- 特性差异：DeepSeek 不支持 Embedding API、Function Calling 可能有限制

**面试展示：** 说明自己了解不同厂商 API 的差异，能灵活适配。

### 5.7 Prompt Engineering

**本项目使用的技巧：**
- 角色设定（System Prompt）："你是一个专业的知识库问答助手"
- 输出格式约束：要求包含 `【来源引用】` 标记
- 边界条件处理："文档中未找到相关信息"（防幻觉）
- Few-shot 提示：历史对话作为示例

---

## 6. 面试问答准备

### 6.1 "介绍一下这个项目"

> **30 秒版本：**
> 我开发了一个 RAG 知识库问答系统，支持上传 PDF、Word 等多格式文档，利用 DeepSeek V4 的 100 万 token 上下文窗口实现全文智能问答。系统具备流式输出、来源引用标注、多轮对话和结果导出功能，Web 界面用 Streamlit 构建。

> **2 分钟版本（加上技术细节）：**
> [30秒版本] + 技术选型上，我评估了向量检索和全文推理两种方案。因为 DeepSeek 不支持嵌入 API，且当前文档规模远小于 1M 上下文窗口，我选择了全文直接推理，避免了向量数据库的复杂度。API 调用了做了指数退避重试保证稳定性，对话历史管理支持自动截断防止超出上下文。整个项目采用模块化架构，配置通过 dotenv 管理，API Key 不硬编码。

### 6.2 展示用的 3 个问答对

**Q1（基础 — 展示理解能力）：**
> 这份文档主要讲了什么？

文档从基础层和进阶层两个维度，系统梳理了大模型训练与推理的核心技术，包括注意力机制、分布式并行、量化压缩、Flash Attention、RLHF 训练流程等。

---

**Q2（进阶 — 展示精确检索能力）：**
> Flash Attention 的原理是什么？和标准 Attention 有什么区别？

Flash Attention 的核心是不把完整注意力矩阵写回显存，而是采用分块计算、分块 softmax、分块累加的方式。显存复杂度从 O(N²) 降到 O(N)。速度方面，利用 SRAM 计算比多次读写 HBM 快 20 倍，整体提速 2-4 倍。

---

**Q3（深度 — 展示综合能力）：**
> 如果我要训练一个大模型，从预训练到部署，应该经过哪些阶段？每个阶段的关键技术是什么？

三阶段：PT（预训练，需要处理梯度爆炸/消失、混合精度 FP16/BF16、分布式并行 DP/MP/PP）→ SFT（监督微调，可用 QLoRA 高效微调）→ RLHF（人类反馈强化学习，对齐人类偏好）。部署阶段关注 KV Cache、量化 INT8/INT4、Flash Attention、Chunked Prefill 等推理优化技术。

### 6.3 展示流程建议

```
1. [30秒] 一句话介绍项目
2. [1分钟] 打开 Streamlit 界面，上传 PDF，问 Q1 — 展示基础功能
3. [1分钟] 问 Q2，展示来源引用展开 — 展示精确性
4. [1分钟] 问 Q3，展示多轮对话 — 展示深度
5. [30秒] 导出对话记录 — 展示完整产品
6. [30秒] 总结技术亮点
```

---

## 7. 后续演进方向

| 方向 | 技术点 | 面试加分 |
|------|--------|---------|
| 向量检索 RAG | 嵌入模型 + ChromaDB/Milvus | 🔥🔥🔥 |
| 混合检索 | 向量检索 + BM25 关键词检索 | 🔥🔥🔥 |
| 多模态支持 | 图片 OCR / 表格提取 | 🔥🔥 |
| 权限管理 | 用户登录 / API Key 隔离 | 🔥🔥 |
| 异步处理 | FastAPI + Celery 大文件异步解析 | 🔥🔥 |
| Docker 部署 | 一键部署脚本 | 🔥 |
| 评估体系 | RAGAS 评估框架（Faithfulness/Relevance） | 🔥🔥🔥 |
| LangSmith 追踪 | LLM 调用链可观测 | 🔥🔥 |
