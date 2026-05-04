# CLAUDE.md — RAG 知识库问答系统

## 项目概述

基于 DeepSeek V4 1M 上下文窗口的 RAG 文档问答系统。不使用向量检索，全文直接传入 LLM 推理。

## 环境

- Python: `C:\Users\23050\Desktop\Python\3.10.11\python.exe`
- 虚拟环境: `venv/`（项目根目录，已创建）
- API: DeepSeek V4（兼容 OpenAI SDK），模型 `deepseek-v4-pro`
- DeepSeek 没有 Embedding API，无法使用向量检索

## 架构

```
app.py                  # Streamlit Web UI 入口
config.py               # 集中配置（.env + 环境变量）
run.bat                 # Windows 一键启动脚本

src/
├── llm_client.py       # DeepSeek API 封装（流式 + 重试）
├── loader.py           # 多格式文档加载器（PDF/TXT/MD/DOCX）
├── rag_engine.py       # RAG 核心（prompt 构造 + 来源提取）
├── history.py          # 对话历史管理（截断 + 导出 MD/JSON）
└── cache.py            # 磁盘持久化（刷新不丢文档和对话）

data/                   # 知识库文档 + 缓存文件
exports/                # 问答导出目录
```

## 关键设计决策

1. **全文推理而非向量检索** — DeepSeek 无 Embedding API；ChromaDB ONNX 模型下载超时（80MB，50KB/s）；当前文档 ~6000 字符远小于 1M 窗口
2. **Streamlit 而非 Gradio** — `st.chat_message` 原生聊天 UI 支持
3. **流式绕过 retry** — `@retry` 装饰器与 generator 不兼容，流式走独立的 `_chat_stream()` 方法
4. **磁盘缓存** — `data/.doc_cache.json` + `data/.conv_cache.json`，刷新浏览器数据不丢失

## 运行方式

```bash
cd C:\Users\23050\Desktop\rag_project
C:\Users\23050\Desktop\Python\3.10.11\python.exe -m streamlit run app.py --server.address 0.0.0.0
```

或双击 `run.bat`。

## 配置

- `.env` — 实际 API Key（不提交 Git）
- `.env.example` — 模板
- `config.py` — `Config` 类，所有值可通过同名环境变量覆盖

## 注意事项

- Windows 环境 stdin/stdout 有 GBK 编码问题，`app.py` 已处理
- PyPDF 读取某些 PDF 产生 `\udc80` 代理字符，`loader.py` 已清理
- 首次启动无文档时，主界面显示上传区 + data/ 中已有文件的一键加载按钮
