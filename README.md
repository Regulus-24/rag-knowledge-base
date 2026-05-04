# 📚 RAG 知识库问答系统

基于 DeepSeek V4 的 1M 上下文窗口，实现文档智能问答。

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![LLM](https://img.shields.io/badge/LLM-DeepSeek%20V4-green)

## 功能

- 📄 **多格式文档** — 支持 PDF、TXT、Markdown、DOCX 上传
- ⚡ **流式输出** — 打字机效果，实时逐字渲染答案
- 📖 **来源标注** — 每个回答附带原文引用片段，可展开查看
- 💬 **多轮对话** — 对话历史记忆，支持上下文追问
- 💾 **数据持久化** — 刷新浏览器文档和对话自动恢复
- 📥 **结果导出** — 对话记录支持 Markdown / JSON 格式导出
- 🔄 **容错重试** — API 调用失败自动指数退避重试

## 技术栈

| 层级 | 技术 |
|------|------|
| 大模型 | DeepSeek V4 Pro（1M 上下文窗口） |
| Web 框架 | Streamlit |
| API 客户端 | OpenAI SDK（兼容 DeepSeek API） |
| 文档解析 | PyPDF + python-docx + LangChain Loader |
| 工程化 | python-dotenv / loguru / tenacity |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env，填入你的 DeepSeek API Key
# DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 3. 放置文档

将知识库文档放入 `data/` 目录，启动后界面会自动检测并提供一键加载按钮。

支持格式：`.pdf` `.txt` `.md` `.docx`

### 4. 启动

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`。

如需局域网内其他设备访问：

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Windows 用户可直接双击 `run.bat`。

## 项目结构

```
rag_project/
├── app.py                  # Web UI 入口
├── config.py               # 集中配置管理
├── run.bat                 # Windows 一键启动
├── .env                    # API Key（不提交）
├── .env.example            # 配置模板
├── requirements.txt        # 依赖清单
├── README.md               # 项目说明
├── CLAUDE.md               # 开发文档
├── PROJECT_DOC.md          # 面试项目文档
│
├── src/
│   ├── llm_client.py       # DeepSeek API 客户端
│   ├── loader.py           # 多格式文档加载器
│   ├── rag_engine.py       # RAG 引擎核心
│   ├── history.py          # 对话历史管理
│   └── cache.py            # 磁盘缓存持久化
│
├── data/                   # 文档存放 + 缓存
└── exports/                # 问答导出
```

## 技术方案

### 为什么不用向量检索？

DeepSeek V4 的 1M token 上下文窗口可以直接容纳整本文档。相较于传统 RAG 的「嵌入 + 向量检索」方案：

- ✅ 信息零丢失（无 chunk 切分损失）
- ✅ 免嵌入模型（DeepSeek 不提供 Embedding API）
- ✅ 实现简洁，维护成本低

当文档规模增长至百万 token 级别时，可引入向量检索作为补充。

## 界面截图

启动后：
1. 主界面中央显示上传区 + `data/` 目录已有文档的快速加载按钮
2. 点击加载文档 → 左侧显示文档统计 → 底部出现聊天输入框
3. 输入问题 → 流式生成答案 → 可展开来源引用
4. 底部可导出 Markdown / JSON 对话记录

## License

MIT
