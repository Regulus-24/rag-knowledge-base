"""
RAG 知识库问答系统 — Streamlit Web UI
======================================
基于 DeepSeek V4 1M 上下文窗口 | 多格式文档 | 流式输出 | 来源标注 | 缓存持久化
"""

import sys
from pathlib import Path

# 确保 src/ 在路径中
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from loguru import logger

from config import config
from src.loader import load_document
from src.rag_engine import ask, extract_sources, get_document_stats
from src.history import ConversationHistory
from src.cache import (
    save_document_cache,
    load_document_cache,
    clear_document_cache,
    save_conversation_cache,
    load_conversation_cache,
    clear_conversation_cache,
)


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(f"📚 {config.APP_TITLE}")
st.caption(config.APP_DESCRIPTION)


# ============================================================
# 初始化 Session State（启动时自动从磁盘恢复）
# ============================================================
def init_session():
    """初始化 session state，优先从磁盘缓存恢复。"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True

        # 1. 尝试恢复文档
        cached = load_document_cache()
        if cached:
            st.session_state.full_text = cached[1]
            st.session_state.file_name = cached[0]
            st.session_state.doc_loaded = True
        else:
            st.session_state.full_text = ""
            st.session_state.file_name = ""
            st.session_state.doc_loaded = False

        # 2. 尝试恢复对话
        cached_msgs = load_conversation_cache()
        st.session_state.messages = cached_msgs

        # 3. 重建对话历史对象
        st.session_state.history = ConversationHistory()
        for msg in cached_msgs:
            if msg["role"] == "user":
                st.session_state.history.add_user(msg["content"])
            elif msg["role"] == "assistant":
                st.session_state.history.add_assistant(msg["content"])


init_session()


# ============================================================
# 侧边栏 — 文档上传 & 配置
# ============================================================
with st.sidebar:
    st.header("📁 文档管理")

    # 显示缓存状态
    if st.session_state.doc_loaded:
        st.success(f"📄 {st.session_state.file_name}")
    else:
        st.info("尚未加载文档")

    uploaded_file = st.file_uploader(
        "上传知识库文档",
        type=list(fmt.lstrip(".") for fmt in config.SUPPORTED_FORMATS.keys()),
        help=f"支持格式：{', '.join(config.SUPPORTED_FORMATS.values())}",
        label_visibility="collapsed" if st.session_state.doc_loaded else "visible",
    )

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > config.MAX_FILE_SIZE_MB:
            st.error(f"文件过大（{file_size_mb:.1f}MB），最大 {config.MAX_FILE_SIZE_MB}MB")
        else:
            temp_path = Path(f"data/{uploaded_file.name}")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_bytes(uploaded_file.getvalue())

            with st.spinner("正在解析文档..."):
                try:
                    file_name, full_text = load_document(temp_path)
                    st.session_state.full_text = full_text
                    st.session_state.file_name = file_name
                    st.session_state.doc_loaded = True
                    st.session_state.messages = []
                    st.session_state.history = ConversationHistory()

                    # 持久化文档 + 清除旧对话缓存
                    save_document_cache(file_name, full_text)
                    clear_conversation_cache()

                    logger.info(f"文档加载成功：{file_name}")
                    st.success(f"✅ {file_name} 加载完成（已缓存，刷新不丢）")
                    st.rerun()
                except Exception as e:
                    st.error(f"文档加载失败：{e}")
                    logger.error(f"文档加载失败：{e}")

    # 文档统计
    if st.session_state.doc_loaded:
        st.markdown("---")
        st.subheader("📊 文档概况")
        stats = get_document_stats(st.session_state.full_text)
        for label, value in stats.items():
            st.write(f"- {label}: {value:,}")

    st.markdown("---")
    st.subheader("⚙️ 模型设置")

    # 这些值存在 session 中，刷新不丢
    if "temperature" not in st.session_state:
        st.session_state.temperature = config.TEMPERATURE
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = config.MAX_TOKENS

    st.session_state.temperature = st.slider(
        "Temperature", 0.0, 1.0, st.session_state.temperature, 0.05,
        help="越高回答越有创意，越低回答越确定",
    )
    st.session_state.max_tokens = st.number_input(
        "Max Tokens", 256, 8192, st.session_state.max_tokens, 256,
        help="单次回答最大长度",
    )

    if st.button("🗑️ 清除对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = ConversationHistory()
        clear_conversation_cache()
        st.rerun()

    if st.button("🗑️ 清除全部缓存", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = ConversationHistory()
        st.session_state.doc_loaded = False
        st.session_state.full_text = ""
        st.session_state.file_name = ""
        clear_document_cache()
        clear_conversation_cache()
        st.rerun()

    st.markdown("---")
    st.caption(f"Powered by {config.DEEPSEEK_MODEL}")


# ============================================================
# 主区域 — 聊天历史
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📖 来源引用"):
                for i, src in enumerate(msg["sources"], 1):
                    st.info(f"**[{i}]** {src}")


# ============================================================
# 输入区 — 问答交互
# ============================================================
if not st.session_state.doc_loaded:
    # 大而显眼的中央上传区，替代侧边栏的小上传组件
    st.markdown("### 📄 上传知识库文档开始问答")

    # 检测 data/ 中已有文件，提供一键加载
    data_dir = Path("data")
    existing_files = []
    if data_dir.exists():
        for f in data_dir.iterdir():
            if f.suffix.lower() in config.SUPPORTED_FORMATS:
                existing_files.append(f)

    if existing_files:
        st.markdown("**🗂️ 检测到本地文档，点击即可加载：**")
        cols = st.columns(min(len(existing_files), 3))
        for i, fpath in enumerate(existing_files):
            with cols[i % 3]:
                if st.button(
                    f"📄 {fpath.name}\n({fpath.stat().st_size // 1024} KB)",
                    key=f"quick_load_{fpath.name}",
                    use_container_width=True,
                ):
                    try:
                        file_name, full_text = load_document(str(fpath))
                        st.session_state.full_text = full_text
                        st.session_state.file_name = file_name
                        st.session_state.doc_loaded = True
                        st.session_state.messages = []
                        st.session_state.history = ConversationHistory()
                        save_document_cache(file_name, full_text)
                        clear_conversation_cache()
                        st.rerun()
                    except Exception as e:
                        st.error(f"加载失败：{e}")
        st.markdown("---")
        st.markdown("**📤 或上传新文档：**")

    # 主区域大号上传器
    uploaded_main = st.file_uploader(
        "拖拽文件到此处，或点击浏览",
        type=list(fmt.lstrip(".") for fmt in config.SUPPORTED_FORMATS.keys()),
        key="main_uploader",
        help=f"支持：PDF、TXT、Markdown、DOCX | 最大 {config.MAX_FILE_SIZE_MB}MB",
    )

    if uploaded_main is not None:
        file_size_mb = uploaded_main.size / (1024 * 1024)
        if file_size_mb > config.MAX_FILE_SIZE_MB:
            st.error(f"文件过大（{file_size_mb:.1f}MB），最大 {config.MAX_FILE_SIZE_MB}MB")
        else:
            temp_path = Path(f"data/{uploaded_main.name}")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_bytes(uploaded_main.getvalue())

            with st.spinner("正在解析文档..."):
                try:
                    file_name, full_text = load_document(temp_path)
                    st.session_state.full_text = full_text
                    st.session_state.file_name = file_name
                    st.session_state.doc_loaded = True
                    st.session_state.messages = []
                    st.session_state.history = ConversationHistory()
                    save_document_cache(file_name, full_text)
                    clear_conversation_cache()
                    st.rerun()
                except Exception as e:
                    st.error(f"文档加载失败：{e}")
else:
    if prompt := st.chat_input("输入你的问题，按 Enter 发送..."):

        # 1. 记录用户消息
        st.session_state.messages.append({
            "role": "user", "content": prompt, "sources": []
        })
        st.session_state.history.add_user(prompt)

        # 2. 渲染用户消息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 3. 流式生成回答（不再用 st.spinner 包裹，避免冲突）
        with st.chat_message("assistant"):
            status = st.status("思考中...", expanded=False)
            placeholder = st.empty()
            full_answer = ""

            try:
                history_ctx = st.session_state.history.get_context()
                stream = ask(
                    question=prompt,
                    full_text=st.session_state.full_text,
                    history=history_ctx,
                    stream=True,
                )

                # 逐 chunk 渲染 — 不阻塞、不 sleep
                for chunk in stream:
                    full_answer += chunk
                    placeholder.markdown(full_answer + "▌")

                # 流式完成，显示最终答案
                placeholder.markdown(full_answer)
                status.update(label="回答完成", state="complete")

                # 4. 来源引用
                sources = extract_sources(full_answer, st.session_state.full_text)
                if sources:
                    with st.expander("📖 来源引用"):
                        for i, src in enumerate(sources, 1):
                            st.info(f"**[{i}]** {src}")

            except Exception as e:
                status.update(label="生成失败", state="error")
                st.error(f"生成回答失败：{e}")
                logger.error(f"生成回答失败：{e}")
                full_answer = f"[错误] {e}"

        # 5. 保存助手消息（在 chat_message 块之后）
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "sources": sources if 'sources' in dir() else [],
        })
        st.session_state.history.add_assistant(full_answer)

        # 6. 持久化对话到磁盘
        save_conversation_cache(st.session_state.messages)

        # 7. 重载页面以显示完整历史
        st.rerun()


# ============================================================
# 导出功能
# ============================================================
if st.session_state.messages and st.session_state.doc_loaded:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 导出 Markdown", use_container_width=True):
            try:
                path = st.session_state.history.export_markdown()
                st.success(f"已导出到 {path}")
            except Exception as e:
                st.error(f"导出失败：{e}")

    with col2:
        if st.button("📥 导出 JSON", use_container_width=True):
            try:
                path = st.session_state.history.export_json()
                st.success(f"已导出到 {path}")
            except Exception as e:
                st.error(f"导出失败：{e}")

    with col3:
        st.caption(
            f"对话轮数：{len(st.session_state.history)} | "
            f"文档：{st.session_state.file_name}"
        )
