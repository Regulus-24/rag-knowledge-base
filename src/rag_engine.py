"""
RAG 引擎核心
- 基于 DeepSeek V4 1M 上下文窗口
- 全文直接推理 + 来源引用标注
- 支持流式输出
"""

from loguru import logger
from config import config
from src.llm_client import llm_client

# 引用标注提示词模板
CITATION_SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请严格根据用户提供的文档内容回答用户的问题。

要求：
1. 如果文档中有明确答案，请直接引用并总结
2. 如果文档中只有部分相关信息，请说明哪些是文档提到的、哪些不确定
3. 如果文档中完全没有相关信息，请如实告知："文档中未找到相关信息"
4. 回答使用中文，简洁明了
5. 在回答末尾，用【来源引用】标注你引用的原文片段（1-3条），格式：
   【来源引用】
   > 原文片段1
   > 原文片段2"""


def build_user_message(question: str, full_text: str, history: list[dict] | None = None) -> str:
    """构造发送给 LLM 的用户消息。"""
    parts = [f"=== 知识库文档全文 ===\n{full_text}"]

    if history:
        history_text = "\n".join(
            f"[{msg['role']}]: {msg['content'][:500]}" for msg in history[-6:]  # 最近 3 轮
        )
        parts.insert(0, f"=== 历史对话 ===\n{history_text}")

    parts.append(f"=== 用户问题 ===\n{question}")
    parts.append("=== 回答 ===")

    return "\n\n".join(parts)


def ask(
    question: str,
    full_text: str,
    history: list[dict] | None = None,
    stream: bool = False,
):
    """
    RAG 问答。

    Args:
        question: 用户问题
        full_text: 文档全文
        history: 对话历史（可选）
        stream: 是否流式返回

    Returns:
        stream=False: 回答文本（含来源引用）
        stream=True: 生成器，逐 chunk yield
    """
    logger.info(f"RAG 问答：question='{question[:50]}...' stream={stream}")
    user_message = build_user_message(question, full_text, history)

    if stream:
        return llm_client.ask_stream(CITATION_SYSTEM_PROMPT, user_message)
    else:
        return llm_client.ask(CITATION_SYSTEM_PROMPT, user_message)


def extract_sources(answer: str, full_text: str, max_sources: int = 3) -> list[str]:
    """
    从回答中提取来源引用片段。
    如果 LLM 已在回答中包含【来源引用】，则从中解析；
    否则尝试在原文中搜索相似片段。

    Returns:
        来源文本片段列表
    """
    # 检查 LLM 是否已生成引用
    if "【来源引用】" in answer:
        marker_idx = answer.index("【来源引用】")
        citation_block = answer[marker_idx:]
        lines = citation_block.split("\n")
        sources = [line.lstrip("> ").strip() for line in lines if line.startswith(">")]
        return sources[:max_sources]

    # 回退：从回答中提取关键句，在原文中匹配
    sentences = [s.strip() for s in answer.replace("\n", "").split("。") if len(s.strip()) > 10]
    sources = []
    for sent in sentences[:max_sources]:
        # 取前 30 字符作为搜索词
        keyword = sent[:30]
        if keyword in full_text:
            idx = full_text.index(keyword)
            snippet = full_text[max(0, idx - 20): idx + len(keyword) + 50]
            sources.append(snippet.strip())

    return sources[:max_sources]


def get_document_stats(full_text: str) -> dict:
    """获取文档统计信息，用于 UI 展示。"""
    lines = full_text.split("\n")
    return {
        "总字符数": len(full_text),
        "总行数": len(lines),
        "预估 tokens": len(full_text) // 2,
        "段落数": len([l for l in lines if l.strip()]),
    }
