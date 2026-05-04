"""
多格式文档加载器
支持：PDF、TXT、Markdown、DOCX
"""

import re
from pathlib import Path
from loguru import logger
from config import config


def clean_text(text: str) -> str:
    """清理文本中的无效 Unicode 代理字符和多余空白。"""
    text = text.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
    text = re.sub(r'[\ud800-\udfff]', '', text)
    # 合并多余空行
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text.strip()


def load_pdf(file_path: str | Path) -> str:
    """加载 PDF 文件，返回全文。"""
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(file_path))
    documents = loader.load()
    logger.info(f"PDF 加载完成：{len(documents)} 页，路径={file_path}")

    full_text = "\n\n".join(doc.page_content for doc in documents)
    return clean_text(full_text)


def load_txt(file_path: str | Path) -> str:
    """加载 TXT/Markdown 文件，返回全文。"""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    logger.info(f"文本文件加载完成：{len(text)} 字符，路径={file_path}")
    return clean_text(text)


def load_docx(file_path: str | Path) -> str:
    """加载 DOCX 文件，返回全文。"""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("加载 DOCX 需要 python-docx 库。请执行：pip install python-docx")

    doc = Document(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)
    logger.info(f"DOCX 加载完成：{len(paragraphs)} 段落，路径={file_path}")
    return clean_text(full_text)


# 格式 → 加载函数 映射
LOADER_MAP = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".md": load_txt,   # Markdown 按纯文本处理
    ".docx": load_docx,
}


def load_document(file_path: str | Path) -> tuple[str, str]:
    """
    根据文件后缀自动选择合适的加载器。

    Args:
        file_path: 文件路径

    Returns:
        (file_name, full_text) 元组

    Raises:
        ValueError: 不支持的文件格式
        FileNotFoundError: 文件不存在
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    suffix = path.suffix.lower()
    if suffix not in LOADER_MAP:
        supported = ", ".join(config.SUPPORTED_FORMATS.keys())
        raise ValueError(f"不支持的文件格式 '{suffix}'，支持：{supported}")

    loader_fn = LOADER_MAP[suffix]
    full_text = loader_fn(path)

    # 粗略估算 token 数
    estimated_tokens = len(full_text) // 2
    logger.info(
        f"文档加载完成：{path.name} | {len(full_text)} 字符 | 预估 {estimated_tokens} tokens"
    )

    return path.name, full_text


def load_multiple_documents(file_paths: list[str | Path]) -> dict[str, str]:
    """
    批量加载多个文档。

    Returns:
        {file_name: full_text} 字典
    """
    results = {}
    for fp in file_paths:
        try:
            name, text = load_document(fp)
            results[name] = text
        except Exception as e:
            logger.error(f"加载失败 {fp}：{e}")
    logger.info(f"批量加载完成：{len(results)}/{len(file_paths)} 个文件成功")
    return results
