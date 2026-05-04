"""
磁盘缓存管理
- 文档持久化（刷新不丢）
- 对话历史持久化（刷新不丢）
- 启动时自动恢复
"""

import json
from pathlib import Path
from loguru import logger
from config import config


CACHE_DIR = config.DATA_DIR
DOC_CACHE_FILE = CACHE_DIR / ".doc_cache.json"
CONV_CACHE_FILE = CACHE_DIR / ".conv_cache.json"


# ============================================================
# 文档缓存
# ============================================================
def save_document_cache(file_name: str, full_text: str) -> None:
    """将文档信息保存到磁盘。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {"file_name": file_name, "full_text": full_text}
    DOC_CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    logger.info(f"文档已缓存：{file_name} ({len(full_text)} 字符)")


def load_document_cache() -> tuple[str, str] | None:
    """从磁盘恢复文档。返回 (file_name, full_text) 或 None。"""
    if not DOC_CACHE_FILE.exists():
        return None
    try:
        data = json.loads(DOC_CACHE_FILE.read_text(encoding="utf-8"))
        file_name = data.get("file_name", "")
        full_text = data.get("full_text", "")
        if full_text:
            logger.info(f"文档已恢复：{file_name} ({len(full_text)} 字符)")
            return file_name, full_text
    except Exception as e:
        logger.warning(f"文档缓存读取失败：{e}")
    return None


def clear_document_cache() -> None:
    """清除文档缓存。"""
    if DOC_CACHE_FILE.exists():
        DOC_CACHE_FILE.unlink()
        logger.info("文档缓存已清除")


# ============================================================
# 对话缓存
# ============================================================
def save_conversation_cache(messages: list[dict]) -> None:
    """将对话消息保存到磁盘。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CONV_CACHE_FILE.write_text(
        json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.debug(f"对话已缓存：{len(messages)} 条消息")


def load_conversation_cache() -> list[dict]:
    """从磁盘恢复对话消息。"""
    if not CONV_CACHE_FILE.exists():
        return []
    try:
        messages = json.loads(CONV_CACHE_FILE.read_text(encoding="utf-8"))
        if isinstance(messages, list):
            logger.info(f"对话已恢复：{len(messages)} 条消息")
            return messages
    except Exception as e:
        logger.warning(f"对话缓存读取失败：{e}")
    return []


def clear_conversation_cache() -> None:
    """清除对话缓存。"""
    if CONV_CACHE_FILE.exists():
        CONV_CACHE_FILE.unlink()
        logger.info("对话缓存已清除")
