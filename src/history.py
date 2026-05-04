"""
对话历史管理器
- 多轮对话上下文管理
- 自动截断（防止超出 token 限制）
- 导出功能
"""

import json
from datetime import datetime
from pathlib import Path
from loguru import logger
from config import config


class ConversationHistory:
    """管理单次会话的对话历史。"""

    def __init__(self, max_turns: int = 20):
        """
        Args:
            max_turns: 最大保留的对话轮数（超出后自动丢弃最早的）
        """
        self.messages: list[dict] = []
        self.max_turns = max_turns
        self.created_at = datetime.now().isoformat()

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self) -> None:
        """保留最近 N 轮对话（1 轮 = user + assistant 两条消息）。"""
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]
            logger.debug(f"对话历史已截断，保留最近 {self.max_turns} 轮")

    def get_context(self, include_system: bool = False) -> list[dict]:
        """返回消息列表，用于传入 LLM。"""
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()
        logger.info("对话历史已清除")

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "turns": len(self.messages) // 2,
            "messages": self.messages,
        }

    def export_json(self, file_path: str | Path | None = None) -> Path:
        """导出对话历史为 JSON 文件。"""
        path = Path(file_path) if file_path else config.EXPORT_DIR / f"history_{self._timestamp()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"对话历史已导出：{path}")
        return path

    def export_markdown(self, file_path: str | Path | None = None) -> Path:
        """导出对话历史为 Markdown 文件。"""
        path = Path(file_path) if file_path else config.EXPORT_DIR / f"history_{self._timestamp()}.md"
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# RAG 问答对话记录",
            f"",
            f"**时间**: {self.created_at}",
            f"**轮次**: {len(self.messages) // 2}",
            f"",
            "---",
            "",
        ]
        for i in range(0, len(self.messages), 2):
            q = self.messages[i]["content"] if i < len(self.messages) else ""
            a = self.messages[i + 1]["content"] if i + 1 < len(self.messages) else ""
            lines.append(f"### Q{i // 2 + 1}: {q[:80]}{'...' if len(q) > 80 else ''}")
            lines.append(f"")
            lines.append(a)
            lines.append(f"")
            lines.append("---")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"对话历史已导出（Markdown）：{path}")
        return path

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def __len__(self) -> int:
        return len(self.messages) // 2
