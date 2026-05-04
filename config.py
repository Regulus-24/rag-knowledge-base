"""
RAG 知识库问答系统 — 集中配置模块

所有配置项通过环境变量覆盖，开发环境使用 .env 文件。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件（项目根目录）
load_dotenv(Path(__file__).parent / ".env")


class Config:
    """应用配置，所有值可通过同名的环境变量覆盖。"""

    # ============================================================
    # DeepSeek API
    # ============================================================
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

    # ============================================================
    # LLM 参数
    # ============================================================
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4096"))

    # ============================================================
    # 应用设置
    # ============================================================
    APP_TITLE: str = os.getenv("APP_TITLE", "RAG 知识库问答系统")
    APP_DESCRIPTION: str = "基于 DeepSeek V4 1M 上下文窗口的智能文档问答系统"

    # 文件上传限制（MB）
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "20"))

    # ============================================================
    # 路径（相对于项目根目录）
    # ============================================================
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = PROJECT_ROOT / os.getenv("DATA_DIR", "./data")
    EXPORT_DIR: Path = PROJECT_ROOT / os.getenv("EXPORT_DIR", "./exports")
    LOG_DIR: Path = PROJECT_ROOT / "logs"

    # ============================================================
    # 支持的文件格式
    # ============================================================
    SUPPORTED_FORMATS: dict = {
        ".pdf": "PDF 文档",
        ".txt": "纯文本",
        ".md": "Markdown",
        ".docx": "Word 文档",
    }

    # ============================================================
    # 系统提示词
    # ============================================================
    SYSTEM_PROMPT: str = """你是一个专业的知识库问答助手。请严格根据用户提供的文档内容回答用户的问题。

要求：
1. 如果文档中有明确答案，请直接引用并总结
2. 如果文档中只有部分相关信息，请说明哪些是文档提到的、哪些不确定
3. 如果文档中完全没有相关信息，请如实告知："文档中未找到相关信息"
4. 回答使用中文，简洁明了"""

    # ============================================================
    # 验证
    # ============================================================
    @classmethod
    def validate(cls) -> bool:
        """验证必要配置是否齐全。"""
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError(
                "未设置 DEEPSEEK_API_KEY。请在 .env 文件中配置，"
                "参考 .env.example 模板。"
            )
        return True


# 单例
config = Config()
