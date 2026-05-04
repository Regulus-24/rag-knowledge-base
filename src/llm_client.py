"""
DeepSeek V4 API 客户端封装
- 非流式：带指数退避重试
- 流式：独立通道（generator 不兼容 retry）
"""

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger
from config import config


class DeepSeekClient:
    """DeepSeek V4 API 客户端，兼容 OpenAI SDK。"""

    def __init__(self):
        self._client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        logger.info(f"DeepSeek 客户端初始化：model={config.DEEPSEEK_MODEL}")

    # ---- 非流式（带重试） ----

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _chat_sync(self, messages: list[dict], temperature: float, max_tokens: int):
        """同步调用（支持重试）。"""
        return self._client.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def ask(self, system_prompt: str, user_message: str) -> str:
        """非流式问答，返回完整回答。"""
        logger.debug(f"发起问答请求，user_message 长度={len(user_message)}")
        response = self._chat_sync(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
        content = response.choices[0].message.content
        logger.debug(f"获得回答，长度={len(content)}")
        return content

    # ---- 流式（无重试） ----

    def _chat_stream(self, messages: list[dict], temperature: float, max_tokens: int):
        """流式调用（不重试 — generator 不兼容 retry 装饰器）。"""
        return self._client.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def ask_stream(self, system_prompt: str, user_message: str):
        """
        流式问答，逐 chunk yield 文本增量。

        用法：
            for chunk_text in client.ask_stream(sys_prompt, user_msg):
                yield chunk_text
        """
        logger.debug(f"发起流式问答，user_message 长度={len(user_message)}")
        try:
            response = self._chat_stream(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except GeneratorExit:
            # 正常中断（用户停止或页面关闭）
            logger.debug("流式输出被中断（GeneratorExit）")
        except Exception as e:
            logger.error(f"流式输出异常：{e}")
            yield f"\n\n[流式输出中断：{e}]"


# 全局单例
llm_client = DeepSeekClient()
