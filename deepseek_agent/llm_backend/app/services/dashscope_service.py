"""DashScope 原生聊天接口的流式 SSE 适配器。"""

import json
from collections.abc import AsyncGenerator, Mapping
from typing import Any, Optional

from app.core.config import settings


class DashScopeServiceError(RuntimeError):
    """向调用方暴露的安全 DashScope 异常，不包含上游响应内容。"""

    _MESSAGES = {
        "authentication": "DashScope authentication failed.",
        "rate_limit": "DashScope request rate limit exceeded.",
        "timeout": "DashScope request timed out.",
        "upstream": "DashScope upstream request failed.",
    }

    def __init__(self, category: str):
        self.category = category
        super().__init__(self._MESSAGES[category])


class DashScopeService:
    """将 DashScope ``Generation.call`` 的同步流转换为 SSE 事件流。"""

    def __init__(self, generation: Optional[Any] = None) -> None:
        self._generation = generation or self._load_generation()

    @staticmethod
    def _load_generation() -> Any:
        """仅在实际调用 SDK 时导入，便于测试注入替身。"""
        import dashscope
        from dashscope import Generation

        dashscope.base_http_api_url = settings.DASHSCOPE_BASE_URL
        return Generation

    async def generate_stream(
        self, messages: list[dict[str, Any]], *, thinking: bool = False
    ) -> AsyncGenerator[str, None]:
        """生成带 reasoning/content/done 事件的 SSE 数据流。"""
        model = (
            settings.DASHSCOPE_REASON_MODEL
            if thinking
            else settings.DASHSCOPE_CHAT_MODEL
        )

        try:
            chunks = self._generation.call(
                api_key=settings.DASHSCOPE_API_KEY,
                model=model,
                messages=messages,
                result_format="message",
                stream=True,
                incremental_output=True,
                enable_thinking=thinking,
            )
            for chunk in chunks:
                status_code = self._get_field(chunk, "status_code")
                if status_code is not None and status_code != 200:
                    raise DashScopeServiceError(
                        self._status_code_category(status_code)
                    )

                message = self._get_message(chunk)
                if message is None:
                    continue

                if thinking:
                    reasoning = self._get_field(message, "reasoning_content")
                    if reasoning:
                        yield self._sse_event("reasoning", reasoning)

                content = self._get_field(message, "content")
                if content:
                    yield self._sse_event("content", content)

            yield self._sse_event("done")
        except DashScopeServiceError:
            raise
        except Exception as error:
            raise DashScopeServiceError(self._error_category(error)) from None

    @staticmethod
    def _get_message(chunk: Any) -> Any:
        output = DashScopeService._get_field(chunk, "output")
        choices = DashScopeService._get_field(output, "choices")
        if not choices:
            return None
        return DashScopeService._get_field(choices[0], "message")

    @staticmethod
    def _get_field(value: Any, name: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    @staticmethod
    def _sse_event(event_type: str, content: Optional[str] = None) -> str:
        payload: dict[str, str] = {"type": event_type}
        if content is not None:
            payload["content"] = content
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @staticmethod
    def _error_category(error: Exception) -> str:
        status_code = getattr(error, "status_code", None)
        if status_code is not None:
            return DashScopeService._status_code_category(status_code)
        if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower():
            return "timeout"
        return "upstream"

    @staticmethod
    def _status_code_category(status_code: int) -> str:
        if status_code in (401, 403):
            return "authentication"
        if status_code == 429:
            return "rate_limit"
        if status_code in (408, 504):
            return "timeout"
        return "upstream"
