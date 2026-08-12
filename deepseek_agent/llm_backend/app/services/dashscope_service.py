"""DashScope 多模态聊天接口的流式 SSE 适配器。"""

import asyncio
import inspect
import json
import logging
import threading
from collections.abc import AsyncGenerator, Callable, Mapping
from typing import Any, Optional

from app.core.config import settings


logger = logging.getLogger(__name__)


class DashScopeServiceError(RuntimeError):
    """面向客户端的安全 DashScope 异常，不包含上游响应内容。"""

    _MESSAGES = {
        "authentication": "DashScope authentication failed.",
        "rate_limit": "DashScope request rate limit exceeded.",
        "timeout": "DashScope request timed out.",
        "upstream": "DashScope upstream request failed.",
    }

    def __init__(self, category: str):
        self.category = category if category in self._MESSAGES else "upstream"
        super().__init__(self._MESSAGES[self.category])


class DashScopeService:
    """把 DashScope ``MultiModalConversation`` 的模型流转换为 SSE。"""

    def __init__(self, conversation_client: Optional[Any] = None) -> None:
        self._conversation_client = (
            conversation_client or self._load_multimodal_conversation()
        )

    @staticmethod
    def _load_multimodal_conversation() -> Any:
        """延迟导入 SDK，允许测试注入本地客户端替身。"""
        import dashscope
        from dashscope import MultiModalConversation

        dashscope.base_http_api_url = settings.DASHSCOPE_BASE_URL
        return MultiModalConversation

    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        user_id: Optional[int] = None,
        conversation_id: Optional[int] = None,
        on_complete: Optional[Callable[..., Any]] = None,
        *,
        thinking: bool = False,
    ) -> AsyncGenerator[str, None]:
        """生成 reasoning、content、error 与 done 四类 SSE 事件。"""
        model = (
            settings.DASHSCOPE_REASON_MODEL
            if thinking
            else settings.DASHSCOPE_CHAT_MODEL
        )
        request = {
            "api_key": settings.DASHSCOPE_API_KEY,
            "model": model,
            "messages": messages,
            "result_format": "message",
            "stream": True,
            "incremental_output": True,
            "enable_thinking": thinking,
        }
        response_parts: list[str] = []

        try:
            async for chunk in self._iterate_chunks(request):
                self._raise_for_error_chunk(chunk)
                message = self._get_message(chunk)
                if message is None:
                    continue

                if thinking:
                    reasoning = self._extract_text(
                        self._get_field(message, "reasoning_content")
                    )
                    if reasoning:
                        yield self._sse_event("reasoning", reasoning)

                content = self._extract_text(self._get_field(message, "content"))
                if content:
                    response_parts.append(content)
                    yield self._sse_event("content", content)

            response = "".join(response_parts)
            if on_complete is not None:
                await self._notify_complete(
                    on_complete, user_id, conversation_id, messages, response
                )
            yield self._sse_event("done")
        except Exception as error:
            service_error = self._as_service_error(error)
            logger.error("DashScope stream failed: category=%s", service_error.category)
            yield self._sse_event("error", message=str(service_error))

    async def generate_tool_response(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Any:
        """请求 Qwen 的单次工具调用决策，供搜索等服务使用。"""
        request = {
            "api_key": settings.DASHSCOPE_API_KEY,
            "model": settings.DASHSCOPE_CHAT_MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "result_format": "message",
            "stream": False,
        }
        try:
            response = await self._call_once(request)
            self._raise_for_error_chunk(response)
            output = self._get_field(response, "output")
            choices = self._get_field(output, "choices")
            if not choices:
                raise DashScopeServiceError("upstream")
            return choices[0]
        except Exception as error:
            raise self._as_service_error(error) from None

    async def generate_search_summary_stream(
        self,
        messages: list[dict[str, Any]],
        user_id: Optional[int] = None,
        conversation_id: Optional[int] = None,
        on_complete: Optional[Callable[..., Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """根据搜索上下文生成普通答案流。"""
        async for event in self.generate_stream(
            messages,
            user_id=user_id,
            conversation_id=conversation_id,
            on_complete=on_complete,
            thinking=False,
        ):
            yield event

    async def _iterate_chunks(
        self, request: dict[str, Any]
    ) -> AsyncGenerator[Any, None]:
        call = self._conversation_client.call
        if inspect.iscoroutinefunction(call):
            stream = await call(**request)
            if inspect.isawaitable(stream):
                stream = await stream
            if hasattr(stream, "__aiter__"):
                async for chunk in stream:
                    yield chunk
                return
            async for chunk in self._threaded_chunks(lambda: stream):
                yield chunk
            return

        async for chunk in self._threaded_chunks(lambda: call(**request)):
            yield chunk

    async def _call_once(self, request: dict[str, Any]) -> Any:
        call = self._conversation_client.call
        if inspect.iscoroutinefunction(call):
            response = await call(**request)
        else:
            response = await asyncio.to_thread(call, **request)
        if inspect.isawaitable(response):
            response = await response
        return response

    @staticmethod
    async def _threaded_chunks(
        stream_factory: Callable[[], Any],
    ) -> AsyncGenerator[Any, None]:
        """用单一后台线程执行同步网络调用及迭代，避免阻塞事件循环。"""
        event_loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def publish(kind: str, value: Any = None) -> None:
            try:
                event_loop.call_soon_threadsafe(queue.put_nowait, (kind, value))
            except RuntimeError:
                # 消费者取消后事件循环可能已关闭；后台线程无需再发布。
                return

        def produce() -> None:
            try:
                for chunk in stream_factory():
                    publish("chunk", chunk)
            except Exception as error:
                publish("error", error)
            finally:
                publish("done")

        threading.Thread(target=produce, name="dashscope-stream", daemon=True).start()
        while True:
            kind, value = await queue.get()
            if kind == "chunk":
                yield value
            elif kind == "error":
                raise value
            else:
                return

    @staticmethod
    async def _notify_complete(
        callback: Callable[..., Any],
        user_id: Optional[int],
        conversation_id: Optional[int],
        messages: list[dict[str, Any]],
        response: str,
    ) -> None:
        if inspect.iscoroutinefunction(callback):
            result = callback(user_id, conversation_id, messages, response)
        else:
            result = await asyncio.to_thread(
                callback, user_id, conversation_id, messages, response
            )
        if inspect.isawaitable(result):
            await result

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

    @classmethod
    def _extract_text(cls, value: Any) -> str:
        """统一提取 str、对象、字典和多模态内容块列表中的文本。"""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return "".join(cls._extract_text(item) for item in value)
        if isinstance(value, Mapping):
            for field in ("text", "content", "reasoning_content"):
                nested = value.get(field)
                if nested is not None:
                    return cls._extract_text(nested)
            return ""
        for field in ("text", "content", "reasoning_content"):
            nested = getattr(value, field, None)
            if nested is not None and nested is not value:
                return cls._extract_text(nested)
        return ""

    @staticmethod
    def _sse_event(
        event_type: str, content: Optional[str] = None, message: Optional[str] = None
    ) -> str:
        payload: dict[str, str] = {"type": event_type}
        if content is not None:
            payload["content"] = content
        if message is not None:
            payload["message"] = message
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @classmethod
    def _raise_for_error_chunk(cls, chunk: Any) -> None:
        status_code = cls._get_field(chunk, "status_code")
        if status_code is not None and status_code != 200:
            raise DashScopeServiceError(cls._status_code_category(status_code))

    @classmethod
    def _as_service_error(cls, error: Exception) -> DashScopeServiceError:
        if isinstance(error, DashScopeServiceError):
            return error
        return DashScopeServiceError(cls._error_category(error))

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
