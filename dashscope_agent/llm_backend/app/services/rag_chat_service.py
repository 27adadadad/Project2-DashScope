"""通过已验证的检索上下文生成 RAG 回复。"""

import inspect
import json
from collections.abc import AsyncGenerator, Mapping
from typing import Any


class RAGRetrievalUnavailableError(RuntimeError):
    """当前部署没有可安全调用的 GraphRAG 查询适配器。"""


class RAGChatService:
    """把检索结果注入系统提示词，再交给 DashScope 流式生成。"""

    def __init__(self, retriever: Any | None = None, model_service: Any | None = None):
        self._retriever = retriever
        self._model_service = model_service

    async def generate_stream(
        self, messages: list[dict[str, str]], index_id: str
    ) -> AsyncGenerator[str, None]:
        try:
            query = self._last_user_message(messages)
            if not query:
                raise ValueError("A user message is required for RAG retrieval.")
            documents = await self._retrieve(query, index_id)
            if not documents:
                raise RAGRetrievalUnavailableError("No retrieval context is available.")
            prompt_messages = [
                {"role": "system", "content": self._system_prompt(documents)},
                *messages,
            ]
            async for event in self._get_model_service().generate_stream(
                prompt_messages, thinking=False
            ):
                yield event
        except RAGRetrievalUnavailableError:
            yield self._error_event("RAG retrieval is unavailable.")
        except Exception:
            yield self._error_event("RAG request could not be completed.")

    async def _retrieve(self, query: str, index_id: str) -> list[Any]:
        if self._retriever is None:
            raise RAGRetrievalUnavailableError("No GraphRAG retriever is configured.")
        retrieve = getattr(self._retriever, "retrieve", self._retriever)
        result = retrieve(query, index_id)
        if inspect.isawaitable(result):
            result = await result
        return list(result or [])

    def _get_model_service(self) -> Any:
        if self._model_service is None:
            from app.services.model_service_factory import ModelServiceFactory

            self._model_service = ModelServiceFactory.create_chat_service()
        return self._model_service

    @staticmethod
    def _last_user_message(messages: list[dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    @classmethod
    def _system_prompt(cls, documents: list[Any]) -> str:
        context = "\n\n".join(cls._document_text(document) for document in documents)
        return (
            "你是基于检索资料回答问题的客服助手。只能依据以下资料回答；"
            "资料不足时请明确说明。\n\n检索资料：\n"
            f"{context}"
        )

    @staticmethod
    def _document_text(document: Any) -> str:
        if isinstance(document, Mapping):
            return str(document.get("content") or document.get("text") or "")
        return str(document)

    @staticmethod
    def _error_event(message: str) -> str:
        return f"data: {json.dumps({'type': 'error', 'message': message}, ensure_ascii=False)}\n\n"
