"""通过已验证的检索上下文生成 RAG 回复。"""

import inspect
from collections.abc import AsyncGenerator, Mapping
from typing import Any

from loguru import logger

from app.core.sse import sse_event


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
            if not self.has_retrieval_context(documents):
                raise RAGRetrievalUnavailableError("No retrieval context is available.")
            prompt_messages = self.build_prompt_messages(messages, documents)
            async for event in self._get_model_service().generate_stream(
                prompt_messages, thinking=False
            ):
                yield event
            yield self._sources_event(documents)
        except RAGRetrievalUnavailableError:
            yield self._error_event("RAG retrieval is unavailable.")
        except Exception:
            logger.exception("RAG request failed")
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

    @staticmethod
    def has_retrieval_context(documents: list[Any]) -> bool:
        """线上基线只检查结果是否为空，不判断内容是否足够支持回答。"""
        return bool(documents)

    @classmethod
    def build_prompt_messages(
        cls, messages: list[dict[str, str]], documents: list[Any]
    ) -> list[dict[str, str]]:
        """线上生成和评测共用同一上下文 Prompt。"""
        return [{"role": "system", "content": cls._system_prompt(documents)}, *messages]

    @classmethod
    def _system_prompt(cls, documents: list[Any]) -> str:
        context = "\n\n".join(
            f"[资料 {number}｜来源：{cls._document_source(document)}]\n"
            f"{cls._document_text(document)}"
            for number, document in enumerate(documents, start=1)
        )
        return (
            "你是基于检索资料回答问题的客服助手。只能依据以下资料回答；"
            "资料不足时请明确说明。回答中可使用 [资料 N] 标识引用依据。\n\n检索资料：\n"
            f"{context}"
        )

    @staticmethod
    def _document_text(document: Any) -> str:
        if isinstance(document, Mapping):
            return str(document.get("content") or document.get("text") or "")
        return str(document)

    @staticmethod
    def _document_source(document: Any) -> str:
        if not isinstance(document, Mapping):
            return "检索资料"
        metadata = document.get("metadata")
        return str(
            document.get("source")
            or (metadata.get("source") if isinstance(metadata, Mapping) else None)
            or "检索资料"
        )

    @classmethod
    def _sources_event(cls, documents: list[Any]) -> str:
        sources = []
        for number, document in enumerate(documents, start=1):
            channels = (
                document.get("retrieval_channels", [])
                if isinstance(document, Mapping)
                else []
            )
            sources.append(
                {
                    "id": number,
                    "source": cls._document_source(document),
                    "retrieval_channels": list(channels or []),
                }
            )
        return sse_event("sources", sources=sources)

    @staticmethod
    def _error_event(message: str) -> str:
        return sse_event("error", message=message)
