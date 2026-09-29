"""DashScope 原生文本向量封装。"""

import asyncio
from collections.abc import Mapping
from typing import Any, Sequence

from app.core.config import settings


class DashScopeEmbeddings:
    """使用 DashScope ``TextEmbedding`` 生成 1024 维向量。"""

    _MAX_DOCUMENT_BATCH_SIZE = 20

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ):
        self.model = model or settings.DASHSCOPE_EMBEDDING_MODEL
        self.dimension = 1024
        if client is None:
            import dashscope
            from dashscope import TextEmbedding

            dashscope.api_key = api_key or settings.DASHSCOPE_API_KEY
            self._client = TextEmbedding
        else:
            self._client = client

    async def embed_query(self, text: str) -> list[float]:
        """为检索查询生成向量。"""
        vectors = await self._embed(text, text_type="query")
        return vectors[0]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """为待索引文档生成向量。"""
        if not texts:
            return []

        documents = list(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(documents), self._MAX_DOCUMENT_BATCH_SIZE):
            batch = documents[start : start + self._MAX_DOCUMENT_BATCH_SIZE]
            vectors.extend(await self._embed(batch, text_type="document"))
        return vectors

    async def _embed(self, input_text: str | list[str], *, text_type: str) -> list[list[float]]:
        response = await asyncio.to_thread(
            self._client.call,
            model=self.model,
            input=input_text,
            text_type=text_type,
            dimension=self.dimension,
        )
        status_code = self._get_field(response, "status_code")
        if status_code is not None and status_code != 200:
            raise RuntimeError("DashScope text embedding request failed")

        output = self._get_field(response, "output")
        raw_embeddings = self._get_field(output, "embeddings")
        vectors = [self._get_field(item, "embedding") for item in raw_embeddings or []]
        if not vectors or any(vector is None for vector in vectors):
            raise RuntimeError("DashScope text embedding response is empty")
        return [list(vector) for vector in vectors]

    @staticmethod
    def _get_field(value: Any, field: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(field)
        return getattr(value, field, None)


class DashScopeSyncEmbeddings:
    """为要求同步 ``embed_query`` 的 Neo4j 检索器适配 DashScope。"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ):
        self.model = model or settings.DASHSCOPE_EMBEDDING_MODEL
        self.dimension = 1024
        if client is None:
            import dashscope
            from dashscope import TextEmbedding

            dashscope.api_key = api_key or settings.DASHSCOPE_API_KEY
            self._client = TextEmbedding
        else:
            self._client = client

    def embed_query(self, text: str) -> list[float]:
        response = self._client.call(
            model=self.model,
            input=text,
            text_type="query",
            dimension=self.dimension,
        )
        status_code = DashScopeEmbeddings._get_field(response, "status_code")
        if status_code is not None and status_code != 200:
            raise RuntimeError("DashScope text embedding request failed")

        output = DashScopeEmbeddings._get_field(response, "output")
        embeddings = DashScopeEmbeddings._get_field(output, "embeddings") or []
        embedding = DashScopeEmbeddings._get_field(embeddings[0], "embedding") if embeddings else None
        if embedding is None:
            raise RuntimeError("DashScope text embedding response is empty")
        return list(embedding)
