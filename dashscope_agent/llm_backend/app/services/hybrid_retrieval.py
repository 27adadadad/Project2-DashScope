"""文本块的稠密检索、BM25 检索及 RRF 融合。

这里刻意区分两类计算：
- 只依赖语料的量：文档向量、向量范数、BM25 倒排统计量，集中在
  ``HybridTextUnitIndex`` 中构建一次，可跨请求复用并持久化；
- 只依赖查询的量：查询向量、查询分词、打分与排名融合，每次请求才计算。

这样单次问答的向量计算由 O(语料量) 降为 O(1)，BM25 由 O(文档数×词数)
降为 O(命中文档数)。
"""

from __future__ import annotations

import inspect
import math
import re
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Sequence
from typing import Any


EmbeddingCallable = Callable[[Any], Any | Awaitable[Any]]

# Okapi BM25 参数，与项目既有评测基线保持一致；改动会使历史指标失去可比性
BM25_K1 = 1.5
BM25_B = 0.75
BM25_K3_PLUS_ONE = 2.5


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """用排序而非不可比较的原始分数融合多个召回通道。"""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] += 1 / (k + rank)
    return [
        document_id
        for document_id, _score in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def tokenize(text: str) -> list[str]:
    """保留英文术语，同时将连续中文拆为可工作的字符级 token。"""
    return re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())


def normalise_text_unit(text_unit: dict[str, Any]) -> dict[str, str]:
    """把 GraphRAG 的原始 text_unit 收敛为稳定的 id/content/source 结构。"""
    content = str(text_unit.get("text") or text_unit.get("content") or "").strip()
    return {
        "id": str(text_unit.get("id") or text_unit.get("human_readable_id") or content),
        "content": content,
        "source": str(text_unit.get("document_id") or text_unit.get("source") or "未知文档"),
    }


def normalise_text_units(text_units: Sequence[dict[str, Any]] | None) -> list[dict[str, str]]:
    """批量归一化，且对已归一化的结果可重复调用（幂等）。"""
    return [normalise_text_unit(unit) for unit in (text_units or [])]


def _vector_norm(vector: Sequence[float] | None) -> float | None:
    """返回 L2 范数；空向量、非有限值或零范数返回 None 表示该向量不可用。"""
    try:
        if vector is None or not len(vector):
            return None
        values = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in values):
            return None
        norm = math.sqrt(sum(value * value for value in values))
        return norm if norm else None
    except (TypeError, ValueError):
        return None


def _cosine_similarity(
    left_values: Sequence[float],
    left_norm: float | None,
    right: Sequence[float] | None,
    right_norm: float | None,
    *,
    strict: bool = False,
) -> float | None:
    """在已预计算范数的前提下算余弦相似度；不可用时按 strict 决定抛错或返回 None。"""
    try:
        if left_norm is None or right is None or right_norm is None:
            raise ValueError("Embedding vectors must be non-empty with matching dimensions")
        right_values = [float(value) for value in right]
        if len(left_values) != len(right_values):
            raise ValueError("Embedding vectors must be non-empty with matching dimensions")
        dot_product = sum(a * b for a, b in zip(left_values, right_values))
        if not math.isfinite(dot_product):
            raise ValueError("Embedding similarity calculation is non-finite")
        score = dot_product / (left_norm * right_norm)
        if not math.isfinite(score):
            raise ValueError("Embedding similarity is non-finite")
        return score
    except (TypeError, ValueError, OverflowError) as error:
        if strict:
            raise ValueError(str(error)) from error
        return None


async def _call(function: EmbeddingCallable, value: Any) -> Any:
    result = function(value)
    return await result if inspect.isawaitable(result) else result


class HybridTextUnitIndex:
    """语料级检索索引：文档向量、向量范数与 BM25 倒排表。

    这些量只依赖语料，因此预期由索引阶段或首个请求构建一次并长期复用。
    """

    def __init__(
        self,
        units: Sequence[dict[str, str]],
        *,
        document_vectors: Sequence[Sequence[float] | None] | None = None,
    ) -> None:
        self.units: tuple[dict[str, str], ...] = tuple(units)
        self.document_vectors: tuple[Sequence[float] | None, ...] = (
            tuple(document_vectors) if document_vectors is not None else ()
        )
        self.document_norms = tuple(
            _vector_norm(vector) for vector in self.document_vectors
        )
        self._postings, self._document_lengths = self._build_postings()
        count = len(self._document_lengths)
        self._average_length = (sum(self._document_lengths) / count) if count else 0.0

    @property
    def has_dense_channel(self) -> bool:
        """文档向量是否可用；不可用时只保留 BM25 通道。"""
        return bool(self.document_vectors)

    @classmethod
    async def build(
        cls,
        text_units: Sequence[dict[str, Any]],
        embed_documents: EmbeddingCallable | None = None,
        *,
        strict: bool = False,
    ) -> "HybridTextUnitIndex":
        """构建语料索引；strict=True 时把向量化失败原样抛出，便于调用方记录。"""
        units = normalise_text_units(text_units)
        if embed_documents is None or not units:
            return cls(units)
        try:
            vectors = await _call(embed_documents, [unit["content"] for unit in units])
            if vectors is None or len(vectors) != len(units):
                raise ValueError("Embedding document count does not match the corpus")
            vectors = list(vectors)
        except Exception:
            if strict:
                raise
            return cls(units)
        return cls(units, document_vectors=vectors)

    def dense_scores(
        self, query_vector: Sequence[float] | None, *, strict: bool = False
    ) -> list[tuple[str, float]]:
        """用预计算的文档向量与范数打分，按分数降序、id 升序返回。"""
        if not self.document_vectors:
            return []
        query_norm = _vector_norm(query_vector)
        if query_norm is None:
            if strict:
                raise ValueError(
                    "Embedding vectors must be non-empty with matching dimensions"
                )
            return []
        query_values = [float(value) for value in query_vector]
        scored: list[tuple[str, float]] = []
        for position, unit in enumerate(self.units):
            score = _cosine_similarity(
                query_values,
                query_norm,
                self.document_vectors[position],
                self.document_norms[position],
                strict=strict,
            )
            if score is not None:
                scored.append((unit["id"], score))
        return sorted(scored, key=lambda item: (-item[1], item[0]))

    def bm25_scores(self, query_tokens: Sequence[str]) -> list[tuple[str, float]]:
        """借助倒排表只遍历命中文档，等价于原全语料实现但复杂度更低。"""
        if not query_tokens or not self.units:
            return []
        total_documents = len(self.units)
        average_length = max(self._average_length, 1.0)
        scores: dict[int, float] = defaultdict(float)
        for token, query_count in Counter(query_tokens).items():
            posting = self._postings.get(token)
            if not posting:
                continue
            document_frequency = len(posting)
            inverse_frequency = math.log(
                1
                + (total_documents - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for position, frequency in posting:
                length = self._document_lengths[position]
                denominator = frequency + BM25_K1 * (
                    1 - BM25_B + BM25_B * length / average_length
                )
                scores[position] += (
                    query_count * inverse_frequency * frequency * BM25_K3_PLUS_ONE
                ) / denominator
        return sorted(
            ((self.units[position]["id"], score) for position, score in scores.items()),
            key=lambda item: (-item[1], item[0]),
        )

    def _build_postings(self) -> tuple[dict[str, tuple[tuple[int, int], ...]], tuple[int, ...]]:
        """构建 词 -> ((文档位置, 词频), ...) 的倒排表，并记录每篇文档长度。"""
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths: list[int] = []
        for position, unit in enumerate(self.units):
            tokens = tokenize(unit["content"])
            lengths.append(len(tokens))
            for token, frequency in Counter(tokens).items():
                postings[token].append((position, frequency))
        return (
            {token: tuple(entries) for token, entries in postings.items()},
            tuple(lengths),
        )


class HybridTextUnitRetriever:
    """在同一组 GraphRAG 文本块上执行向量和 BM25 混合召回。"""

    def __init__(
        self,
        text_units: Sequence[dict[str, Any]] | None = None,
        embed_query: EmbeddingCallable | None = None,
        embed_documents: EmbeddingCallable | None = None,
        *,
        index: HybridTextUnitIndex | None = None,
        candidate_limit: int = 20,
        rrf_k: int = 60,
    ) -> None:
        # index 由调用方预构建（生产路径）；否则用 text_units 现场建 BM25 部分
        self._index = index or HybridTextUnitIndex(normalise_text_units(text_units))
        self._embed_query = embed_query
        self._embed_documents = embed_documents
        self._candidate_limit = candidate_limit
        self._rrf_k = rrf_k
        self._build_error: Exception | None = None

    async def retrieve(
        self, query: str, *, limit: int = 8, strict: bool = False
    ) -> list[dict[str, Any]]:
        """返回融合文本块；strict 模式使评测能显式记录 Dense 通道失败。"""
        query = query.strip()
        if not query or not self._index.units or limit <= 0:
            return []

        index = await self._ensure_index(strict=strict)
        dense_scored = await self._dense_scores(query, index=index, strict=strict)
        dense_scores = dict(dense_scored)
        dense_ranking = [unit_id for unit_id, _ in dense_scored[: self._candidate_limit]]
        bm25_ranking = self._bm25_ranking(query, index=index)
        ranked_ids = reciprocal_rank_fusion(
            [ranking for ranking in (dense_ranking, bm25_ranking) if ranking],
            k=self._rrf_k,
        )
        units_by_id = {unit["id"]: unit for unit in index.units}
        dense_positions = {value: rank for rank, value in enumerate(dense_ranking, 1)}
        bm25_positions = {value: rank for rank, value in enumerate(bm25_ranking, 1)}

        results = []
        for unit_id in ranked_ids[:limit]:
            unit = units_by_id[unit_id]
            dense_rank = dense_positions.get(unit_id)
            bm25_rank = bm25_positions.get(unit_id)
            channels = [
                channel
                for channel, rank in (("dense", dense_rank), ("bm25", bm25_rank))
                if rank is not None
            ]
            rrf_score = sum(
                1 / (self._rrf_k + rank)
                for rank in (dense_rank, bm25_rank)
                if rank is not None
            )
            results.append(
                {
                    "id": unit_id,
                    "content": unit["content"],
                    "source": unit["source"],
                    "dense_score": dense_scores.get(unit_id),
                    "dense_rank": dense_rank,
                    "bm25_rank": bm25_rank,
                    "rrf_score": rrf_score,
                    "retrieval_channels": channels,
                }
            )
        return results

    async def retrieve_dense(
        self, query: str, *, limit: int = 8, strict: bool = False
    ) -> list[dict[str, Any]]:
        """使用与混合通道相同的余弦打分和排序，供 Dense 对照使用。"""
        query = query.strip()
        if not query or not self._index.units or limit <= 0:
            return []
        index = await self._ensure_index(strict=strict)
        scored = await self._dense_scores(query, index=index, strict=strict)
        units_by_id = {unit["id"]: unit for unit in index.units}
        return [
            {
                **units_by_id[unit_id],
                "dense_score": score,
                "dense_rank": rank,
                "retrieval_channels": ["dense"],
            }
            for rank, (unit_id, score) in enumerate(scored[:limit], 1)
        ]

    async def _ensure_index(self, *, strict: bool) -> HybridTextUnitIndex:
        """文档向量只构建一次；失败不缓存，使后续请求可以重试。"""
        if self._index.has_dense_channel or self._embed_documents is None:
            return self._index
        if self._build_error is not None and not strict:
            return self._index
        try:
            # 内部统一用 strict=True，把失败转成可记录的异常，避免静默退化为纯 BM25
            built = await HybridTextUnitIndex.build(
                self._index.units, self._embed_documents, strict=True
            )
        except Exception as error:
            self._build_error = error
            if strict:
                raise
            return self._index
        self._build_error = None
        self._index = built
        return self._index

    async def _dense_scores(
        self,
        query: str,
        *,
        index: HybridTextUnitIndex,
        strict: bool = False,
    ) -> list[tuple[str, float]]:
        if self._embed_query is None:
            return []
        try:
            query_vector = await _call(self._embed_query, query)
        except Exception:
            if strict:
                raise
            return []
        return index.dense_scores(query_vector, strict=strict)

    def _bm25_ranking(self, query: str, *, index: HybridTextUnitIndex) -> list[str]:
        ranking = [unit_id for unit_id, _ in index.bm25_scores(tokenize(query))]
        return ranking[: self._candidate_limit]
