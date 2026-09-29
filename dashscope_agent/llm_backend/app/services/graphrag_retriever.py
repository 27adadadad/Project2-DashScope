"""面向用户隔离 GraphRAG 索引的检索适配器。"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import uuid
from array import array
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.graphrag_paths import resolve_graphrag_project_dir
from app.services.hybrid_retrieval import (
    HybridTextUnitIndex,
    HybridTextUnitRetriever,
    normalise_text_units,
)


# 用标准库日志，保持本模块可被离线脚本独立导入（不引入额外第三方依赖）
logger = logging.getLogger(__name__)

QueryRunner = Callable[[str, Path], str | Awaitable[str]]
TextUnitLoader = Callable[[Path], Sequence[dict[str, Any]] | Awaitable[Sequence[dict[str, Any]]]]
HybridRetrieverFactory = Callable[[Sequence[dict[str, Any]]], Any]

# 缓存格式版本：改动词向量/分词口径时必须递增，避免读到旧口径的索引
HYBRID_INDEX_CACHE_VERSION = 1
MAX_MEMORY_INDEXES = 8
_CACHE_SUBDIR = ("cache", "hybrid_index")


def corpus_fingerprint(units: Sequence[dict[str, str]], model: str) -> str:
    """按语料内容（而非文件名或时间戳）标识索引，避免复用陈旧向量。"""
    digest = hashlib.sha256()
    digest.update(f"v{HYBRID_INDEX_CACHE_VERSION}|{model}|".encode("utf-8"))
    for unit in units:
        digest.update(unit["id"].encode("utf-8"))
        digest.update(b"\x00")
        digest.update(unit["content"].encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()[:32]


def hybrid_index_cache_dir(
    project_dir: str | Path | None = None, data_dir_name: str | None = None
) -> Path:
    """语料索引缓存目录：索引服务预热与在线检索共用同一规则。"""
    resolved_project = resolve_graphrag_project_dir(
        project_dir or settings.GRAPHRAG_PROJECT_DIR
    )
    return resolved_project.joinpath(
        data_dir_name or settings.GRAPHRAG_DATA_DIR, *_CACHE_SUBDIR
    )


def _read_cached_index(cache_dir: Path, fingerprint: str) -> HybridTextUnitIndex | None:
    """读取磁盘缓存的语料索引；任何不可用情况都退回重新构建。"""
    meta_path = cache_dir / f"{fingerprint}.json"
    vector_path = cache_dir / f"{fingerprint}.bin"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("version") != HYBRID_INDEX_CACHE_VERSION:
            return None
        if meta.get("model") != settings.DASHSCOPE_EMBEDDING_MODEL:
            return None
        units = normalise_text_units(meta["units"])
        positions = [int(position) for position in meta["vector_positions"]]
        dimension = int(meta["dimension"])
        if not positions or dimension <= 0:
            return None

        raw = array("f")
        raw.frombytes(vector_path.read_bytes())
        if len(raw) != len(positions) * dimension:
            return None

        vectors: list[Sequence[float] | None] = [None] * len(units)
        for offset, position in enumerate(positions):
            start = offset * dimension
            vectors[position] = list(raw[start : start + dimension])
    except Exception:
        logger.warning(
            "混合检索索引缓存不可用，将重新构建：%s", fingerprint, exc_info=True
        )
        return None
    return HybridTextUnitIndex(units, document_vectors=vectors)


def _write_cached_index(
    cache_dir: Path, fingerprint: str, index: HybridTextUnitIndex
) -> None:
    """把文档向量与其语料一起落盘，供重启后或多进程直接复用。"""
    positions = [
        position
        for position, vector in enumerate(index.document_vectors)
        if vector is not None
    ]
    if not positions:
        return
    dimension = len(index.document_vectors[positions[0]])
    if any(len(index.document_vectors[position]) != dimension for position in positions):
        logger.warning("文档向量维度不一致，跳过写入缓存：%s", fingerprint)
        return

    flattened = array("f")
    for position in positions:
        flattened.extend(float(value) for value in index.document_vectors[position])
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{fingerprint}.bin").write_bytes(flattened.tobytes())
        (cache_dir / f"{fingerprint}.json").write_text(
            json.dumps(
                {
                    "version": HYBRID_INDEX_CACHE_VERSION,
                    "model": settings.DASHSCOPE_EMBEDDING_MODEL,
                    "dimension": dimension,
                    "units": list(index.units),
                    "vector_positions": positions,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        logger.warning("混合检索索引缓存写入失败：%s", fingerprint, exc_info=True)


class GraphRAGRetriever:
    """将安全的索引标识映射到单个用户的 GraphRAG 输出目录。"""

    def __init__(
        self,
        project_dir: str | Path | None = None,
        data_dir_name: str | None = None,
        query_runner: QueryRunner | None = None,
        text_unit_loader: TextUnitLoader | None = None,
        hybrid_retriever_factory: HybridRetrieverFactory | None = None,
        memory_cache: dict[str, HybridTextUnitIndex] | None = None,
    ) -> None:
        self.project_dir = resolve_graphrag_project_dir(
            project_dir or settings.GRAPHRAG_PROJECT_DIR
        )
        self.data_dir_name = data_dir_name or settings.GRAPHRAG_DATA_DIR
        self._query_runner = query_runner or self._run_local_search
        self._text_unit_loader = text_unit_loader or self._load_text_units
        self._hybrid_retriever_factory = (
            hybrid_retriever_factory or self._create_hybrid_retriever
        )
        self._memory_cache = memory_cache if memory_cache is not None else _SHARED_INDEX_CACHE

    @property
    def cache_dir(self) -> Path:
        """语料索引缓存目录；由内容指纹命名，因此不同用户不会相互串用。"""
        return hybrid_index_cache_dir(self.project_dir, self.data_dir_name)

    def output_dir_for(self, index_id: str) -> Path:
        """校验 UUID，并返回其唯一允许访问的 GraphRAG 输出目录。"""
        try:
            normalized_index_id = str(uuid.UUID(index_id))
        except (ValueError, AttributeError, TypeError) as error:
            raise ValueError("Invalid index id") from error
        return self.project_dir / self.data_dir_name / "output" / normalized_index_id

    async def retrieve(self, query: str, index_id: str) -> list[dict[str, str]]:
        output_dir = self.output_dir_for(index_id)
        if not output_dir.is_dir():
            raise ValueError("GraphRAG index does not exist")

        documents: list[dict[str, Any]] = []
        try:
            response = self._query_runner(query, output_dir)
            if inspect.isawaitable(response):
                response = await response
            text = str(response or "").strip()
            if text:
                documents.append(
                    {
                        "content": text,
                        "source": "GraphRAG Local Search",
                        "retrieval_channels": ["graph"],
                    }
                )
        except Exception:
            logger.warning("GraphRAG Local Search 通道失败，已跳过", exc_info=True)

        try:
            text_units = self._text_unit_loader(output_dir)
            if inspect.isawaitable(text_units):
                text_units = await text_units
            hybrid_retriever = self._hybrid_retriever_factory(text_units)
            if inspect.isawaitable(hybrid_retriever):
                hybrid_retriever = await hybrid_retriever
            evidence = hybrid_retriever.retrieve(query)
            if inspect.isawaitable(evidence):
                evidence = await evidence
            documents.extend(evidence or [])
        except Exception:
            logger.warning("混合召回通道失败，已跳过", exc_info=True)

        return documents

    async def _create_hybrid_retriever(self, text_units: Sequence[dict[str, Any]]) -> Any:
        """复用缓存索引构造检索器，避免每次请求重新向量化整份语料。"""
        units = normalise_text_units(text_units)
        if not units:
            return HybridTextUnitRetriever(index=HybridTextUnitIndex(()))

        fingerprint = corpus_fingerprint(units, settings.DASHSCOPE_EMBEDDING_MODEL)
        index = self._remembered_index(fingerprint)
        if index is None:
            index = _read_cached_index(self.cache_dir, fingerprint)
            if index is None:
                index = await self._build_hybrid_index(units, fingerprint)
            if index.has_dense_channel:
                self._remember_index(fingerprint, index)

        from app.services.dashscope_embeddings import DashScopeEmbeddings

        return HybridTextUnitRetriever(
            index=index, embed_query=DashScopeEmbeddings().embed_query
        )

    async def _build_hybrid_index(
        self, units: Sequence[dict[str, str]], fingerprint: str
    ) -> HybridTextUnitIndex:
        """构建文档向量并落盘；失败时记录日志并退化为纯 BM25。"""
        from app.services.dashscope_embeddings import DashScopeEmbeddings

        embeddings = DashScopeEmbeddings()
        try:
            index = await HybridTextUnitIndex.build(
                units, embeddings.embed_documents, strict=True
            )
        except Exception:
            logger.warning("文档向量构建失败，本次仅使用 BM25 通道", exc_info=True)
            return HybridTextUnitIndex(units)
        _write_cached_index(self.cache_dir, fingerprint, index)
        return index

    def _remembered_index(self, fingerprint: str) -> HybridTextUnitIndex | None:
        index = self._memory_cache.pop(fingerprint, None)
        if index is not None:
            self._memory_cache[fingerprint] = index
        return index

    def _remember_index(self, fingerprint: str, index: HybridTextUnitIndex) -> None:
        self._memory_cache[fingerprint] = index
        while len(self._memory_cache) > MAX_MEMORY_INDEXES:
            self._memory_cache.pop(next(iter(self._memory_cache)))

    @staticmethod
    async def _load_text_units(output_dir: Path) -> Sequence[dict[str, Any]]:
        from graphrag.storage.file_pipeline_storage import FilePipelineStorage
        from graphrag.utils.storage import load_table_from_storage

        storage = FilePipelineStorage(root_dir=str(output_dir))
        text_units = await load_table_from_storage("text_units", storage)
        return text_units.to_dict("records")

    async def _run_local_search(self, query: str, output_dir: Path) -> str:
        """在指定用户输出目录加载 GraphRAG 产物并执行本地查询。"""
        import graphrag.api as api
        from graphrag.config.load_config import load_config
        from graphrag.storage.file_pipeline_storage import FilePipelineStorage
        from graphrag.utils.storage import load_table_from_storage

        data_dir = self.project_dir / self.data_dir_name
        config = load_config(
            data_dir,
            None,
            {"output.base_dir": str(output_dir)},
        )
        storage = FilePipelineStorage(root_dir=str(output_dir))
        entities, communities, community_reports, text_units, relationships = (
            await load_table_from_storage(name, storage)
            for name in (
                "entities",
                "communities",
                "community_reports",
                "text_units",
                "relationships",
            )
        )
        try:
            covariates = await load_table_from_storage("covariates", storage)
        except ValueError:
            covariates = None

        response, _context = await api.local_search(
            config=config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            text_units=text_units,
            relationships=relationships,
            covariates=covariates,
            community_level=settings.GRAPHRAG_COMMUNITY_LEVEL,
            response_type=settings.GRAPHRAG_RESPONSE_TYPE,
            query=query,
        )
        return str(response)


_SHARED_INDEX_CACHE: dict[str, HybridTextUnitIndex] = {}


def _remember_shared_index(fingerprint: str, index: HybridTextUnitIndex) -> None:
    _SHARED_INDEX_CACHE[fingerprint] = index
    while len(_SHARED_INDEX_CACHE) > MAX_MEMORY_INDEXES:
        _SHARED_INDEX_CACHE.pop(next(iter(_SHARED_INDEX_CACHE)))


async def warm_text_unit_index(output_dir: str | Path) -> bool:
    """索引构建完成后预计算并持久化文本块向量，供后续问答直接复用。

    返回是否成功写入缓存；失败只记录日志，不影响索引本身是否可用。
    """
    from graphrag.storage.file_pipeline_storage import FilePipelineStorage
    from graphrag.utils.storage import load_table_from_storage

    from app.services.dashscope_embeddings import DashScopeEmbeddings

    try:
        storage = FilePipelineStorage(root_dir=str(output_dir))
        table = await load_table_from_storage("text_units", storage)
        units = normalise_text_units(table.to_dict("records"))
        if not units:
            return False
        fingerprint = corpus_fingerprint(units, settings.DASHSCOPE_EMBEDDING_MODEL)
        index = await HybridTextUnitIndex.build(
            units, DashScopeEmbeddings().embed_documents, strict=True
        )
        if not index.has_dense_channel:
            return False
        _write_cached_index(hybrid_index_cache_dir(), fingerprint, index)
        _remember_shared_index(fingerprint, index)
        return True
    except Exception:
        logger.warning("混合检索索引预热失败，将在首次问答时构建", exc_info=True)
        return False
