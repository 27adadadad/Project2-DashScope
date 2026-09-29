"""基于语义相似度的 Redis 缓存。

设计要点：
- 使用 ``redis.asyncio``：同步客户端在事件循环里没有切换点，会阻塞整个进程的并发请求；
- 客户端进程级共享：每次请求新建连接池会快速耗尽 Redis 连接数；
- 用 ``SCAN`` 而非 ``KEYS``：``KEYS`` 是阻塞式全库扫描，会卡住 Redis 主线程；
- 向量存 float32 原始字节而非 JSON 文本，体积与解析成本都更低；
- 候选向量一次 ``MGET`` 批量取回，避免「一个缓存项一次往返」。
"""

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logger import get_logger
from app.services.dashscope_embeddings import DashScopeEmbeddings

logger = get_logger(service="redis_cache")

_VECTOR_DTYPE = np.float32
_shared_client: Optional[aioredis.Redis] = None


def get_shared_redis_client() -> aioredis.Redis:
    """返回进程级共享的异步 Redis 客户端（惰性创建，不在此处建立连接）。"""
    global _shared_client
    if _shared_client is None:
        _shared_client = aioredis.from_url(settings.REDIS_URL)
    return _shared_client


class RedisSemanticCache:
    """基于语义的 Redis 缓存实现（按用户前缀隔离）。"""

    def __init__(
        self,
        redis_url: str = None,
        model_name: str = None,
        score_threshold: float = None,
        prefix: str = "cache",
        user_id: Optional[int] = None,
        client: Any = None,
        max_cache_size: int = 1000,
        cleanup_interval: int = 3600,
        start_auto_cleanup: bool = False,
    ):
        if client is not None:
            self.redis = client
        elif redis_url:
            self.redis = aioredis.from_url(redis_url)
        else:
            # 主链路使用共享客户端，避免每个请求都新建连接池
            self.redis = get_shared_redis_client()
        self.model_name = model_name or settings.DASHSCOPE_EMBEDDING_MODEL
        self.embeddings = DashScopeEmbeddings(model=self.model_name)
        self.score_threshold = score_threshold or settings.REDIS_CACHE_THRESHOLD
        self.prefix = f"{prefix}:{user_id}" if user_id else prefix
        self.max_cache_size = max_cache_size
        self.cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None

        if start_auto_cleanup:
            self._cleanup_task = asyncio.create_task(self._auto_cleanup())

    async def _get_embedding(self, text: str) -> List[float]:
        """获取文本向量。"""
        embedding = await self.embeddings.embed_query(text)
        if not embedding:
            raise ValueError("Failed to get embedding")
        return embedding

    @staticmethod
    def _hash(message: str) -> str:
        return hashlib.md5(message.encode()).hexdigest()

    def _get_vector_key(self, message: str) -> str:
        return f"{self.prefix}:vec:{self._hash(message)}"

    def _get_response_key(self, message: str) -> str:
        return f"{self.prefix}:resp:{self._hash(message)}"

    def _get_metadata_key(self, message: str) -> str:
        return f"{self.prefix}:meta:{self._hash(message)}"

    def _get_last_user_message(self, messages: List[Dict]) -> str:
        """获取最后一条用户消息。"""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    async def _scan_keys(self, pattern: str) -> List[str]:
        """用 SCAN 增量遍历键，避免 KEYS 阻塞 Redis 主线程。"""
        keys: List[str] = []
        async for key in self.redis.scan_iter(match=pattern, count=200):
            keys.append(key.decode("utf-8") if isinstance(key, bytes) else key)
        return keys

    async def _auto_cleanup(self):
        """缓存条数超过上限时按最后访问时间淘汰；仅显式开启时运行。"""
        while True:
            try:
                meta_keys = await self._scan_keys(f"{self.prefix}:meta:*")
                if len(meta_keys) > self.max_cache_size:
                    metadata = await self.redis.mget(meta_keys)
                    items = sorted(
                        zip(meta_keys, metadata), key=lambda item: self._last_access(item[1])
                    )
                    for key, _ in items[: len(meta_keys) - self.max_cache_size]:
                        await self._remove_cache_item(key.split(":")[-1])
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("缓存清理失败", exc_info=True)

            await asyncio.sleep(self.cleanup_interval)

    @staticmethod
    def _last_access(raw_metadata: Any) -> float:
        if not raw_metadata:
            return 0.0
        try:
            return float(json.loads(raw_metadata)["last_access"])
        except (ValueError, KeyError, TypeError):
            return 0.0

    async def _remove_cache_item(self, hash_id: str):
        """删除一个缓存项的所有相关键。"""
        await self.redis.delete(
            f"{self.prefix}:vec:{hash_id}",
            f"{self.prefix}:resp:{hash_id}",
            f"{self.prefix}:meta:{hash_id}",
        )

    async def _update_metadata(self, message: str):
        """更新缓存项的元数据（访问次数与最后访问时间）。"""
        meta_key = self._get_metadata_key(message)
        current_meta = await self.redis.get(meta_key)
        if current_meta:
            try:
                access_count = int(json.loads(current_meta)["access_count"]) + 1
            except (ValueError, KeyError, TypeError):
                access_count = 1
        else:
            access_count = 1

        metadata = {
            "last_access": time.monotonic(),
            "access_count": access_count,
        }
        await self.redis.set(
            meta_key, json.dumps(metadata), ex=settings.REDIS_CACHE_EXPIRE
        )

    async def lookup(self, messages: List[Dict]) -> Optional[str]:
        """查找语义最接近的缓存回答；任何故障都降级为未命中。"""
        try:
            user_message = self._get_last_user_message(messages)
            if not user_message:
                return None

            current_vector = np.asarray(await self._get_embedding(user_message), dtype=_VECTOR_DTYPE)
            vector_keys = await self._scan_keys(f"{self.prefix}:vec:*")
            if not vector_keys:
                return None

            # 一次 MGET 取回全部候选向量，避免逐键往返
            payloads = await self.redis.mget(vector_keys)
            best_similarity = 0.0
            best_key: Optional[str] = None
            for key, payload in zip(vector_keys, payloads):
                similarity = self._cosine(current_vector, payload)
                if similarity is not None and similarity > best_similarity:
                    best_similarity = similarity
                    best_key = key

            if best_key is None or best_similarity < self.score_threshold:
                return None

            hash_id = best_key.split(":")[-1]
            cached_response = await self.redis.get(f"{self.prefix}:resp:{hash_id}")
            if not cached_response:
                return None

            await self._update_metadata(user_message)
            logger.info(f"Cache hit with similarity: {best_similarity:.4f}")
            return cached_response.decode("utf-8")

        except Exception:
            logger.warning("语义缓存查询失败，已降级为直接调用模型", exc_info=True)
            return None

    async def update(self, messages: List[Dict], response: str, expire: int = None):
        """写入缓存项（向量 + 回答 + 元数据），失败只记录日志。"""
        try:
            user_message = self._get_last_user_message(messages)
            if not user_message or not response:
                return

            vector = await self._get_embedding(user_message)
            expire = expire or settings.REDIS_CACHE_EXPIRE
            now = time.monotonic()

            # 用 pipeline 合并三次写入，减少网络往返
            async with self.redis.pipeline(transaction=False) as pipe:
                pipe.set(
                    self._get_vector_key(user_message),
                    np.asarray(vector, dtype=_VECTOR_DTYPE).tobytes(),
                    ex=expire,
                )
                pipe.set(self._get_response_key(user_message), response, ex=expire)
                pipe.set(
                    self._get_metadata_key(user_message),
                    json.dumps(
                        {"created_at": now, "last_access": now, "access_count": 1}
                    ),
                    ex=expire,
                )
                await pipe.execute()

            logger.info(f"Cache updated for message: {user_message[:50]}...")

        except Exception:
            logger.warning("语义缓存写入失败，不影响本次回答", exc_info=True)

    @staticmethod
    def _cosine(vector: np.ndarray, payload: Any) -> Optional[float]:
        """计算查询向量与缓存向量的余弦相似度；数据不可用时返回 None。"""
        try:
            if not payload:
                return None
            cached = np.frombuffer(payload, dtype=_VECTOR_DTYPE)
            if cached.size != vector.size:
                return None
            denominator = float(np.linalg.norm(vector)) * float(np.linalg.norm(cached))
            if not denominator:
                return None
            return float(np.dot(vector, cached) / denominator)
        except (TypeError, ValueError):
            return None
