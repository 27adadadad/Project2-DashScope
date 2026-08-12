import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class FakeTextEmbedding:
    def __init__(self):
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            status_code=200,
            output=SimpleNamespace(
                embeddings=[SimpleNamespace(embedding=[0.1, 0.2])]
            ),
        )


def test_embed_query_uses_native_text_embedding_query_mode(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("SERPAPI_KEY", "test-serpapi-key")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "3306")
    monkeypatch.setenv("DB_USER", "test-user")
    monkeypatch.setenv("DB_PASSWORD", "test-password")
    monkeypatch.setenv("DB_NAME", "test-db")
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("REDIS_PORT", "6379")
    sys.modules.pop("app.core.config", None)
    sys.modules.pop("app.services.dashscope_embeddings", None)
    from app.services.dashscope_embeddings import DashScopeEmbeddings

    client = FakeTextEmbedding()
    embedding = asyncio.run(DashScopeEmbeddings(client=client).embed_query("客户问题"))

    assert embedding == [0.1, 0.2]
    assert client.calls == [
        {
            "model": "qwen3.7-text-embedding",
            "input": "客户问题",
            "text_type": "query",
            "dimension": 1024,
        }
    ]


def test_embed_documents_uses_native_text_embedding_document_mode(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("SERPAPI_KEY", "test-serpapi-key")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "3306")
    monkeypatch.setenv("DB_USER", "test-user")
    monkeypatch.setenv("DB_PASSWORD", "test-password")
    monkeypatch.setenv("DB_NAME", "test-db")
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("REDIS_PORT", "6379")
    sys.modules.pop("app.core.config", None)
    sys.modules.pop("app.services.dashscope_embeddings", None)
    from app.services.dashscope_embeddings import DashScopeEmbeddings

    client = FakeTextEmbedding()
    embeddings = asyncio.run(
        DashScopeEmbeddings(client=client).embed_documents(["文档一", "文档二"])
    )

    assert embeddings == [[0.1, 0.2]]
    assert client.calls[0]["input"] == ["文档一", "文档二"]
    assert client.calls[0]["text_type"] == "document"
