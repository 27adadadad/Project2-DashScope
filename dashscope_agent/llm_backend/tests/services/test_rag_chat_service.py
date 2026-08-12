import asyncio
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class FakeRetriever:
    async def retrieve(self, query, index_id):
        assert query == "保修多久？"
        assert index_id == "index-1"
        return [{"content": "冰箱整机保修一年。", "metadata": {"source": "保修卡"}}]


class FakeDashScopeService:
    def __init__(self):
        self.calls = []

    async def generate_stream(self, messages, *, thinking):
        self.calls.append((messages, thinking))
        yield 'data: {"type": "content", "content": "一年"}\n\n'


async def collect(stream):
    return [item async for item in stream]


def test_rag_chat_adds_retrieval_context_to_system_prompt_and_disables_thinking():
    from app.services.rag_chat_service import RAGChatService

    service = FakeDashScopeService()
    rag = RAGChatService(retriever=FakeRetriever(), model_service=service)
    events = asyncio.run(
        collect(rag.generate_stream([{"role": "user", "content": "保修多久？"}], "index-1"))
    )

    assert events == ['data: {"type": "content", "content": "一年"}\n\n']
    messages, thinking = service.calls[0]
    assert thinking is False
    assert messages[0]["role"] == "system"
    assert "冰箱整机保修一年。" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "保修多久？"}


def test_rag_chat_emits_safe_sse_error_when_retriever_is_unavailable():
    from app.services.rag_chat_service import RAGChatService

    events = asyncio.run(
        collect(RAGChatService().generate_stream([{"role": "user", "content": "保修多久？"}], "index-1"))
    )

    payload = json.loads(events[0].removeprefix("data: ").strip())
    assert payload == {"type": "error", "message": "RAG retrieval is unavailable."}
