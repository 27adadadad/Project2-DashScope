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

    assert events[0] == 'data: {"type": "content", "content": "一年"}\n\n'
    assert json.loads(events[1].removeprefix("data: ").strip()) == {
        "type": "sources",
        "sources": [{"id": 1, "source": "保修卡", "retrieval_channels": []}],
    }
    messages, thinking = service.calls[0]
    assert thinking is False
    assert messages[0]["role"] == "system"
    assert "[资料 1｜来源：保修卡]" in messages[0]["content"]
    assert "冰箱整机保修一年。" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "保修多久？"}


def test_rag_chat_emits_safe_sse_error_when_retriever_is_unavailable():
    from app.services.rag_chat_service import RAGChatService

    events = asyncio.run(
        collect(RAGChatService().generate_stream([{"role": "user", "content": "保修多久？"}], "index-1"))
    )

    payload = json.loads(events[0].removeprefix("data: ").strip())
    assert payload == {"type": "error", "message": "RAG retrieval is unavailable."}


def test_rag_chat_logs_the_cause_of_an_unexpected_stream_error(monkeypatch):
    import app.services.rag_chat_service as module

    class FailingRetriever:
        async def retrieve(self, _query, _index_id):
            raise RuntimeError("missing GraphRAG output")

    class FakeLogger:
        def __init__(self):
            self.messages = []

        def exception(self, message):
            self.messages.append(message)

    logger = FakeLogger()
    monkeypatch.setattr(module, "logger", logger)

    events = asyncio.run(
        collect(
            module.RAGChatService(retriever=FailingRetriever()).generate_stream(
                [{"role": "user", "content": "保修多久？"}], "index-1"
            )
        )
    )

    assert json.loads(events[0].removeprefix("data: ").strip()) == {
        "type": "error",
        "message": "RAG request could not be completed.",
    }
    assert logger.messages == ["RAG request failed"]


def test_shared_context_gate_keeps_online_nonempty_baseline():
    from app.services.rag_chat_service import RAGChatService

    assert RAGChatService.has_retrieval_context([]) is False
    assert RAGChatService.has_retrieval_context([{"content": "无关资料"}]) is True
    assert RAGChatService.has_retrieval_context([{"content": ""}]) is True


def test_shared_prompt_builder_matches_online_messages_without_mutating_input():
    from app.services.rag_chat_service import RAGChatService

    messages = [{"role": "user", "content": "保修多久？"}]
    documents = [{"content": "冰箱整机保修一年。", "source": "保修卡"}]
    result = RAGChatService.build_prompt_messages(messages, documents)

    assert result == [
        {"role": "system", "content": RAGChatService._system_prompt(documents)},
        {"role": "user", "content": "保修多久？"},
    ]
    assert messages == [{"role": "user", "content": "保修多久？"}]


def test_online_chat_uses_shared_gate_and_prompt_builder():
    from app.services.rag_chat_service import RAGChatService

    class SharedService(RAGChatService):
        gate_calls = []
        prompt_calls = []

        @staticmethod
        def has_retrieval_context(documents):
            SharedService.gate_calls.append(documents)
            return RAGChatService.has_retrieval_context(documents)

        @classmethod
        def build_prompt_messages(cls, messages, documents):
            cls.prompt_calls.append((messages, documents))
            return super().build_prompt_messages(messages, documents)

    model = FakeDashScopeService()
    service = SharedService(retriever=FakeRetriever(), model_service=model)
    asyncio.run(collect(service.generate_stream([{"role": "user", "content": "保修多久？"}], "index-1")))

    assert len(SharedService.gate_calls) == 1
    assert len(SharedService.prompt_calls) == 1
    assert len(model.calls) == 1


def test_empty_online_context_emits_error_without_calling_model():
    from app.services.rag_chat_service import RAGChatService

    model = FakeDashScopeService()
    service = RAGChatService(retriever=lambda *_: [], model_service=model)
    events = asyncio.run(collect(service.generate_stream([{"role": "user", "content": "保修多久？"}], "index-1")))

    assert model.calls == []
    assert len(events) == 1
    assert json.loads(events[0].removeprefix("data: ").strip()) == {
        "type": "error", "message": "RAG retrieval is unavailable."
    }
