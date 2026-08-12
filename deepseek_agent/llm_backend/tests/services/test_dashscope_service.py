import asyncio
import importlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class FakeConversation:
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.kwargs = None

    def call(self, **kwargs):
        self.kwargs = kwargs
        return iter(self.chunks)


class FakeUpstreamError(Exception):
    def __init__(self, message, status_code=500):
        super().__init__(message)
        self.status_code = status_code


class FailingStream:
    def __iter__(self):
        yield make_chunk(reasoning_content="不应暴露", content="已收到")
        raise FakeUpstreamError("upstream failed: test-secret-dashscope-key", 429)


class BlockingConversation(FakeConversation):
    def call(self, **kwargs):
        self.kwargs = kwargs
        time.sleep(0.05)
        return iter([make_chunk(content="线程桥接")])


def make_chunk(*, reasoning_content=None, content=None, status_code=200):
    message = SimpleNamespace(
        reasoning_content=reasoning_content,
        content=content,
    )
    return SimpleNamespace(
        status_code=status_code,
        output=SimpleNamespace(choices=[SimpleNamespace(message=message)]),
    )


def import_service(monkeypatch):
    settings = {
        "DASHSCOPE_API_KEY": "test-secret-dashscope-key",
        "SERPAPI_KEY": "test-serpapi-key",
        "DB_HOST": "localhost",
        "DB_PORT": "3306",
        "DB_USER": "test-user",
        "DB_PASSWORD": "test-password",
        "DB_NAME": "test-db",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
    }
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    sys.modules.pop("app.core.config", None)
    sys.modules.pop("app.services.dashscope_service", None)
    return importlib.import_module("app.services.dashscope_service")


async def collect_events(service, messages, **kwargs):
    return [event async for event in service.generate_stream(messages, **kwargs)]


def decode_event(event):
    return json.loads(event.removeprefix("data: ").strip())


def test_multimodal_reasoning_and_content_blocks_are_normalized(monkeypatch):
    module = import_service(monkeypatch)
    conversation = FakeConversation(
        [
            make_chunk(reasoning_content=[{"text": "先"}, SimpleNamespace(text="分析")]),
            make_chunk(
                content=[
                    {"text": "答"},
                    {"image": "https://example.invalid/image.png"},
                    SimpleNamespace(text="案"),
                ]
            ),
        ]
    )
    service = module.DashScopeService(conversation_client=conversation)

    events = asyncio.run(
        collect_events(service, [{"role": "user", "content": "你好"}], thinking=True)
    )

    assert [decode_event(event) for event in events] == [
        {"type": "reasoning", "content": "先分析"},
        {"type": "content", "content": "答案"},
        {"type": "done"},
    ]
    assert conversation.kwargs == {
        "api_key": "test-secret-dashscope-key",
        "model": module.settings.DASHSCOPE_REASON_MODEL,
        "messages": [{"role": "user", "content": "你好"}],
        "result_format": "message",
        "stream": True,
        "incremental_output": True,
        "enable_thinking": True,
    }
    assert "\\u" not in events[0]


def test_successful_stream_calls_on_complete_with_only_final_content(monkeypatch):
    module = import_service(monkeypatch)
    conversation = FakeConversation(
        [
            make_chunk(reasoning_content="不计入结果", content="答"),
            make_chunk(content={"text": "案"}),
        ]
    )
    completed = []

    async def on_complete(user_id, conversation_id, messages, response):
        completed.append((user_id, conversation_id, messages, response))

    service = module.DashScopeService(conversation_client=conversation)
    messages = [{"role": "user", "content": "你好"}]
    events = asyncio.run(
        collect_events(
            service,
            messages,
            thinking=True,
            user_id=7,
            conversation_id=11,
            on_complete=on_complete,
        )
    )

    assert [decode_event(event) for event in events][-1] == {"type": "done"}
    assert completed == [(7, 11, messages, "答案")]


def test_stream_error_becomes_safe_error_sse_without_done(monkeypatch):
    module = import_service(monkeypatch)
    service = module.DashScopeService(
        conversation_client=FakeConversation(FailingStream())
    )

    events = asyncio.run(
        collect_events(service, [{"role": "user", "content": "你好"}], thinking=False)
    )

    payloads = [decode_event(event) for event in events]
    assert payloads == [
        {"type": "content", "content": "已收到"},
        {
            "type": "error",
            "message": "DashScope request rate limit exceeded.",
        },
    ]
    assert all(payload["type"] != "done" for payload in payloads)
    assert "test-secret-dashscope-key" not in events[-1]


def test_non_200_response_chunk_becomes_safe_error_sse(monkeypatch):
    module = import_service(monkeypatch)
    error_chunk = SimpleNamespace(
        status_code=401,
        message="unauthorized test-secret-dashscope-key",
    )
    service = module.DashScopeService(
        conversation_client=FakeConversation([error_chunk])
    )

    events = asyncio.run(
        collect_events(service, [{"role": "user", "content": "你好"}], thinking=False)
    )

    assert [decode_event(event) for event in events] == [
        {"type": "error", "message": "DashScope authentication failed."}
    ]
    assert "test-secret-dashscope-key" not in events[0]


def test_sync_client_call_does_not_block_event_loop(monkeypatch):
    module = import_service(monkeypatch)
    service = module.DashScopeService(conversation_client=BlockingConversation())

    async def consume_while_heartbeat_runs():
        stream = service.generate_stream(
            [{"role": "user", "content": "你好"}], thinking=False
        )
        first_event = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.01)
        yielded_before_sync_call_finished = not first_event.done()
        event = await first_event
        await stream.aclose()
        return yielded_before_sync_call_finished, event

    yielded, event = asyncio.run(consume_while_heartbeat_runs())

    assert yielded is True
    assert decode_event(event) == {"type": "content", "content": "线程桥接"}
