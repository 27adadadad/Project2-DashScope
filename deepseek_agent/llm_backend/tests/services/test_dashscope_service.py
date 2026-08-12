import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class FakeGeneration:
    def __init__(self, chunks=None, error=None):
        self.chunks = chunks or []
        self.error = error
        self.kwargs = None

    def call(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return iter(self.chunks)


class FakeUpstreamError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


def make_chunk(*, reasoning_content=None, content=None):
    message = SimpleNamespace(
        reasoning_content=reasoning_content,
        content=content,
    )
    return SimpleNamespace(output=SimpleNamespace(choices=[SimpleNamespace(message=message)]))


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


async def collect_events(service, messages, *, thinking):
    return [event async for event in service.generate_stream(messages, thinking=thinking)]


def decode_event(event):
    return json.loads(event.removeprefix("data: ").strip())


def test_thinking_stream_separates_reasoning_answer_and_finishes(monkeypatch):
    module = import_service(monkeypatch)
    generation = FakeGeneration(
        [
            make_chunk(reasoning_content="先分析"),
            make_chunk(reasoning_content="问题", content="答案："),
            make_chunk(content="好的"),
        ]
    )
    service = module.DashScopeService(generation=generation)

    events = asyncio.run(
        collect_events(service, [{"role": "user", "content": "你好"}], thinking=True)
    )

    assert [decode_event(event) for event in events] == [
        {"type": "reasoning", "content": "先分析"},
        {"type": "reasoning", "content": "问题"},
        {"type": "content", "content": "答案："},
        {"type": "content", "content": "好的"},
        {"type": "done"},
    ]
    assert generation.kwargs == {
        "api_key": "test-secret-dashscope-key",
        "model": module.settings.DASHSCOPE_REASON_MODEL,
        "messages": [{"role": "user", "content": "你好"}],
        "result_format": "message",
        "stream": True,
        "incremental_output": True,
        "enable_thinking": True,
    }
    assert "\\u" not in events[0]


def test_standard_stream_omits_reasoning_and_uses_chat_model(monkeypatch):
    module = import_service(monkeypatch)
    generation = FakeGeneration(
        [make_chunk(reasoning_content="不应暴露", content="普通回复")]
    )
    service = module.DashScopeService(generation=generation)

    events = asyncio.run(
        collect_events(service, [{"role": "user", "content": "你好"}], thinking=False)
    )

    assert [decode_event(event) for event in events] == [
        {"type": "content", "content": "普通回复"},
        {"type": "done"},
    ]
    assert generation.kwargs["model"] == module.settings.DASHSCOPE_CHAT_MODEL
    assert generation.kwargs["enable_thinking"] is False


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (FakeUpstreamError("unauthorized test-secret-dashscope-key", 401), "authentication"),
        (FakeUpstreamError("rate limit test-secret-dashscope-key", 429), "rate_limit"),
        (TimeoutError("timeout test-secret-dashscope-key"), "timeout"),
        (RuntimeError("unexpected test-secret-dashscope-key"), "upstream"),
    ],
)
def test_upstream_errors_are_safe_and_do_not_leak_api_key(monkeypatch, error, category):
    module = import_service(monkeypatch)
    service = module.DashScopeService(generation=FakeGeneration(error=error))

    with pytest.raises(module.DashScopeServiceError) as exc_info:
        asyncio.run(collect_events(service, [{"role": "user", "content": "你好"}], thinking=False))

    assert exc_info.value.category == category
    assert "test-secret-dashscope-key" not in str(exc_info.value)
