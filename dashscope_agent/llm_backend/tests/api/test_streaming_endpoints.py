import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import APIRouter, HTTPException


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def import_main(monkeypatch):
    settings = {
        "DASHSCOPE_API_KEY": "test-dashscope-key",
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

    class FakeLogger:
        def info(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    logger_module = ModuleType("app.core.logger")
    logger_module.get_logger = lambda **kwargs: FakeLogger()
    logger_module.log_structured = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "app.core.logger", logger_module)

    middleware_module = ModuleType("app.core.middleware")
    middleware_module.LoggingMiddleware = object
    monkeypatch.setitem(sys.modules, "app.core.middleware", middleware_module)

    api_module = ModuleType("app.api")
    api_module.api_router = APIRouter()
    monkeypatch.setitem(sys.modules, "app.api", api_module)

    sqlalchemy_module = ModuleType("sqlalchemy")
    sqlalchemy_module.select = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy_module)

    multipart_module = ModuleType("python_multipart")
    multipart_module.__version__ = "0.0.13"
    monkeypatch.setitem(sys.modules, "python_multipart", multipart_module)

    staticfiles_module = ModuleType("fastapi.staticfiles")
    staticfiles_module.StaticFiles = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, "fastapi.staticfiles", staticfiles_module)

    for module_name, attributes in {
        "app.core.database": {"AsyncSessionLocal": None},
        "app.models.conversation": {"Conversation": object, "DialogueType": object},
        "app.models.message": {"Message": object},
        "app.services.conversation_service": {
            "ConversationService": type(
                "ConversationService", (), {"save_message": staticmethod(lambda *args: None)}
            )
        },
        "app.services.indexing_service": {"IndexingService": object},
        "app.lg_agent.lg_states": {"AgentState": object, "InputState": object},
        "app.lg_agent.utils": {"new_uuid": lambda: "test-id"},
        "app.lg_agent.lg_builder": {"graph": object},
        "langgraph.types": {"Command": object},
    }.items():
        stub = ModuleType(module_name)
        for name, value in attributes.items():
            setattr(stub, name, value)
        monkeypatch.setitem(sys.modules, module_name, stub)
    for module_name in ("main", "app.core.config"):
        sys.modules.pop(module_name, None)
    return importlib.import_module("main")


async def collect_response(response):
    return [chunk async for chunk in response.body_iterator]


class FakeDashScopeService:
    def __init__(self, events):
        self.events = events
        self.calls = []

    async def generate_stream(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        for event in self.events:
            yield event


class FakeChunk:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.additional_kwargs = {"tool_calls": tool_calls} if tool_calls else {}


class FakeGraph:
    def __init__(self, events, error=None):
        self.events = events
        self.error = error

    def get_state(self, _config):
        return []

    async def astream(self, *args, **kwargs):
        if self.error:
            raise self.error
        for event in self.events:
            yield event, {"tags": []}


def test_chat_uses_dashscope_normal_stream_and_preserves_completion_callback(monkeypatch):
    module = import_main(monkeypatch)
    service = FakeDashScopeService(
        ['data: {"type": "content", "content": "你好"}\n\n']
    )
    monkeypatch.setattr(
        module.ModelServiceFactory,
        "create_chat_service",
        lambda: service,
    )

    request = module.ChatMessage(
        messages=[{"role": "user", "content": "测试"}],
        user_id=7,
        conversation_id=9,
    )
    response = asyncio.run(module.chat_endpoint(request))

    assert asyncio.run(collect_response(response)) == [
        'data: {"type": "content", "content": "你好"}\n\n'
    ]
    assert service.calls == [
        (
            (),
            {
                "messages": request.messages,
                "user_id": 7,
                "conversation_id": 9,
                "on_complete": module.ConversationService.save_message,
                "thinking": False,
            },
        )
    ]


def test_reason_uses_dashscope_thinking_stream_with_reasoning_and_content(monkeypatch):
    module = import_main(monkeypatch)
    service = FakeDashScopeService(
        [
            'data: {"type": "reasoning", "content": "分析"}\n\n',
            'data: {"type": "content", "content": "答案"}\n\n',
        ]
    )
    monkeypatch.setattr(
        module.ModelServiceFactory,
        "create_reasoner_service",
        lambda: service,
    )

    request = module.ReasonRequest(messages=[{"role": "user", "content": "推理"}], user_id=3)
    response = asyncio.run(module.reason_endpoint(request))
    payloads = [
        json.loads(event.removeprefix("data: ").strip())
        for event in asyncio.run(collect_response(response))
    ]

    assert payloads == [
        {"type": "reasoning", "content": "分析"},
        {"type": "content", "content": "答案"},
    ]
    assert service.calls == [((request.messages,), {"thinking": True})]


def test_reason_maps_dashscope_error_to_safe_http_error(monkeypatch):
    module = import_main(monkeypatch)

    def raise_dashscope_error():
        raise module.DashScopeServiceError("rate_limit")

    monkeypatch.setattr(
        module.ModelServiceFactory,
        "create_reasoner_service",
        raise_dashscope_error,
    )

    request = module.ReasonRequest(messages=[{"role": "user", "content": "推理"}], user_id=3)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(module.reason_endpoint(request))

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "DashScope request rate limit exceeded."


def test_langgraph_query_streams_typed_content_and_done(monkeypatch):
    module = import_main(monkeypatch)
    monkeypatch.setattr(module, "graph", FakeGraph([FakeChunk("回答")]))
    monkeypatch.setattr(module, "InputState", lambda **kwargs: SimpleNamespace(**kwargs))

    response = asyncio.run(
        module.langgraph_query(query="问题", user_id=7, conversation_id=None, image=None)
    )

    assert response.headers["X-Conversation-ID"]
    assert asyncio.run(collect_response(response)) == [
        'data: {"type": "content", "content": "回答"}\n\n',
        'data: {"type": "done"}\n\n',
    ]


def test_langgraph_resume_streams_safe_error_instead_of_aborting(monkeypatch):
    module = import_main(monkeypatch)
    monkeypatch.setattr(module, "graph", FakeGraph([], error=RuntimeError("secret upstream detail")))
    monkeypatch.setattr(module, "Command", lambda **kwargs: kwargs)

    request = module.LangGraphResumeRequest(
        query="继续", user_id=7, conversation_id="conversation-1"
    )
    response = asyncio.run(module.langgraph_resume(request))

    assert asyncio.run(collect_response(response)) == [
        'data: {"type": "error", "message": "LangGraph 流式处理失败。"}\n\n'
    ]


def test_readmes_describe_the_json_type_sse_protocol():
    project_root = Path(__file__).resolve().parents[4]
    for readme in (project_root / "README.md", project_root / "dashscope_agent" / "README.md"):
        content = readme.read_text(encoding="utf-8")
        assert "data:" in content
        assert '"type"' in content
        assert "event: <类型>" not in content
