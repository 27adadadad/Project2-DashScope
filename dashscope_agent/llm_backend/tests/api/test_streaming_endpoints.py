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
        "app.models.user": {"User": object},
        "app.core.security": {"get_current_user": lambda: None},
        "app.services.conversation_service": {
            "ConversationService": type(
                "ConversationService",
                (),
                {
                    "save_message": staticmethod(lambda *args: None),
                    "delete_conversation": staticmethod(lambda *args: None),
                    "update_conversation_name": staticmethod(lambda *args: None),
                },
            )
        },
        "app.services.redis_semantic_cache": {
            "RedisSemanticCache": object,
            "get_shared_redis_client": lambda: None,
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
    response = asyncio.run(
        module.chat_endpoint(request, current_user=SimpleNamespace(id=7))
    )

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
    response = asyncio.run(module.reason_endpoint(request, current_user=SimpleNamespace(id=3)))
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
        asyncio.run(module.reason_endpoint(request, current_user=SimpleNamespace(id=3)))

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "DashScope request rate limit exceeded."


def test_langgraph_query_streams_typed_content_and_done(monkeypatch):
    module = import_main(monkeypatch)
    monkeypatch.setattr(module, "graph", FakeGraph([FakeChunk("回答")]))
    monkeypatch.setattr(module, "InputState", lambda **kwargs: SimpleNamespace(**kwargs))

    response = asyncio.run(
        module.langgraph_query(
            query="问题",
            user_id=7,
            conversation_id=None,
            image=None,
            current_user=SimpleNamespace(id=7),
        )
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
    response = asyncio.run(
        module.langgraph_resume(request, current_user=SimpleNamespace(id=7))
    )

    assert asyncio.run(collect_response(response)) == [
        'data: {"type": "error", "message": "LangGraph 流式处理失败。"}\n\n'
    ]


def test_chat_returns_cached_answer_for_an_independent_question(monkeypatch):
    module = import_main(monkeypatch)
    construction = []

    class FakeSemanticCache:
        async def lookup(self, _messages):
            return "缓存回答"

    model = FakeDashScopeService([])
    monkeypatch.setattr(
        module,
        "RedisSemanticCache",
        lambda **kwargs: construction.append(kwargs) or FakeSemanticCache(),
    )
    monkeypatch.setattr(module.ModelServiceFactory, "create_chat_service", lambda: model)

    request = module.ChatMessage(
        messages=[{"role": "user", "content": "退款多久到账？"}],
        user_id=7,
        conversation_id=9,
    )
    response = asyncio.run(module.chat_endpoint(request, current_user=SimpleNamespace(id=7)))

    assert asyncio.run(collect_response(response)) == [
        'data: {"type": "content", "content": "缓存回答"}\n\n',
        'data: {"type": "done"}\n\n',
    ]
    assert model.calls == []
    assert construction == [{"user_id": 7, "start_auto_cleanup": False}]


def test_chat_skips_semantic_cache_for_context_dependent_messages(monkeypatch):
    module = import_main(monkeypatch)
    constructed = []
    model = FakeDashScopeService(['data: {"type": "content", "content": "模型回答"}\n\n'])
    monkeypatch.setattr(
        module,
        "RedisSemanticCache",
        lambda **kwargs: constructed.append(kwargs),
    )
    monkeypatch.setattr(module.ModelServiceFactory, "create_chat_service", lambda: model)

    request = module.ChatMessage(
        messages=[
            {"role": "user", "content": "A1 保修多久？"},
            {"role": "assistant", "content": "两年。"},
            {"role": "user", "content": "那 A2 呢？"},
        ],
        user_id=7,
        conversation_id=9,
    )
    response = asyncio.run(module.chat_endpoint(request, current_user=SimpleNamespace(id=7)))

    assert asyncio.run(collect_response(response)) == model.events
    assert constructed == []


def test_chat_caches_complete_answer_after_a_normal_stream(monkeypatch):
    module = import_main(monkeypatch)
    writes = []

    class FakeSemanticCache:
        async def lookup(self, _messages):
            return None

        async def update(self, messages, response):
            writes.append((messages, response))

    model = FakeDashScopeService(
        [
            'data: {"type": "content", "content": "你好"}\n\n',
            'data: {"type": "content", "content": "世界"}\n\n',
            'data: {"type": "done"}\n\n',
        ]
    )
    monkeypatch.setattr(module, "RedisSemanticCache", lambda **_kwargs: FakeSemanticCache())
    monkeypatch.setattr(module.ModelServiceFactory, "create_chat_service", lambda: model)

    request = module.ChatMessage(
        messages=[{"role": "user", "content": "你好"}], user_id=7, conversation_id=9
    )
    response = asyncio.run(module.chat_endpoint(request, current_user=SimpleNamespace(id=7)))

    assert asyncio.run(collect_response(response)) == model.events
    assert writes == [(request.messages, "你好世界")]


def test_chat_falls_back_to_model_when_cache_lookup_fails(monkeypatch):
    module = import_main(monkeypatch)

    class FakeSemanticCache:
        async def lookup(self, _messages):
            raise RuntimeError("redis unavailable")

    model = FakeDashScopeService(['data: {"type": "content", "content": "模型回答"}\n\n'])
    monkeypatch.setattr(module, "RedisSemanticCache", lambda **_kwargs: FakeSemanticCache())
    monkeypatch.setattr(module.ModelServiceFactory, "create_chat_service", lambda: model)

    request = module.ChatMessage(
        messages=[{"role": "user", "content": "退款多久到账？"}],
        user_id=7,
        conversation_id=9,
    )
    response = asyncio.run(module.chat_endpoint(request, current_user=SimpleNamespace(id=7)))

    assert asyncio.run(collect_response(response)) == model.events
    assert len(model.calls) == 1


def test_conversation_mutations_use_authenticated_user_id(monkeypatch):
    module = import_main(monkeypatch)
    calls = []

    async def delete_conversation(conversation_id, user_id):
        calls.append(("delete", conversation_id, user_id))

    async def update_conversation_name(conversation_id, name, user_id):
        calls.append(("rename", conversation_id, name, user_id))

    monkeypatch.setattr(
        module.ConversationService,
        "delete_conversation",
        staticmethod(delete_conversation),
    )
    monkeypatch.setattr(
        module.ConversationService,
        "update_conversation_name",
        staticmethod(update_conversation_name),
    )
    user = SimpleNamespace(id=7)

    delete_response = asyncio.run(
        module.delete_conversation(conversation_id=9, current_user=user)
    )
    rename_response = asyncio.run(
        module.update_conversation_name(
            conversation_id=9,
            request=module.UpdateConversationNameRequest(name="新名称"),
            current_user=user,
        )
    )

    assert delete_response == {"message": "会话已删除"}
    assert rename_response == {"message": "会话名称已更新"}
    assert calls == [
        ("delete", 9, 7),
        ("rename", 9, "新名称", 7),
    ]


def test_chat_rag_injects_the_user_scoped_graphrag_retriever(monkeypatch):
    module = import_main(monkeypatch)
    constructed_with = []

    class FakeRetriever:
        pass

    class FakeRAGChatService:
        def __init__(self, *, retriever):
            constructed_with.append(retriever)

        async def generate_stream(self, _messages, _index_id):
            yield 'data: {"type": "content", "content": "有依据的回答"}\n\n'

    monkeypatch.setattr(module, "GraphRAGRetriever", FakeRetriever)
    monkeypatch.setattr(module, "RAGChatService", FakeRAGChatService)

    request = module.RAGChatRequest(
        messages=[{"role": "user", "content": "保修多久？"}],
        index_id="5d417677-9436-5c5b-997a-a3d989d9abbd",
        user_id=7,
    )
    response = asyncio.run(
        module.rag_chat_endpoint(request, current_user=SimpleNamespace(id=7))
    )

    assert isinstance(constructed_with[0], FakeRetriever)
    assert asyncio.run(collect_response(response)) == [
        'data: {"type": "content", "content": "有依据的回答"}\n\n'
    ]


def test_chat_rag_rejects_an_index_not_owned_by_authenticated_user(monkeypatch):
    module = import_main(monkeypatch)
    request = module.RAGChatRequest(
        messages=[{"role": "user", "content": "保修多久？"}],
        index_id="123e4567-e89b-12d3-a456-426614174000",
        user_id=999,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            module.rag_chat_endpoint(
                request,
                current_user=SimpleNamespace(id=7),
            )
        )

    assert exc_info.value.status_code == 403


def test_upload_returns_the_index_id_required_by_chat_rag(monkeypatch, tmp_path):
    module = import_main(monkeypatch)

    class FakeIndexingService:
        async def process_file(self, _file_info):
            return {"status": "success"}

    class FakeUpload:
        filename = "policy.txt"
        content_type = "text/plain"

        def __init__(self):
            self._chunks = [b"warranty policy"]

        async def read(self, size=-1):
            # 真实 UploadFile 读到末尾会返回空字节；替身也必须如此，否则分块写盘不会终止
            return self._chunks.pop(0) if self._chunks else b""

    monkeypatch.setattr(module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(module, "IndexingService", FakeIndexingService)

    result = asyncio.run(
        module.upload_file(
            FakeUpload(), user_id=7, current_user=SimpleNamespace(id=7)
        )
    )

    assert result["index_id"] == "5d417677-9436-5c5b-997a-a3d989d9abbd"


def test_readmes_describe_the_json_type_sse_protocol():
    project_root = Path(__file__).resolve().parents[4]
    for readme in (project_root / "README.md", project_root / "dashscope_agent" / "README.md"):
        content = readme.read_text(encoding="utf-8")
        assert "data:" in content
        assert '"type"' in content
        assert "event: <类型>" not in content


def test_frontend_static_directory_directly_contains_index_html(monkeypatch):
    module = import_main(monkeypatch)

    assert (module.STATIC_DIR / "index.html").is_file()


def test_frontend_stream_parser_extracts_typed_sse_content():
    asset = (
        Path(__file__).resolve().parents[2]
        / "static"
        / "dist"
        / "dist"
        / "assets"
        / "index-B0CElU7P.js"
    ).read_text(encoding="utf-8")

    assert "const d=JSON.parse(f)" in asset
    assert 'd.type==="content"&&(s+=d.content,u({type:"response",content:s}))' in asset
    assert 'let r="",s="",a=""' in asset
    assert 'a=l.pop()||""' in asset
    assert 'const b=JSON.parse(se),S=u.value[u.value.length-1];if(b.type==="content")W+=b.content||""' in asset
    assert 'else if(b.type==="sources")S.sources=Array.isArray(b.sources)?b.sources:[]' in asset
    assert 'else if(b.type==="error")throw new Error(b.message||"知识库问答失败")' in asset


def test_frontend_bundle_contains_auto_rag_upload_and_stream_protocol():
    asset = (
        Path(__file__).resolve().parents[2]
        / "static"
        / "dist"
        / "dist"
        / "assets"
        / "index-B0CElU7P.js"
    ).read_text(encoding="utf-8")

    assert "/api/upload" in asset
    assert "/chat-rag" in asset
    assert "index_id" in asset
    assert 'index_result.status==="success"' in asset
    assert 'b.type==="sources"' in asset
    assert 'b.type==="error"' in asset
