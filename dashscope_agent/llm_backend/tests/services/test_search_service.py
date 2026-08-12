import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class FakeDashScopeService:
    def __init__(self):
        self.tool_messages = None
        self.tools = None
        self.summary_messages = None

    async def generate_tool_response(self, messages, tools):
        self.tool_messages = messages
        self.tools = tools
        return {
            "finish_reason": "tool_calls",
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "Qwen"}',
                        }
                    }
                ]
            },
        }

    async def generate_search_summary_stream(self, messages, **kwargs):
        self.summary_messages = messages
        yield 'data: {"type": "content", "content": "总结"}\n\n'
        yield 'data: {"type": "done"}\n\n'


def test_search_service_emits_search_events_and_summarizes_with_dashscope(monkeypatch):
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
    logger_module = ModuleType("app.core.logger")
    logger_module.get_logger = lambda **kwargs: type(
        "Logger", (), {"warning": lambda *args, **kwargs: None, "exception": lambda *args, **kwargs: None}
    )()
    monkeypatch.setitem(sys.modules, "app.core.logger", logger_module)
    for module_name in ("app.core.config", "app.tools.search", "app.services.search_service"):
        sys.modules.pop(module_name, None)
    from app.services.search_service import SearchService

    dashscope = FakeDashScopeService()
    service = SearchService(model_service=dashscope)

    async def fake_execute_tool(name, arguments):
        assert name == "search"
        assert json.loads(arguments) == {"query": "Qwen"}
        return [{"title": "Qwen", "url": "https://example.invalid", "snippet": "模型"}]

    monkeypatch.setattr(service.tool_registry, "execute_tool", fake_execute_tool)
    events = asyncio.run(
        collect(service.generate_stream("Qwen"))
    )

    payloads = [json.loads(event.removeprefix("data: ").strip()) for event in events]
    assert payloads[:2] == [
        {"type": "search_start"},
        {
            "type": "search_results",
            "total": 1,
            "query": "Qwen",
            "results": [{"title": "Qwen", "url": "https://example.invalid", "snippet": "模型"}],
        },
    ]
    assert payloads[2:] == [
        {"type": "content", "content": "总结"},
        {"type": "done"},
    ]
    assert dashscope.tools
    assert dashscope.summary_messages == [{"role": "system", "content": dashscope.summary_messages[0]["content"]}]


async def collect(stream):
    return [event async for event in stream]
