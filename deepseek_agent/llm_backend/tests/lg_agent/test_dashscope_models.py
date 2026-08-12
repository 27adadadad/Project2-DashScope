import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def configure_dashscope_environment(monkeypatch):
    values = {
        "DASHSCOPE_API_KEY": "test-dashscope-key",
        "DASHSCOPE_COMPATIBLE_BASE_URL": "https://dashscope.example/v1",
        "DASHSCOPE_CHAT_MODEL": "qwen-test",
        "SERPAPI_KEY": "test-serpapi-key",
        "DB_HOST": "localhost",
        "DB_PORT": "3306",
        "DB_USER": "test-user",
        "DB_PASSWORD": "test-password",
        "DB_NAME": "test-db",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def clear_dashscope_modules():
    for module_name in (
        "app.core.config",
        "app.services.dashscope_langchain",
        "app.services.dashscope_service",
        "app.lg_agent.lg_builder",
        "app.lg_agent.kg_sub_graph.agentic_rag_agents.components.cypher_tools.node",
    ):
        sys.modules.pop(module_name, None)


def test_create_agent_model_uses_dashscope_compatible_openai_settings(monkeypatch):
    configure_dashscope_environment(monkeypatch)
    clear_dashscope_modules()
    module = importlib.import_module("app.services.dashscope_langchain")
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, "ChatOpenAI", FakeChatOpenAI)

    model = module.create_agent_model(["router"])

    assert isinstance(model, FakeChatOpenAI)
    assert captured == {
        "model": "qwen-test",
        "api_key": "test-dashscope-key",
        "base_url": "https://dashscope.example/v1",
        "temperature": 0.7,
        "tags": ["router"],
        "extra_body": {"enable_thinking": False},
    }


def test_langgraph_modules_import_with_dashscope_environment_only(monkeypatch):
    configure_dashscope_environment(monkeypatch)
    clear_dashscope_modules()

    builder = importlib.import_module("app.lg_agent.lg_builder")
    cypher_node = importlib.import_module(
        "app.lg_agent.kg_sub_graph.agentic_rag_agents.components.cypher_tools.node"
    )

    assert builder.graph is not None
    assert callable(cypher_node.create_cypher_query_node)


def test_main_imports_with_fake_non_model_dashscope_environment(monkeypatch):
    configure_dashscope_environment(monkeypatch)
    clear_dashscope_modules()
    sys.modules.pop("app.services.indexing_service", None)
    sys.modules.pop("main", None)
    static_dist = PROJECT_ROOT / "static" / "dist"
    created_static_dist = not static_dist.exists()
    static_dist.mkdir(parents=True, exist_ok=True)

    try:
        main = importlib.import_module("main")
    finally:
        if created_static_dist:
            static_dist.rmdir()

    assert main.app is not None


def test_image_query_uses_fake_dashscope_service_without_network(monkeypatch, tmp_path):
    configure_dashscope_environment(monkeypatch)
    clear_dashscope_modules()
    builder = importlib.import_module("app.lg_agent.lg_builder")
    image_path = tmp_path / "product.jpg"
    image_path.write_bytes(b"image-data")
    described_images = []

    class FakeDashScopeService:
        async def describe_image(self, path):
            described_images.append(path)
            return "图片里是一盏智能台灯"

    class FakeAgentModel:
        async def ainvoke(self, messages):
            return SimpleNamespace(content="这是一盏支持语音控制的智能台灯。")

    monkeypatch.setattr(builder, "DashScopeService", FakeDashScopeService)
    monkeypatch.setattr(builder, "create_agent_model", lambda tags: FakeAgentModel())
    state = SimpleNamespace(messages=[SimpleNamespace(content="这是什么产品？")])

    result = asyncio.run(
        builder.create_image_query(
            state, config={"configurable": {"image_path": str(image_path)}}
        )
    )

    assert described_images == [str(image_path)]
    assert result["messages"][0].content == "这是一盏支持语音控制的智能台灯。"
