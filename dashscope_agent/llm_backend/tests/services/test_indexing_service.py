import asyncio
import builtins
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_process_file_returns_error_with_path_when_graphrag_import_fails(monkeypatch):
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
    sys.modules.pop("app.core.config", None)
    sys.modules.pop("app.services.indexing_service", None)
    from app.services.indexing_service import IndexingService

    original_import = builtins.__import__

    def fail_graphrag_import(name, *args, **kwargs):
        if name == "graphrag.api":
            raise ModuleNotFoundError("No module named 'graphrag'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_graphrag_import)

    result = asyncio.run(
        IndexingService().process_file({"path": "/tmp/missing-graphrag.txt"})
    )

    assert result["status"] == "error"
    assert result["file_path"] == "/tmp/missing-graphrag.txt"


def test_indexing_service_defaults_to_empty_file_type_mapping(monkeypatch):
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
    sys.modules.pop("app.core.config", None)
    sys.modules.pop("app.services.indexing_service", None)
    from app.services.indexing_service import IndexingService

    service = IndexingService()

    assert service.config_mapping == {}
    assert service._get_config_file("application/pdf") == "settings.yaml"


def test_indexing_service_exposes_validated_dashscope_settings_to_graphrag(monkeypatch):
    from app.services import indexing_service as module

    monkeypatch.delenv("DASHSCOPE_COMPATIBLE_BASE_URL", raising=False)
    monkeypatch.delenv("DASHSCOPE_CHAT_MODEL", raising=False)
    monkeypatch.delenv("DASHSCOPE_EMBEDDING_MODEL", raising=False)
    monkeypatch.setattr(
        module.settings,
        "DASHSCOPE_COMPATIBLE_BASE_URL",
        "https://example.test/compatible/v1",
    )
    monkeypatch.setattr(module.settings, "DASHSCOPE_CHAT_MODEL", "test-chat")
    monkeypatch.setattr(
        module.settings,
        "DASHSCOPE_EMBEDDING_MODEL",
        "test-embedding",
    )

    module.IndexingService()._configure_graphrag_environment()

    assert __import__("os").environ["DASHSCOPE_COMPATIBLE_BASE_URL"] == (
        "https://example.test/compatible/v1"
    )
    assert __import__("os").environ["DASHSCOPE_CHAT_MODEL"] == "test-chat"
    assert __import__("os").environ["DASHSCOPE_EMBEDDING_MODEL"] == "test-embedding"
