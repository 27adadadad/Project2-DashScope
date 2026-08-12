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
