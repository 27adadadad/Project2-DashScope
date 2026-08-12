import importlib
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_model_factory_creates_only_dashscope_services(monkeypatch):
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
    for module_name in (
        "app.core.config",
        "app.services.dashscope_service",
        "app.services.model_service_factory",
    ):
        sys.modules.pop(module_name, None)

    module = importlib.import_module("app.services.model_service_factory")

    class FakeDashScopeService:
        pass

    monkeypatch.setattr(module, "DashScopeService", FakeDashScopeService)

    assert isinstance(module.ModelServiceFactory.create_chat_service(), FakeDashScopeService)
    assert isinstance(module.ModelServiceFactory.create_reasoner_service(), FakeDashScopeService)
