import importlib
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_settings_require_dashscope_api_key(monkeypatch):
    required_non_model_settings = {
        "SERPAPI_KEY": "test-serpapi-key",
        "DB_HOST": "localhost",
        "DB_PORT": "3306",
        "DB_USER": "test-user",
        "DB_PASSWORD": "test-password",
        "DB_NAME": "test-db",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
    }
    for name, value in required_non_model_settings.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    sys.modules.pop("app.core.config", None)

    with pytest.raises(ValidationError) as exc_info:
        importlib.import_module("app.core.config")

    assert "DASHSCOPE_API_KEY" in str(exc_info.value)
