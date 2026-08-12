"""回归测试：GraphRAG 的 PDF 表格描述必须使用 DashScope 默认值。"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace


BACKEND_ROOT = Path(__file__).resolve().parents[1]
GRAPHRAG_ROOT = BACKEND_ROOT / "app" / "graphrag"
PDF_INPUT_MODULE = GRAPHRAG_ROOT / "graphrag" / "index" / "input" / "pdf.py"


def _load_table_description_config_resolver():
    """仅执行被测的纯函数，避免导入可选的 GraphRAG 运行时依赖。"""
    module = ast.parse(PDF_INPUT_MODULE.read_text(encoding="utf-8"))
    function = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_resolve_table_description_llm_config"
        ),
        None,
    )
    assert function is not None, "PDF 表格描述应提供 DashScope 配置解析函数"

    namespace = {"os": os}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PDF_INPUT_MODULE), "exec"), namespace)
    return namespace[function.name]


def test_pdf_table_description_uses_dashscope_defaults_without_config_base_url(monkeypatch):
    resolver = _load_table_description_config_resolver()
    monkeypatch.delenv("DASHSCOPE_COMPATIBLE_BASE_URL", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_CHAT_MODEL", raising=False)

    api_key, base_url, model = resolver(SimpleNamespace(base_url=None, table_description_model=None))

    assert api_key is None
    assert base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert model == "qwen3.7-plus"


def test_pdf_table_description_reads_dashscope_environment_when_config_is_empty(monkeypatch):
    resolver = _load_table_description_config_resolver()
    monkeypatch.setenv("DASHSCOPE_COMPATIBLE_BASE_URL", "https://dashscope.example/compatible-mode/v1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-dashscope-key")
    monkeypatch.setenv("DASHSCOPE_CHAT_MODEL", "qwen3.7-plus-test")

    api_key, base_url, model = resolver(SimpleNamespace(base_url="", table_description_model=""))

    assert api_key == "test-dashscope-key"
    assert base_url == "https://dashscope.example/compatible-mode/v1"
    assert model == "qwen3.7-plus-test"


def test_project_level_graphrag_settings_use_dashscope_environment_variables():
    settings = (GRAPHRAG_ROOT / "settings.yaml").read_text(encoding="utf-8")

    assert "api_base: ${DASHSCOPE_COMPATIBLE_BASE_URL}" in settings
    assert "api_key: ${DASHSCOPE_API_KEY}" in settings
    assert "model: ${DASHSCOPE_CHAT_MODEL}" in settings
    assert "api.deepseek.com" not in settings
    assert "deepseek-chat" not in settings
