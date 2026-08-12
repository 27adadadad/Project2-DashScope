from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "app"
FORBIDDEN_TOKENS = (
    "langchain_deepseek",
    "langchain_ollama",
    "DEEPSEEK_",
    "OLLAMA_",
    "ChatDeepSeek",
    "ChatOllama",
    "deepseek_service",
    "ollama_service",
)


def test_runtime_has_no_legacy_provider_references_outside_vendored_graphrag():
    for path in RUNTIME_ROOT.rglob("*.py"):
        if "graphrag" in path.relative_to(RUNTIME_ROOT).parts:
            continue
        content = path.read_text(encoding="utf-8")
        assert not any(token in content for token in FORBIDDEN_TOKENS), path


def test_legacy_provider_service_and_manual_test_files_are_removed():
    for relative_path in (
        "services/deepseek_service.py",
        "services/ollama_service.py",
        "test/deepseek_sync.py",
        "test/deepseek_stream.py",
        "test/deepseek_disk_cache.py",
        "test/ollama_benchmark.py",
    ):
        assert not (RUNTIME_ROOT / relative_path).exists(), relative_path


def test_graphrag_embedding_configs_use_dashscope_compatible_1024_dimensions():
    data_dir = RUNTIME_ROOT / "graphrag" / "data"
    for filename in ("settings.yaml", "settings_csv.yaml", "settings_pdf.yaml"):
        content = (data_dir / filename).read_text(encoding="utf-8")
        embedding_section = content.split("default_embedding_model:", 1)[1].split(
            "vector_store:", 1
        )[0]
        assert "${DASHSCOPE_COMPATIBLE_BASE_URL}" in embedding_section
        assert "${DASHSCOPE_API_KEY}" in embedding_section
        assert "${DASHSCOPE_EMBEDDING_MODEL}" in embedding_section
        assert "dimensions: 1024" in embedding_section
        assert "deepseek" not in content.lower()
        assert "ollama" not in content.lower()
