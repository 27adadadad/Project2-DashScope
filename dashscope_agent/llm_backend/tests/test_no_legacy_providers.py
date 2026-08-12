from pathlib import Path
import subprocess


RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "app"
BACKEND_ROOT = RUNTIME_ROOT.parent
PROJECT_ROOT = BACKEND_ROOT.parent.parent
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
FORBIDDEN_PROVIDER_NAMES = ("deepseek", "ollama")


def _is_vendored_graphrag(path: Path) -> bool:
    return "graphrag" in path.relative_to(BACKEND_ROOT).parts


def _has_legacy_provider_name(path: Path) -> bool:
    """忽略保留的包目录名，仅检查实际文件与其他目录名称。"""
    parts = [part.lower() for part in path.relative_to(PROJECT_ROOT).parts]
    return any(
        provider in part
        for provider in FORBIDDEN_PROVIDER_NAMES
        for part in parts
    )


def _tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "ls-files"], text=True
    )
    return [PROJECT_ROOT / relative_path for relative_path in output.splitlines()]


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


def test_controlled_file_names_have_no_legacy_provider_names_outside_vendor():
    for path in _tracked_files():
        if path.is_relative_to(RUNTIME_ROOT / "graphrag"):
            continue
        assert not _has_legacy_provider_name(path), path


def test_runtime_and_current_docs_have_no_legacy_provider_content():
    paths = [
        *(path for path in RUNTIME_ROOT.rglob("*.py") if not _is_vendored_graphrag(path)),
        BACKEND_ROOT.parent / "requirements.txt",
        BACKEND_ROOT.parent / ".env.example",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "dashscope_agent" / "README.md",
    ]
    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert not any(
            provider in content.lower() for provider in FORBIDDEN_PROVIDER_NAMES
        ), path


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


def test_top_level_agent_directory_uses_dashscope_name():
    assert (PROJECT_ROOT / "dashscope_agent").is_dir()
    assert not (PROJECT_ROOT / "deepseek_agent").exists()
