"""解析内嵌 GraphRAG 项目的运行路径。"""

from pathlib import Path


def resolve_graphrag_project_dir(configured_path: str | Path) -> Path:
    """优先使用存在的配置路径，否则回退到本项目内嵌的 GraphRAG。"""
    project_dir = Path(configured_path)
    embedded_project_dir = Path(__file__).resolve().parents[1] / "graphrag"

    # 旧配置以 dashscope_agent 为工作目录；run.py 实际会切到 llm_backend，
    # 因此该相对路径会额外多出一层 llm_backend。
    if project_dir.as_posix() == "llm_backend/app/graphrag":
        return embedded_project_dir

    if project_dir.is_dir():
        return project_dir
    if embedded_project_dir.is_dir():
        return embedded_project_dir

    return project_dir
