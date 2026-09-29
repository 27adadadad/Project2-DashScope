from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent.parent
ENV_FILE = ROOT_DIR / ".env"


class Settings(BaseSettings):
    # DashScope settings
    DASHSCOPE_API_KEY: str
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/api/v1"
    DASHSCOPE_COMPATIBLE_BASE_URL: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    DASHSCOPE_CHAT_MODEL: str = "qwen3.7-plus"
    DASHSCOPE_REASON_MODEL: str = "qwen3.7-plus"
    DASHSCOPE_VISION_MODEL: str = "qwen3-vl-plus"
    DASHSCOPE_EMBEDDING_MODEL: str = "qwen3.7-text-embedding"

    # Search settings
    SERPAPI_KEY: str
    SEARCH_RESULT_COUNT: int = 3

    # Database settings
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    # Neo4j settings
    NEO4J_URL: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"

    # JWT settings
    SECRET_KEY: str = "your-secret-key"  # 在生产环境中使用安全的密钥
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS 允许的前端来源；生产环境为同源部署，跨域主要服务本地开发
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    # Redis settings
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_CACHE_EXPIRE: int = 3600
    # 与 .env.example 保持一致，避免未显式配置时命中率/准确率预期出现偏差
    REDIS_CACHE_THRESHOLD: float = 0.90

    # GraphRAG settings
    GRAPHRAG_PROJECT_DIR: str = "app/graphrag"  # GraphRAG项目目录
    GRAPHRAG_DATA_DIR: str = "data"  # 数据目录名称
    GRAPHRAG_QUERY_TYPE: str = "local"  # 查询类型
    GRAPHRAG_RESPONSE_TYPE: str = "text"  # 响应类型
    GRAPHRAG_COMMUNITY_LEVEL: int = 3  # 社区级别
    GRAPHRAG_DYNAMIC_COMMUNITY: bool = False  # 是否动态选择社区

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    @property
    def DATABASE_URL(self) -> str:
        return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def REDIS_URL(self) -> str:
        """构建Redis URL"""
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def NEO4J_CONN_URL(self) -> str:
        """构建Neo4j连接URL"""
        return self.NEO4J_URL

    @property
    def uses_default_secret_key(self) -> bool:
        """是否仍在用示例密钥签发 JWT（用于启动时告警）。"""
        return self.SECRET_KEY == "your-secret-key"


settings = Settings()
