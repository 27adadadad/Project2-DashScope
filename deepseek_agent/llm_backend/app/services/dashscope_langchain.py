"""LangGraph 使用的 DashScope 兼容 OpenAI 模型工厂。"""

from langchain_openai import ChatOpenAI

from app.core.config import settings


def create_agent_model(tags: list[str]) -> ChatOpenAI:
    """创建关闭思考模式的常规 Qwen agent 模型。"""
    return ChatOpenAI(
        model=settings.DASHSCOPE_CHAT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_COMPATIBLE_BASE_URL,
        temperature=0.7,
        tags=tags,
        extra_body={"enable_thinking": False},
    )
