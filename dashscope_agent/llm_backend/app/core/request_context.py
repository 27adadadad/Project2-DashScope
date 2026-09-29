"""请求级上下文：把请求 ID 绑定到当前协程，供日志与响应头共享。

使用 ``ContextVar`` 而不是模块级全局变量：FastAPI 用同一事件循环并发处理多个请求，
全局变量会被并发请求互相覆盖；ContextVar 由 asyncio 在协程/任务之间自动隔离，
set 之后只对当前请求的调用链可见。
"""

from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(request_id: str) -> None:
    _request_id.set(request_id)


def get_request_id() -> str:
    return _request_id.get()
