import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logger import get_logger
from app.core.request_context import set_request_id

logger = get_logger(service="http")

REQUEST_ID_HEADER = "X-Request-ID"


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # perf_counter 单调递增且不受系统时钟调整影响，比 time.time() 更适合测耗时
        start_time = time.perf_counter()

        # 优先复用上游/前端传入的请求 ID，便于把一次调用的多条日志串起来；没有则生成
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        set_request_id(request_id)

        response = await call_next(request)

        # 计算处理时间
        process_time = time.perf_counter() - start_time
        response.headers[REQUEST_ID_HEADER] = request_id

        # 记录请求日志
        client = request.client
        client_label = f"{client.host}:{client.port}" if client else "-"
        logger.info(
            f"[{request_id}] {client_label} - "
            f"\"{request.method} {request.url.path} HTTP/{request.scope.get('http_version', '1.1')}\" "
            f"{response.status_code} - {process_time:.2f}s"
        )

        return response
