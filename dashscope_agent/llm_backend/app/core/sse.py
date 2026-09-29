"""SSE（Server-Sent Events）输出协议的唯一定义处。

前端约定：每个事件的负载都是形如 ``data: {"type": "...", ...}\\n\\n`` 的 JSON 文本，
用 ``type`` 字段区分 content/reasoning/sources/error/done 等语义。
所有服务都从这里生成事件，避免各自拼接 JSON 导致协议细节漂移。
"""

import json


def sse_event(event_type: str, **payload: object) -> str:
    """构造一条 SSE 事件文本。

    ``ensure_ascii=False`` 保留中文原文（更省带宽、便于排查），
    ``json.dumps`` 默认分隔符保证输出形如 ``{"type": "content", "content": "..."}``，
    与前端解析逻辑及既有测试断言保持一致。
    """
    return f"data: {json.dumps({'type': event_type, **payload}, ensure_ascii=False)}\n\n"
