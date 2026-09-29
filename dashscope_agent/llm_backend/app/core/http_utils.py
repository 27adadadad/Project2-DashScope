"""HTTP 层的通用校验工具：鉴权归属、上传安全、请求体判定。

这里都是不依赖 FastAPI 应用实例的纯函数，因此可以脱离路由单独测试。
"""

import os
import re
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile

from app.models.user import User

# 上传安全策略：白名单扩展名 + 体积上限，避免可执行文件落盘与磁盘被打满
ALLOWED_DOCUMENT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".pdf", ".csv", ".json",
    ".doc", ".docx", ".xls", ".xlsx",
}
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
# 客户端传入的会话ID只允许字母/数字/下划线/短横线，避免被拼进目录名
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def require_same_user(claimed_user_id: int, current_user: User) -> None:
    """校验请求声明的 user_id 与 JWT 身份一致，堵住“登录后改别人的ID”这类横向越权。"""
    if claimed_user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权访问其他用户的数据")


def is_independent_question(messages: list) -> bool:
    """仅让没有上下文依赖的单条用户问题参与语义缓存。"""
    return (
        len(messages) == 1
        and messages[0].get("role") == "user"
        and bool(messages[0].get("content", "").strip())
    )


def sanitize_filename(filename: Optional[str], allowed_suffixes: set) -> tuple:
    """剥离客户端路径、校验扩展名、替换危险字符，返回 (文件名主体, 小写扩展名)。"""
    if not filename:
        raise HTTPException(status_code=400, detail="缺少文件名")
    # basename 同时处理 Windows 的 \ 与 POSIX 的 /，阻断 ../../ 之类的相对路径
    base_name = os.path.basename(filename.replace("\\", "/"))
    stem, ext = os.path.splitext(base_name)
    ext = ext.lower()
    if ext not in allowed_suffixes:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{ext or '无扩展名'}")
    # 只保留字母/数字/下划线/连字符/中文，顺带保证不含正则元字符
    stem = re.sub(r"[^\w\-]", "_", stem, flags=re.UNICODE).strip("_")[:80]
    return (stem or "file"), ext


def ensure_within_root(root: Path, target: Path) -> Path:
    """确保最终落盘路径位于允许的根目录内，作为路径穿越的第二道防线。"""
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError:
        raise HTTPException(status_code=400, detail="非法的文件路径") from None
    return target_resolved


async def save_upload(upload: UploadFile, dest: Path, max_bytes: int) -> int:
    """分块写盘并限制体积；超限立即中止并清理残留文件，避免整文件读入内存。"""
    total = 0
    try:
        with open(dest, "wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="文件超过大小限制")
                handle.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    return total
