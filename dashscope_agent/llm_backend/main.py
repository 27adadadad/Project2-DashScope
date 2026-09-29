import asyncio
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command

from app.api import api_router
from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.http_utils import (
    ALLOWED_DOCUMENT_SUFFIXES,
    ALLOWED_IMAGE_SUFFIXES,
    MAX_DOCUMENT_BYTES,
    MAX_IMAGE_BYTES,
    SAFE_ID_PATTERN,
    ensure_within_root,
    is_independent_question,
    require_same_user,
    sanitize_filename,
    save_upload,
)
from app.core.logger import get_logger, log_structured
from app.core.middleware import LoggingMiddleware
from app.core.security import get_current_user
from app.core.sse import sse_event
from app.lg_agent.lg_builder import graph
from app.lg_agent.lg_states import InputState
from app.lg_agent.utils import new_uuid
from app.models.user import User
from app.schemas.chat import (
    ChatMessage,
    CreateConversationRequest,
    LangGraphResumeRequest,
    RAGChatRequest,
    ReasonRequest,
    UpdateConversationNameRequest,
)
from app.services.conversation_service import ConversationService
from app.services.dashscope_service import DashScopeServiceError
from app.services.graphrag_retriever import GraphRAGRetriever
from app.services.indexing_service import IndexingService
from app.services.model_service_factory import ModelServiceFactory
from app.services.rag_chat_service import RAGChatService
from app.services.redis_semantic_cache import (
    RedisSemanticCache,
    get_shared_redis_client,
)
from app.services.search_service import SearchService


# 配置上传目录 - RAG 功能的
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# logger 变量就被初始化为一个日志记录器实例。
# 之后，便可以在当前文件中直接使用 logger.info()、logger.error() 等方法来记录日志，而不需要进行其他操作。
logger = get_logger(service="main")

# GraphRAG 索引构建耗时以分钟计且以 CPU/磁盘为主。若直接在事件循环里 await，
# 构建期间的同步阶段会卡住整个进程；交给 asyncio 默认线程池又会和其他阻塞调用抢占。
# 因此单独开一个小线程池：容量固定，既不阻塞事件循环，也不会拖垮其它接口。
INDEXING_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="graphrag-index")


def _run_indexing_in_worker(indexing_service: IndexingService, file_info: dict) -> dict:
    """在工作线程里用独立事件循环执行索引，把阻塞隔离在主事件循环之外。"""
    return asyncio.run(indexing_service.process_file(file_info))


def _user_image_dir(user_id: int) -> Path:
    """图片按用户隔离存放，避免不同用户互相覆盖或读取。"""
    user_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"user_{user_id}"))
    image_dir = UPLOAD_DIR / "images" / user_uuid
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir


def _scoped_thread_id(user_id: int, conversation_id: Optional[str]) -> str:
    """把客户端会话ID收敛到当前用户命名空间，避免跨用户读取同一 LangGraph 线程状态。"""
    prefix = f"u{user_id}_"
    if not conversation_id:
        return prefix + new_uuid()
    raw_id = conversation_id[len(prefix):] if conversation_id.startswith(prefix) else conversation_id
    if not SAFE_ID_PATTERN.match(raw_id):
        raise HTTPException(status_code=400, detail="非法的会话ID")
    return prefix + raw_id


async def _stream_langgraph_messages(stream, thread_config: dict, thread_id: str, *, check_interrupt: bool):
    """转发 LangGraph 输出，并将执行错误作为安全 SSE 事件返回。"""
    try:
        async for chunk, metadata in stream:
            additional_kwargs = getattr(chunk, "additional_kwargs", {})
            if (
                chunk.content
                and "research_plan" not in metadata.get("tags", [])
                and not additional_kwargs.get("tool_calls")
            ):
                yield sse_event("content", content=chunk.content)
            elif additional_kwargs.get("tool_calls"):
                tool_data = additional_kwargs["tool_calls"][0]["function"].get("arguments")
                logger.debug(f"Tool call: {tool_data}")

        if check_interrupt:
            state = graph.get_state(thread_config)
            if len(state) > 0 and len(state[-1]) > 0:
                if len(state[-1][0].interrupts) > 0:
                    yield sse_event("interruption", conversation_id=thread_id)
        yield sse_event("done")
    except Exception:
        logger.error("LangGraph stream failed for conversation %s", thread_id, exc_info=True)
        yield sse_event("error", message="LangGraph 流式处理失败。")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """进程退出时回收索引线程池，避免长任务把进程退出卡住。"""
    yield
    INDEXING_EXECUTOR.shutdown(wait=False, cancel_futures=True)


# 创建 FastAPI 应用实例
app = FastAPI(title="AssistGen REST API", lifespan=lifespan)

# JWT 密钥仍是示例值时，任何人都能伪造令牌，启动即告警
if settings.uses_default_secret_key:
    logger.warning("SECRET_KEY 仍是默认示例值，请在生产环境的 .env 中改用强随机密钥")

# 添加日志中间件， 使用 LoggingMiddleware 来统一处理日志记录，从而替代 FastAPI 的原生打印日志。
app.add_middleware(LoggingMiddleware)

# CORS设置
# allow_origins=["*"] 与 allow_credentials=True 是浏览器规范禁止的组合，
# 且本项目认证靠请求头里的 Bearer Token（而非 Cookie），因此改为来源白名单 + 关闭凭证。
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 用户注册、登录路由通过 api_router 路由挂载到 /api 前缀
app.include_router(api_router, prefix="/api")

async def _stream_chat_with_semantic_cache(request: ChatMessage, user_id: int):
    """缓存命中直接返回；未命中时转发模型流并缓存完整可见回答。"""
    cache = None
    try:
        cache = RedisSemanticCache(
            user_id=user_id,
            start_auto_cleanup=False,
        )
        cached_response = await cache.lookup(request.messages)
    except Exception:
        logger.warning("Semantic cache lookup failed; falling back to model", exc_info=True)
        cached_response = None

    if cached_response is not None:
        yield sse_event("content", content=cached_response)
        yield sse_event("done")
        return

    response_parts = []
    completed_normally = False
    chat_service = ModelServiceFactory.create_chat_service()
    async for event in chat_service.generate_stream(
        messages=request.messages,
        user_id=user_id,
        conversation_id=request.conversation_id,
        on_complete=ConversationService.save_message,
        thinking=False,
    ):
        if event.startswith("data: "):
            payload = json.loads(event[6:].strip())
            if payload.get("type") == "content":
                response_parts.append(payload.get("content", ""))
            elif payload.get("type") == "done":
                completed_normally = True
        yield event

    if cache is not None and completed_normally and response_parts:
        try:
            await cache.update(request.messages, "".join(response_parts))
        except Exception:
            logger.warning("Semantic cache update failed", exc_info=True)

@app.get("/health")
async def health_check():
    """存活探针：只表明进程还能响应，不代表依赖可用。"""
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check():
    """就绪探针：数据库或 Redis 不可用时返回 503，供负载均衡摘除实例。"""
    from sqlalchemy import text

    checks: dict[str, str] = {}
    ready = True
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        logger.warning("Readiness check failed for database", exc_info=True)
        checks["database"] = "unavailable"
        ready = False
    try:
        await get_shared_redis_client().ping()
        checks["redis"] = "ok"
    except Exception:
        # 语义缓存失效不影响聊天主链路，因此 Redis 只降级不摘除实例
        logger.warning("Readiness check failed for redis", exc_info=True)
        checks["redis"] = "degraded"
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ok" if ready else "not_ready", "checks": checks},
    )

@app.post("/api/chat")
async def chat_endpoint(
    request: ChatMessage,
    current_user: User = Depends(get_current_user),
):
    """聊天接口"""
    require_same_user(request.user_id, current_user)
    try:
        logger.info(f"Processing chat request for user {current_user.id} in conversation {request.conversation_id}")
        if is_independent_question(request.messages):
            stream = _stream_chat_with_semantic_cache(request, current_user.id)
        else:
            chat_service = ModelServiceFactory.create_chat_service()
            stream = chat_service.generate_stream(
                messages=request.messages,
                user_id=current_user.id,
                conversation_id=request.conversation_id,
                on_complete=ConversationService.save_message,
                thinking=False,
            )

        return StreamingResponse(
            stream,
            media_type="text/event-stream"
        )
    except DashScopeServiceError as error:
        logger.warning("Chat DashScope error: %s", error.category)
        raise HTTPException(status_code=502, detail=str(error)) from None
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reason")
async def reason_endpoint(
    request: ReasonRequest,
    current_user: User = Depends(get_current_user),
):
    """推理接口"""
    require_same_user(request.user_id, current_user)
    try:
        logger.info(f"Processing reasoning request for user {current_user.id}")
        reasoner = ModelServiceFactory.create_reasoner_service()
        
        log_structured("reason_request", {
            "user_id": current_user.id,
            "message_count": len(request.messages),
            "last_message": request.messages[-1]["content"][:100] + "..."
        })
        
        return StreamingResponse(
            reasoner.generate_stream(request.messages, thinking=True),
            media_type="text/event-stream"
        )
    
    except DashScopeServiceError as error:
        logger.warning("Reasoning DashScope error for user %s: %s", current_user.id, error.category)
        raise HTTPException(status_code=502, detail=str(error)) from None
    except Exception as e:
        logger.error(f"Reasoning error for user {current_user.id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search_endpoint(
    request: ChatMessage,
    current_user: User = Depends(get_current_user),
):
    """带搜索功能的聊天接口"""
    require_same_user(request.user_id, current_user)
    try:
        logger.info(f"Processing search request for user {current_user.id} in conversation {request.conversation_id}")
        search_service = SearchService()
        return StreamingResponse(
            search_service.generate_stream(
                query=request.messages[0]["content"],
                user_id=current_user.id,
                conversation_id=request.conversation_id,
                # on_complete=ConversationService.save_message
            ),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    current_user: User = Depends(get_current_user),
):
    """上传文件并准备 RAG 处理"""
    require_same_user(user_id, current_user)
    try:
        logger.info(f"Uploading file for user {user_id}: {file.filename}")

        # 0. 先校验文件名：去掉客户端路径、限制扩展名、清理危险字符
        stem, ext = sanitize_filename(file.filename, ALLOWED_DOCUMENT_SUFFIXES)

        # 1. 创建基于UUID的一级目录
        user_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"user_{user_id}"))
        first_level_dir = UPLOAD_DIR / user_uuid

        # 2. 创建基于时间戳的二级目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        second_level_dir = first_level_dir / timestamp
        second_level_dir.mkdir(parents=True, exist_ok=True)

        # 3. 生成带时间戳的文件名，并确认最终路径没有跳出上传根目录
        new_filename = f"{stem}_{timestamp}{ext}"
        file_path = ensure_within_root(UPLOAD_DIR, second_level_dir / new_filename)

        # 保存文件：分块落盘并限制体积，失败时清理半截文件
        size = await save_upload(file, file_path, MAX_DOCUMENT_BYTES)

        # 获取文件信息
        file_info = {
            "index_id": user_uuid,
            "filename": new_filename,
            "original_name": os.path.basename((file.filename or "").replace("\\", "/")),
            "size": size,
            "type": file.content_type,
            "path": str(file_path).replace('\\', '/'),
            "user_id": user_id,
            "user_uuid": user_uuid,
            "upload_time": timestamp,
            "directory": str(second_level_dir)
        }
        
        # 4. 处理文件索引：交给独立线程池，构建期间主事件循环仍可服务其它请求。
        # 请求仍需等待索引结束才返回，因为当前前端要求响应里直接带上 index_result。
        indexing_service = IndexingService()
        index_result = await asyncio.get_running_loop().run_in_executor(
            INDEXING_EXECUTOR, _run_indexing_in_worker, indexing_service, file_info
        )
        
        # 合并结果
        result = {**file_info, "index_result": index_result}
        
        return result
        
    except Exception as e:
        logger.error(f"Upload failed for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat-rag")
async def rag_chat_endpoint(
    request: RAGChatRequest,
    current_user: User = Depends(get_current_user),
):
    """基于文档的问答接口"""
    try:
        expected_index_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"user_{current_user.id}")
        )
        if request.index_id != expected_index_id:
            raise HTTPException(status_code=403, detail="无权访问该知识库索引")

        logger.info(f"Processing RAG chat request for user {current_user.id}")
        rag_chat_service = RAGChatService(retriever=GraphRAGRetriever())
        
        return StreamingResponse(
            rag_chat_service.generate_stream(
                request.messages,
                request.index_id
            ),
            media_type="text/event-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG chat error for user {current_user.id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversations")
async def create_conversation(
    request: CreateConversationRequest,
    current_user: User = Depends(get_current_user),
):
    """创建新会话"""
    require_same_user(request.user_id, current_user)
    try:
        conversation_id = await ConversationService.create_conversation(current_user.id)
        return {"conversation_id": conversation_id}
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/user/{user_id}")
async def get_user_conversations(
    user_id: int,
    current_user: User = Depends(get_current_user),
):
    """获取用户的所有会话"""
    require_same_user(user_id, current_user)
    try:
        conversations = await ConversationService.get_user_conversations(current_user.id)
        return conversations
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
):
    """获取会话的所有消息"""
    try:
        messages = await ConversationService.get_conversation_messages(
            conversation_id, current_user.id
        )
        return messages
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting messages: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
):
    """删除会话及其所有消息"""
    try:
        conversation_service = ConversationService()
        await conversation_service.delete_conversation(conversation_id, current_user.id)
        return {"message": "会话已删除"}
    except Exception as e:
        logger.error(f"删除会话失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/conversations/{conversation_id}/name")
async def update_conversation_name(
    conversation_id: int,
    request: UpdateConversationNameRequest,
    current_user: User = Depends(get_current_user),
):
    """修改会话名称"""
    try:
        conversation_service = ConversationService()
        await conversation_service.update_conversation_name(
            conversation_id, request.name, current_user.id
        )
        return {"message": "会话名称已更新"}
    except Exception as e:
        logger.error(f"更新会话名称失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/langgraph/query")
async def langgraph_query(
    query: str = Form(...),
    user_id: int = Form(...),
    conversation_id: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
):
    """使用LangGraph处理用户查询，支持图片上传"""
    require_same_user(user_id, current_user)
    try:
        logger.info(f"Processing LangGraph query for user {user_id} and conversation {conversation_id}")

        # 会话ID先收敛到当前用户命名空间，再做后续处理
        thread_id = _scoped_thread_id(current_user.id, conversation_id)

        # 处理图片上传
        image_path = None
        if image:
            stem, ext = sanitize_filename(image.filename, ALLOWED_IMAGE_SUFFIXES)

            # 生成带时间戳的文件名，并确认落在该用户的图片目录内
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{stem}_{timestamp}{ext}"
            image_dir = _user_image_dir(current_user.id)
            image_path = ensure_within_root(UPLOAD_DIR, image_dir / new_filename)

            # 保存图片：分块落盘并限制体积
            await save_upload(image, image_path, MAX_IMAGE_BYTES)

            logger.info(f"Saved image {new_filename} for user {user_id}")

        thread_config = {
            "configurable": {
                "thread_id": thread_id, 
                "user_id": user_id,
                "image_path": str(image_path) if image_path else None
            }
        }
        
        # 获取当前线程状态
        state_history = None
        try:
            # 检查是否有现有的会话状态
            if thread_id:
                state_history = graph.get_state(thread_config)
                if state_history:
                    logger.info(f"Found existing conversation state for thread_id: {thread_id}")
        except Exception as e:
            logger.warning(f"Error retrieving state: {e}. Starting with fresh state.")
        
        # 准备输入状态 - 如果是现有会话，直接传入查询文本
        if state_history and len(state_history) > 0 and len(state_history[-1]) > 0:
            logger.info("Using existing conversation state")
            stream = graph.astream(
                Command(resume=query), stream_mode="messages", config=thread_config
            )
        else:
            logger.info("Creating new conversation state")
            stream = graph.astream(
                input=InputState(messages=query), stream_mode="messages", config=thread_config
            )
        
        response = StreamingResponse(
            _stream_langgraph_messages(
                stream, thread_config, thread_id, check_interrupt=True
            ),
            media_type="text/event-stream"
        )
        
        # 添加会话ID到响应头，方便前端获取
        response.headers["X-Conversation-ID"] = thread_id
        
        return response
        
    except Exception as e:
        logger.error(f"LangGraph query error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/langgraph/resume")
async def langgraph_resume(
    request: LangGraphResumeRequest,
    current_user: User = Depends(get_current_user),
):
    """继续执行LangGraph流程"""
    require_same_user(request.user_id, current_user)
    try:
        logger.info(f"Resuming LangGraph query for user {current_user.id} with conversation {request.conversation_id}")

        # 与 query 接口使用同一套用户命名空间，避免续跑他人的线程
        thread_id = _scoped_thread_id(current_user.id, request.conversation_id)
        thread_config = {
            "configurable": {"thread_id": thread_id, "user_id": current_user.id}
        }

        return StreamingResponse(
            _stream_langgraph_messages(
                graph.astream(
                    Command(resume=request.query),
                    stream_mode="messages",
                    config=thread_config,
                ),
                thread_config,
                thread_id,
                check_interrupt=True,
            ),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"LangGraph resume error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload/image")
async def upload_image(
    image: UploadFile = File(...),
    user_id: int = Form(...),
    conversation_id: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
):
    """上传图片并返回图片存储路径"""
    require_same_user(user_id, current_user)
    try:
        stem, ext = sanitize_filename(image.filename, ALLOWED_IMAGE_SUFFIXES)

        # 图片按用户隔离存放；conversation_id 仅作为子目录名且必须通过格式校验
        image_dir = _user_image_dir(current_user.id)
        if conversation_id:
            if not SAFE_ID_PATTERN.match(conversation_id):
                raise HTTPException(status_code=400, detail="非法的会话ID")
            image_dir = image_dir / conversation_id
        image_dir.mkdir(parents=True, exist_ok=True)

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{stem}_{timestamp}{ext}"
        image_path = ensure_within_root(UPLOAD_DIR, image_dir / new_filename)

        # 保存图片：分块落盘并限制体积
        size = await save_upload(image, image_path, MAX_IMAGE_BYTES)

        # 获取图片信息
        image_info = {
            "filename": new_filename,
            "original_name": os.path.basename((image.filename or "").replace("\\", "/")),
            "size": size,
            "type": image.content_type,
            "path": str(image_path).replace('\\', '/'),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "upload_time": timestamp
        }
        
        logger.info(f"Image uploaded: {image_info}")
        
        return image_info
        
    except Exception as e:
        logger.error(f"Image upload failed for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 最后挂载静态文件，并确保使用绝对路径
STATIC_DIR = Path(__file__).parent / "static" / "dist"
if not (STATIC_DIR / "index.html").is_file() and (STATIC_DIR / "dist" / "index.html").is_file():
    STATIC_DIR = STATIC_DIR / "dist"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
