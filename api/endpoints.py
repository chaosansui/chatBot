import time
from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from fastapi.responses import StreamingResponse, JSONResponse
from models.api_models import SimpleChatRequest, SimpleChatResponse, HealthResponse, SessionInfo
from services.llm_service import llm_service
from storage.session_store import session_store
from core.config import settings
# 只需要 astream_rag，因为我们将所有聊天转换为流式
from core.chains import astream_rag 
from loguru import logger

import uuid
import json
from typing import AsyncIterator, Optional, Any

# 初始化路由
router = APIRouter()

async def _prepare_session(session_id: Optional[str], user_id_card: Optional[str] = None) -> str:
    """确保会话ID存在并返回它。"""
    if not session_id:

        session_id = str(uuid.uuid4())
        await session_store.create_session(session_id, user_id=user_id_card)
    else:

        session = await session_store.get_or_create_session(session_id)
    return session_id


async def _streaming_handler(request: SimpleChatRequest) -> StreamingResponse:
    """内部流式处理函数，被所有 POST 聊天路由调用。"""
    if not llm_service.is_ready:
        raise HTTPException(status_code=503, detail="模型服务暂不可用")
    
    # 1. 准备会话ID
    session_id = await _prepare_session(request.session_id, request.user_id_card)
    
    # 2. 定义异步生成器函数
    async def stream_rag_response() -> AsyncIterator[bytes]:
        
        # 调用 core/chains.py 中的 astream_rag 函数
        stream_generator = astream_rag(
            question=request.message,
            session_id=session_id,
            user_id_card=request.user_id_card
        )
        
        try:
            async for chunk_text in stream_generator:
                if chunk_text:
                    # 格式化为 SSE 格式：data: <json_payload>\n\n
                    sse_payload = json.dumps({"content": chunk_text}, ensure_ascii=False)
                    yield f"data: {sse_payload}\n\n".encode("utf-8")

        except Exception as e:
            logger.error("❌ 流式聊天处理失败 (Session: {}): {}", session_id, e, exc_info=True) 
            
            error_payload = json.dumps({"error": f"服务处理失败: {str(e)}"}, ensure_ascii=False)
            yield f"data: {error_payload}\n\n".encode("utf-8")
        finally:
            # 3. 发送流结束标记
            yield f"data: [DONE]\n\n".encode("utf-8")
            logger.info(f"会话 {session_id} 流式响应结束。")

    # 返回 StreamingResponse
    return StreamingResponse(
        content=stream_rag_response(),
        media_type="text/event-stream"
    )


# 1. 统一的 /chat/stream 端点 (保持原有名称)
@router.post("/chat/stream", name="chat_stream", tags=["Chat"])
async def chat_stream_endpoint(request: SimpleChatRequest):
    """
    流式聊天端点：返回 Server-Sent Events (SSE) 格式的文本流。
    """
    return await _streaming_handler(request)


# 2. 将 /chat (原同步接口) 切换为流式，以兼容 VLLMChatModel
@router.post("/chat", name="chat", tags=["Chat"])
async def chat_streaming_compatibility(request: SimpleChatRequest):
    """
    兼容性端点：将原同步聊天接口重定向到流式处理，解决 LLM 的 _astream 限制。
    """
    return await _streaming_handler(request)


# 3. 根端点 POST 快捷方式
@router.post("/")
async def chat_root_shortcut(request: SimpleChatRequest):
    """
    新增 POST / 快捷方式。将请求转发给流式聊天端点。
    """
    return await _streaming_handler(request)

# ----------------------------------------------------
# 健康检查
# ----------------------------------------------------
@router.get("/health", response_model=HealthResponse, tags=["Monitor"])
async def health_check():
    """健康检查端点"""
    model_ready = await llm_service.health_check()
    
    return HealthResponse(
        status="healthy" if model_ready else "degraded",
        version=settings.VERSION,
        model_ready=model_ready
    )

# ----------------------------------------------------
# 会话管理和根 GET 端点
# ----------------------------------------------------

@router.get("/sessions/{session_id}", response_model=SessionInfo, tags=["Session"])
async def get_session(session_id: str):
    """获取会话信息"""
    session = await session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    return session

@router.delete("/sessions/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    """删除会话"""
    await session_store.delete_session(session_id)
    return {"message": "会话删除成功"}

@router.get("/", tags=["Monitor"])
async def root():
    """根端点：获取应用信息"""
    model_ready = await llm_service.health_check()
    
    return {
        "message": f"欢迎使用 {settings.APP_NAME}",
        "version": settings.VERSION,
        "status": "running",
        "model_status": "ready" if model_ready else "unavailable",
    }