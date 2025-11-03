from fastapi import APIRouter, HTTPException
from models.api_models import SimpleChatRequest, SimpleChatResponse, HealthResponse, SessionInfo
from services.llm_service import llm_service
from storage.session_store import session_store
from core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    model_ready = await llm_service.health_check()
    
    return HealthResponse(
        status="healthy" if model_ready else "degraded",
        version=settings.VERSION,
        model_ready=model_ready
    )

@router.post("/chat", response_model=SimpleChatResponse)
async def chat(request: SimpleChatRequest):
    """聊天端点（支持会话管理）"""
    try:
        # 获取或创建会话
        if request.session_id:
            session = await session_store.get_or_create_session(request.session_id)
            # 获取会话历史
            conversation_history = await session_store.get_session_messages(
                request.session_id, 
                limit=5
            )
            # 转换为LLM需要的格式
            history_for_llm = [
                {"role": msg.role, "content": msg.content} 
                for msg in conversation_history
            ]
        else:
            # 如果没有session_id，创建临时会话
            import uuid
            request.session_id = f"temp_{uuid.uuid4().hex[:8]}"
            session = await session_store.create_session(request.session_id)
            history_for_llm = []
        
        # 调用LLM服务
        response = await llm_service.chat(
            message=request.message,
            session_id=request.session_id,
            conversation_history=history_for_llm
        )
        
        # 保存消息到会话历史
        if request.session_id:
            await session_store.add_message(
                request.session_id, 
                "user", 
                request.message
            )
            await session_store.add_message(
                request.session_id, 
                "assistant", 
                response.answer
            )
        
        return response
        
    except Exception as e:
        logger.error(f"聊天处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"服务处理失败: {str(e)}")

@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """获取会话信息"""
    session = await session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    await session_store.delete_session(session_id)
    return {"message": "会话删除成功"}

@router.get("/")
async def root():
    """根端点"""
    return {
        "message": f"欢迎使用 {settings.APP_NAME}",
        "version": settings.VERSION,
        "status": "running",
        "model_status": "ready" if llm_service.is_ready else "unavailable",
        "session_count": len(session_store.sessions)
    }