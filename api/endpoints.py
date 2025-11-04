from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from models.api_models import SimpleChatRequest, SimpleChatResponse, HealthResponse, SessionInfo
from services.llm_service import llm_service
from storage.session_store import session_store
from core.config import settings
from core.chains import full_rag_chain, update_chat_history # 导入 RAG Chain 和历史更新函数
from loguru import logger # 推荐使用 loguru 作为全局 logger

import uuid
import json
from typing import AsyncIterator

router = APIRouter()

# ----------------------------------------------------
# 辅助函数：处理会话ID和历史记录
# ----------------------------------------------------
async def _prepare_session(session_id: Optional[str]) -> str:
    """确保会话ID存在并返回它。"""
    if not session_id:
        # 临时会话ID
        session_id = f"temp_{uuid.uuid4().hex[:8]}"
        await session_store.create_session(session_id)
    else:
        # 确保会话存在
        await session_store.get_or_create_session(session_id)
    return session_id


# ----------------------------------------------------
# 1. 健康检查 (保持不变)
# ----------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    model_ready = await llm_service.health_check()
    
    return HealthResponse(
        status="healthy" if model_ready else "degraded",
        version=settings.VERSION,
        model_ready=model_ready
    )

# ----------------------------------------------------
# 2. 流式聊天端点 (新增的核心接口)
# ----------------------------------------------------
@router.post("/chat/stream")
async def chat_stream(request: SimpleChatRequest, background_tasks: BackgroundTasks):
    """
    流式聊天端点：使用 LangChain .astream() 返回 Server-Sent Events (SSE) 格式的文本流。
    同时使用 BackgroundTasks 在流式结束后保存历史记录。
    """
    if not llm_service.is_ready:
        raise HTTPException(status_code=503, detail="模型服务暂不可用")
    
    # 1. 准备会话ID
    session_id = await _prepare_session(request.session_id)
    user_question = request.message
    
    # 2. 定义异步生成器函数，用于处理 LangChain 的流式输出
    async def stream_rag_response() -> AsyncIterator[bytes]:
        full_answer = ""
        full_context = []
        
        # 组装 Chain 输入
        chain_input = {
            "question": user_question,
            "session_id": session_id # 传入 session_id 用于加载历史和检索
        }
        
        # 使用 .astream() 调用 RAG Chain
        async for chunk in full_rag_chain.astream(chain_input):
            
            # 提取文本增量
            text_chunk = chunk.get("answer", "")
            
            # 提取上下文（通常在第一个块或最后一个块）
            if chunk.get("context"):
                full_context = chunk["context"]

            if text_chunk:
                full_answer += text_chunk
                
                # 格式化为 SSE 格式：data: <json_payload>\n\n
                sse_payload = json.dumps({"content": text_chunk})
                yield f"data: {sse_payload}\n\n".encode("utf-8")

        # 3. 流式结束后，使用后台任务保存历史记录
        # 注意：这里需要确保 full_answer 包含了完整的模型响应
        if full_answer:
            background_tasks.add_task(
                update_chat_history, 
                session_id, 
                user_question, 
                full_answer
            )
            logger.info(f"会话 {session_id} 历史记录更新任务已加入后台。")
        
        # 4. 发送流结束标记
        yield f"data: [DONE]\n\n".encode("utf-8")

    # 返回 StreamingResponse，使用 text/event-stream 媒体类型
    return StreamingResponse(
        content=stream_rag_response(),
        media_type="text/event-stream"
    )

# ----------------------------------------------------
# 3. 同步聊天端点 (调用流式逻辑进行聚合)
# ----------------------------------------------------
@router.post("/chat", response_model=SimpleChatResponse)
async def chat(request: SimpleChatRequest):
    """
    聊天端点（聚合流式输出）：保留同步接口，但底层使用流式逻辑并等待结果。
    """
    start_time = time.time()
    
    # 1. 准备会话ID
    session_id = await _prepare_session(request.session_id)
    user_question = request.message
    
    if not llm_service.is_ready:
        raise HTTPException(status_code=503, detail="模型服务暂不可用")

    full_answer = ""
    
    # 2. 组装 Chain 输入
    chain_input = {
        "question": user_question,
        "session_id": session_id
    }
    
    try:
        # 使用 .astream() 迭代，并聚合结果
        async for chunk in full_rag_chain.astream(chain_input):
            full_answer += chunk.get("answer", "")
            # 忽略 context 信息的聚合，因为这里是 SimpleChatResponse

        # 3. 保存历史记录 (同步等待保存)
        await update_chat_history(session_id, user_question, full_answer)

        processing_time = time.time() - start_time
        logger.info(f"同步聊天成功 - Session: {session_id}, 耗时: {processing_time:.2f}s")
        
        return SimpleChatResponse(
            answer=full_answer,
            session_id=session_id,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"同步聊天处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务处理失败: {str(e)}")


# ----------------------------------------------------
# 4. 会话管理和根端点 (保持不变，但日志使用 loguru)
# ----------------------------------------------------
# (代码结构保持不变，只是将 logger 替换为 loguru.logger if needed)

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
    # 假设 session_store.sessions 属性不再是公开的，需要通过方法获取会话数量
    try:
        session_count = await session_store.count_sessions()
    except Exception:
        session_count = "N/A" # 处理无法获取会话数量的情况
        
    return {
        "message": f"欢迎使用 {settings.APP_NAME}",
        "version": settings.VERSION,
        "status": "running",
        "model_status": "ready" if llm_service.is_ready else "unavailable",
        "session_count": session_count
    }