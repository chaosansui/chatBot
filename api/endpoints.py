import uuid
import json
from typing import AsyncIterator, Optional
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from loguru import logger

from models.api_models import SimpleChatRequest, HealthResponse, SessionInfo
from services.llm_service import llm_service
from services.rag_service import rag_service
from storage.session_store import session_store
from core.config import settings

router = APIRouter()

async def _prepare_session(session_id: Optional[str], user_id_card: Optional[str] = None) -> str:
    """确保会话ID存在"""
    if not session_id:
        session_id = str(uuid.uuid4())
        await session_store.create_session(session_id, user_id=user_id_card)
    else:
        await session_store.get_or_create_session(session_id)
    return session_id

async def _streaming_handler(request: SimpleChatRequest) -> StreamingResponse:
    if not llm_service.is_ready:
        raise HTTPException(status_code=503, detail="模型服务暂不可用")
    
    session_id = await _prepare_session(request.session_id, request.user_id_card)
    history = await session_store.get_history(session_id) 

    async def stream_generator() -> AsyncIterator[bytes]:
        full_response = ""
        found_sources = []
        
        # 发送一个初始状态
        yield f"data: {json.dumps({'type': 'status', 'text': '正在理解上下文...'}, ensure_ascii=False)}\n\n".encode("utf-8")

        try:
            chain = rag_service.get_rag_chain(request.user_id_card)
            
            async for event in chain.astream_events(
                {"question": request.message, "chat_history": history},
                version="v2"
            ):
                kind = event["event"]
                name = event.get("name")
                
                # 1. 捕获 LLM 输出
                if kind == "on_chat_model_stream":
                    chunk_content = event["data"]["chunk"].content
                    if not chunk_content:
                        continue

                    # 如果是 "改写问题" 的 LLM 在输出 -> 视为 "思考中"
                    if name == "QuestionRewriter":
                        # 这里可以选择不把改写的内容发给前端，只发状态
                        # 或者发给前端显示在 "思考过程" 的折叠框里
                        pass 

                    # 如果是 "生成答案" 的 LLM 在输出 -> 视为 "正文"
                    elif name == "AnswerGenerator":
                        full_response += chunk_content
                        payload = json.dumps({
                            "type": "content",  # 标记为正文
                            "text": chunk_content
                        }, ensure_ascii=False)
                        yield f"data: {payload}\n\n".encode("utf-8")

                # 2. 捕获检索器开始工作
                elif kind == "on_retriever_start":
                     yield f"data: {json.dumps({'type': 'status', 'text': '正在检索知识库...'}, ensure_ascii=False)}\n\n".encode("utf-8")

                # 3. 捕获检索结束
                elif kind == "on_retriever_end":
                    docs = event["data"].get("output", [])
                    if docs:
                        found_sources = list(set(d.metadata.get("source", "未知") for d in docs))
                        # 告诉前端检索到了什么
                        msg = f"已找到 {len(docs)} 篇相关文档"
                        yield f"data: {json.dumps({'type': 'status', 'text': msg}, ensure_ascii=False)}\n\n".encode("utf-8")

            # 4. 循环结束后，发送引用源
            if found_sources:
                sources_payload = json.dumps({
                    "type": "sources",
                    "data": found_sources
                }, ensure_ascii=False)
                yield f"data: {sources_payload}\n\n".encode("utf-8")

            # 5. 保存历史
            await session_store.add_message(session_id, "human", request.message)
            await session_store.add_message(session_id, "ai", full_response)

        except Exception as e:
            logger.error(f"流式异常: {e}")
            err_payload = json.dumps({"type": "error", "text": str(e)}, ensure_ascii=False)
            yield f"data: {err_payload}\n\n".encode("utf-8")
        finally:
            yield f"data: [DONE]\n\n".encode("utf-8")

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
# ----------------------------------------------------
# 路由定义
# ----------------------------------------------------

@router.post("/chat/stream", name="chat_stream", tags=["Chat"])
async def chat_stream_endpoint(request: SimpleChatRequest):
    return await _streaming_handler(request)

@router.post("/chat", name="chat_compat", tags=["Chat"])
async def chat_compatibility(request: SimpleChatRequest):
    return await _streaming_handler(request)

@router.post("/", tags=["Chat"])
async def chat_root_shortcut(request: SimpleChatRequest):
    return await _streaming_handler(request)

@router.get("/health", response_model=HealthResponse, tags=["Monitor"])
async def health_check():
    model_ready = await llm_service.health_check()
    # 简单的 RAG 状态检查
    rag_ready = rag_service.collection is not None
    
    return HealthResponse(
        status="healthy" if (model_ready and rag_ready) else "degraded",
        version=settings.VERSION,
        model_ready=model_ready
    )

# ----------------------------------------------------
# 会话管理
# ----------------------------------------------------

@router.get("/sessions/{session_id}", response_model=SessionInfo, tags=["Session"])
async def get_session(session_id: str):
    session = await session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session

@router.delete("/sessions/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    await session_store.delete_session(session_id)
    return {"message": "会话删除成功"}

@router.get("/", tags=["Monitor"])
async def root():
    return {
        "app": settings.APP_NAME,
        "status": "running",
        "docs": "/docs"
    }