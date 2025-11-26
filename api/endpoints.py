import os
import shutil
import uuid
import json
from typing import AsyncIterator, Optional, List
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from loguru import logger

from models.api_models import SimpleChatRequest, HealthResponse, SessionInfo
from services.llm_service import llm_service
from services.rag_service import rag_service
from storage.session_store import session_store
from storage.vector_store import vector_store 
from core.config import settings

router = APIRouter()

# 定义临时文件存储目录
TEMP_DIR = "data/temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


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
        
        yield f"data: {json.dumps({'type': 'status', 'text': '正在理解上下文...'}, ensure_ascii=False)}\n\n".encode("utf-8")

        try:
            chain = rag_service.get_rag_chain(request.user_id_card)
            
            async for event in chain.astream_events(
                {"question": request.message, "chat_history": history},
                version="v2"
            ):
                kind = event["event"]
                name = event.get("name")
                
                if kind == "on_chat_model_stream":
                    chunk_content = event["data"]["chunk"].content
                    if not chunk_content:
                        continue

                    if name == "QuestionRewriter":
                        pass 

                    elif name == "AnswerGenerator":
                        full_response += chunk_content
                        payload = json.dumps({
                            "type": "content",
                            "text": chunk_content
                        }, ensure_ascii=False)
                        yield f"data: {payload}\n\n".encode("utf-8")

                elif kind == "on_retriever_start":
                     yield f"data: {json.dumps({'type': 'status', 'text': '正在检索知识库...'}, ensure_ascii=False)}\n\n".encode("utf-8")

                # 3. 捕获检索结束
                elif kind == "on_retriever_end":
                    docs = event["data"].get("output", [])
                    if docs:
                        found_sources = list(set(d.metadata.get("source", "未知") for d in docs))
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

@router.post("/chat/stream", name="chat_stream", tags=["Chat"])
async def chat_stream_endpoint(request: SimpleChatRequest):
    return await _streaming_handler(request)

@router.post("/chat", name="chat_compat", tags=["Chat"])
async def chat_compatibility(request: SimpleChatRequest):
    return await _streaming_handler(request)

@router.post("/", tags=["Chat"])
async def chat_root_shortcut(request: SimpleChatRequest):
    return await _streaming_handler(request)

async def _background_indexing(temp_file_path: str, user_id: str, user_name: str, original_filename: str):
    """后台任务：OCR -> 索引 -> 清理"""
    generated_md_path = None
    try:
        logger.info(f"🔄 [1/3] 开始 OCR 识别: {temp_file_path}")
        
        # A. 调用 OCR 服务
        # 注意：这里解包返回的两个值：内容 和 路径
        markdown_content, generated_md_path = await ocr_service.file_to_markdown(temp_file_path)
        
        # B. 准备元数据
        metadata = {
            "source": original_filename,
            "user_id_card": user_id,
            "user_name": user_name,
            "type": "ocr_document"
        }

        logger.info(f"🔄 [2/3] 开始向量化索引 ({len(markdown_content)} 字符)...")
        
        # C. 调用 Markdown 专用索引方法
        await vector_store.index_markdown_content(markdown_content, metadata)
        
        logger.success(f"✅ [3/3] 全流程处理完成！")
        
    except Exception as e:
        logger.error(f"❌ 后台处理失败: {e}")
    finally:
        # D. 清理工作 (非常重要！)
        # 1. 删除用户上传的原始文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        # 2. 删除 OCR 生成的 .md 文件 (因为已经存入数据库了，文件可以删掉节省空间)
        if generated_md_path and os.path.exists(generated_md_path):
            os.remove(generated_md_path)
            logger.debug(f"🗑️ 已清理临时 MD 文件: {generated_md_path}")

@router.post("/knowledge/upload", tags=["Knowledge"])
async def upload_knowledge_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(..., description="用户唯一标识"),
    user_name: str = Form(..., description="用户姓名"),
):

    allowed_exts = [".pdf", ".jpg", ".png", ".jpeg"] 
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"OCR 模式仅支持: {allowed_exts}")

    # 保存临时文件
    safe_filename = f"{user_id}_{uuid.uuid4().hex[:8]}{file_ext}"
    file_path = os.path.join(TEMP_DIR, safe_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"保存失败: {e}")
        raise HTTPException(status_code=500, detail="文件保存失败")

    # 启动后台任务
    # 注意传递 file.filename 用于记录原始文件名
    background_tasks.add_task(_background_indexing, file_path, user_id, user_name, file.filename)

    return {
        "message": "文件已接收，正在后台进行 DeepSeek OCR 识别与索引...",
        "filename": file.filename,
        "user_id": user_id
    }

@router.get("/health", response_model=HealthResponse, tags=["Monitor"])
async def health_check():
    model_ready = await llm_service.health_check()
    rag_ready = rag_service.collection is not None
    
    return HealthResponse(
        status="healthy" if (model_ready and rag_ready) else "degraded",
        version=settings.VERSION,
        model_ready=model_ready
    )

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