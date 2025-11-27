import os
import shutil
import uuid
import json
from datetime import datetime
from typing import AsyncIterator, Optional, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from loguru import logger

from models.api_models import SimpleChatRequest, HealthResponse, SessionInfo
from services.llm_service import llm_service
from services.rag_service import rag_service
from services.ocr_service import ocr_service
from storage.session_store import session_store
from storage.vector_store import vector_store 
from core.config import settings

router = APIRouter()

# Shared directory for uploaded files (Must match OCR service config)
SHARED_INPUT_DIR = "/mnt/data/AI-chatBot/data/temp_inputs"
os.makedirs(SHARED_INPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Chat Logic
# ----------------------------------------------------------------------

async def _prepare_session(session_id: Optional[str], user_id_card: Optional[str] = None) -> str:
    """Ensure session exists or create new one."""
    if not session_id:
        session_id = str(uuid.uuid4())
        await session_store.create_session(session_id, user_id=user_id_card)
    else:
        await session_store.get_or_create_session(session_id)
    return session_id

async def _streaming_handler(request: SimpleChatRequest) -> StreamingResponse:
    if not llm_service.is_ready:
        raise HTTPException(status_code=503, detail="LLM service unavailable")
    
    session_id = await _prepare_session(request.session_id)
    history = await session_store.get_history(session_id) 

    async def stream_generator() -> AsyncIterator[bytes]:
        full_response = ""
        found_sources = []
        
        # Initial status
        yield f"data: {json.dumps({'type': 'status', 'text': 'Understanding context...'}, ensure_ascii=False)}\n\n".encode("utf-8")

        try:
            chain = rag_service.get_rag_chain()
            
            async for event in chain.astream_events(
                {"question": request.message, "chat_history": history},
                version="v2"
            ):
                kind = event["event"]
                name = event.get("name")
                
                # 1. LLM Output Stream
                if kind == "on_chat_model_stream":
                    chunk_content = event["data"]["chunk"].content
                    if not chunk_content:
                        continue

                    # Question Rewriter output
                    if name == "QuestionRewriter":
                        pass 

                    # Final Answer output
                    elif name == "AnswerGenerator":
                        full_response += chunk_content
                        payload = json.dumps({
                            "type": "content",
                            "text": chunk_content
                        }, ensure_ascii=False)
                        yield f"data: {payload}\n\n".encode("utf-8")

                # 2. Retriever Start
                elif kind == "on_retriever_start":
                     yield f"data: {json.dumps({'type': 'status', 'text': 'Searching knowledge base...'}, ensure_ascii=False)}\n\n".encode("utf-8")

                # 3. Retriever End
                elif kind == "on_retriever_end":
                    docs = event["data"].get("output", [])
                    if docs:
                        found_sources = list(set(d.metadata.get("source", "Unknown") for d in docs))
                        msg = f"Found {len(docs)} relevant docs"
                        yield f"data: {json.dumps({'type': 'status', 'text': msg}, ensure_ascii=False)}\n\n".encode("utf-8")

            # 4. Send Sources
            if found_sources:
                sources_payload = json.dumps({
                    "type": "sources",
                    "data": found_sources
                }, ensure_ascii=False)
                yield f"data: {sources_payload}\n\n".encode("utf-8")

            # 5. Save History
            await session_store.add_message(session_id, "human", request.message)
            await session_store.add_message(session_id, "ai", full_response)

        except Exception as e:
            logger.error(f"Stream Error: {e}")
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

# ----------------------------------------------------------------------
# Knowledge Upload Logic
# ----------------------------------------------------------------------

async def _background_pipeline(
    local_file_path: str, 
    user_id: str, 
    user_name: str, 
    original_filename: str
):
    """
    Pipeline: OCR -> Inject Metadata -> Indexing -> Cleanup
    """
    generated_md_path = None
    
    try:
        logger.info(f"🚀 [Task Start] User:{user_name}({user_id}) File:{original_filename}")

        # 1. Call OCR Service
        markdown_content, generated_md_path = await ocr_service.file_to_markdown(local_file_path)
        
        # 2. Inject Context (Metadata Header)
        # Fix: Generate current timestamp
        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        context_header = (
            f"> **🔐 文档归属信息 (Metadata)**\n"
            f"> - **用户姓名**: {user_name}\n"
            f"> - **用户ID**: {user_id}\n"
            f"> - **原始文件名**: {original_filename}\n"
            f"> - **上传时间**: {upload_time}\n"
            f"\n---\n\n"
        )
        
        final_markdown = context_header + markdown_content

        # 3. Prepare Metadata for DB
        metadata = {
            "source": original_filename,
            "user_id_card": user_id,
            "user_name": user_name,
            "type": "ocr_document"
        }

        logger.info(f"💾 [Indexing] Saving to vector store (Length: {len(final_markdown)})...")

        # 4. Indexing
        await vector_store.index_markdown_content(final_markdown, metadata)

        logger.success(f"✅ [Done] File processed: {original_filename}")

    except Exception as e:
        logger.error(f"❌ [Failed] Background task error: {e}")
    
    finally:
        # 5. Cleanup
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        
        if generated_md_path and os.path.exists(generated_md_path):
            os.remove(generated_md_path)


@router.post("/knowledge/upload", tags=["Knowledge"])
async def upload_knowledge_file(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Upload multiple files"), 
    user_id: str = Form(..., description="User ID Card"),
    user_name: str = Form(..., description="User Name"),
):
    """
    Batch Upload -> OCR Pipeline -> RAG Indexing
    """
    uploaded_details = []
    failed_files = []
    
    # Allowed formats
    allowed_exts = [".pdf", ".jpg", ".png", ".jpeg"]

    for file in files:
        original_name = file.filename
        ext = os.path.splitext(original_name)[1].lower()

        # 1. Format Check
        if ext not in allowed_exts:
            logger.warning(f"Skipping unsupported file: {original_name}")
            failed_files.append({"filename": original_name, "reason": "Unsupported format"})
            continue

        # 2. Generate Unique Filename
        unique_filename = f"{user_id}_{uuid.uuid4().hex[:8]}{ext}"
        abs_file_path = os.path.join(SHARED_INPUT_DIR, unique_filename)

        # 3. Save to Shared Directory
        try:
            with open(abs_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"Save failed for {original_name}: {e}")
            failed_files.append({"filename": original_name, "reason": "Save failed"})
            continue

        # 4. Queue Background Task
        background_tasks.add_task(
            _background_pipeline, 
            abs_file_path,   
            user_id,         
            user_name,       
            original_name    
        )

        uploaded_details.append(original_name)

    return {
        "code": 200,
        "message": f"Received {len(uploaded_details)}, Failed {len(failed_files)}",
        "success_files": uploaded_details,
        "failed_files": failed_files,
        "task_info": {
            "user_name": user_name,
            "user_id": user_id
        }
    }

# ----------------------------------------------------------------------
# System & Monitoring
# ----------------------------------------------------------------------

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
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.delete("/sessions/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    await session_store.delete_session(session_id)
    return {"message": "Session deleted"}

@router.get("/", tags=["Monitor"])
async def root():
    return {
        "app": settings.APP_NAME,
        "status": "running",
        "docs": "/docs"
    }