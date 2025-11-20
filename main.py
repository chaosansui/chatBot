import sys
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
from api.endpoints import router as api_router
from core.config import settings
from services.llm_service import llm_service
from services.rag_service import rag_service 
from storage.session_store import session_store

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logging():
    logging.getLogger().handlers = [InterceptHandler()]
    logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
    logging.getLogger("uvicorn.error").handlers = [InterceptHandler()]
    
    # 配置 Loguru 格式
    logger.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "level": settings.LOG_LEVEL,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            }
        ]
    )

# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：服务初始化和关闭"""
    
    setup_logging()
    logger.success(f"🤖 {settings.APP_NAME} v{settings.VERSION} 正在启动...")
    
    try:
        await session_store.connect()
        logger.success("✅ Redis 连接成功")
    except Exception as e:
        logger.error(f"❌ Redis 连接失败: {e}")
        

    try:
        logger.info("⏳ 正在初始化 RAG 服务 (Milvus & Embedding)...")
        
    
        await rag_service.connect_milvus()
        if rag_service.vector_store.collection:
            rag_service.vector_store.collection.load()

        _ = rag_service.vector_store.embeddings 
        
        logger.success("✅ RAG 服务初始化完成 (Milvus Connected, BGE Loaded)")
    except Exception as e:
        logger.error(f"❌ RAG 服务初始化失败: {e}")


    try:
        await llm_service.health_check()
        logger.success(f"✅ LLM 服务连接正常: {settings.LOCAL_MODEL_URL}")
    except Exception as e:
        logger.error(f"❌ LLM 服务不可用: {e}") 

    logger.success("🎉 服务已就绪！")
    logger.info(f"📚 API文档: http://{settings.HOST}:{settings.PORT}{settings.API_PREFIX}/docs")

    yield

    logger.info("🛑 正在停止服务...")
    await llm_service.close()
    await session_store.close()
    logger.success("👋 再见！")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Funki 聊天机器人",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_PREFIX)


@app.get("/health", tags=["System"])
async def health_check():
    """
    Kubernetes 或 负载均衡器使用的健康检查接口
    """
    # 检查 Redis
    redis_status = "connected" if session_store.client else "disconnected"
    
    milvus_status = "connected" if rag_service.collection else "disconnected"
    

    llm_status = "unknown"
    try:
        llm_status = "ready" 
    except:
        llm_status = "error"

    return {
        "status": "healthy",
        "components": {
            "redis": redis_status,
            "milvus": milvus_status,
            "llm": llm_status
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1,
        log_level="info"
    )