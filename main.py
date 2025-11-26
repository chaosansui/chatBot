import sys
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# 导入模块
from api.endpoints import router as api_router
from core.config import settings
from services.llm_service import llm_service
from services.rag_service import rag_service 
from storage.session_store import session_store

# --- 1. 日志拦截器配置 (保持不变，这是很好的实践) ---
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
    
    logger.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "level": settings.LOG_LEVEL,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            }
        ]
    )

# --- 2. 生命周期管理 (核心优化点) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动与关闭流程"""
    
    setup_logging()
    logger.info(f"🚀 正在启动 {settings.APP_NAME} v{settings.VERSION} ...")
    
    try:
        await session_store.connect()
        logger.success(f"✅ Redis 连接成功 ({settings.REDIS_HOST}:{settings.REDIS_PORT})")
    except Exception as e:
        logger.error(f"❌ Redis 连接失败: {e}")
   
    try:
        await rag_service.initialize()
        logger.success("✅ RAG 服务已就绪")
    except Exception as e:
        logger.error(f"❌ RAG 服务初始化失败: {e}")


    try:
        is_ready = await llm_service.health_check()
        if is_ready:
            logger.success(f"✅ LLM 服务连接正常: {settings.LOCAL_MODEL_URL}")
        else:
            logger.warning(f"⚠️ LLM 服务未响应")
    except Exception as e:
        logger.error(f"❌ LLM 服务检查异常: {e}") 

    # 输出访问地址
    docs_url = f"http://{settings.HOST}:{settings.PORT}{settings.API_PREFIX}/docs"
    logger.info(f"📚 API 文档地址: {docs_url}")
    
    yield # 服务运行中...

    # [D] 优雅关闭
    logger.info("🛑 正在停止服务...")
    await session_store.close()
    logger.success("👋 再见！")

# --- 3. FastAPI 应用定义 ---
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Funki AI 智能助手 (Streaming API)",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载路由
app.include_router(api_router, prefix=settings.API_PREFIX)

# --- 4. 系统级健康检查 (K8s/LB 用) ---
@app.get("/health", tags=["System"])
async def health_check():
    """
    基础设施健康检查
    """
    # 1. 检查 Redis
    redis_ok = session_store.client is not None
    
    # 2. 检查 Milvus Collection 是否加载
    milvus_ok = False
    if rag_service.collection:
        milvus_ok = True # 简单检查对象存在即可，不必每次都 ping

    # 3. 检查 LLM (可选：因为 LLM 检查耗时，高频健康检查可以跳过或缓存状态)
    llm_ok = llm_service.is_ready

    status = "healthy" if (redis_ok and milvus_ok) else "degraded"

    return {
        "status": status,
        "components": {
            "redis": "connected" if redis_ok else "disconnected",
            "milvus": "ready" if milvus_ok else "not_ready",
            "llm": "ready" if llm_ok else "not_ready"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1,
    )