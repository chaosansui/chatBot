from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.endpoints import router as api_router
from core.config import settings
from services.llm_service import llm_service
from services.rag_service import rag_service 
from storage.session_store import session_store
import sys
from loguru import logger
import logging

# 配置 Loguru 日志系统
try:
    logger.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "level": settings.LOG_LEVEL,
            }
        ]
    )
    # 将标准 logging 模块的输出重定向到 Loguru 的 stderr
    logging.basicConfig(handlers=[logging.StreamHandler(sys.stderr)], level=0)
    logger.success("Loguru 日志系统初始化完成，接管所有日志输出")
except Exception as e:
    print(f"Loguru 配置失败: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：服务初始化和关闭"""
    
    logger.success(f"🤖 {settings.APP_NAME} v{settings.VERSION} 正在启动...")
    
    # 1. 连接 Redis（会话存储）
    try:
        await session_store.connect()
        logger.success("✅ Redis 连接成功（会话历史已就绪）")
    except Exception as e:
        logger.error(f"❌ Redis 连接失败，会话历史将不可用: {e}")
        
    # 2. 预加载 BGE-M3
    try:
        logger.info("⏳ 预加载 BGE-M3 模型...")
        # ⭐️ 修复：访问正确的属性 ⭐️
        _ = rag_service.vector_store.embeddings
        logger.success("✅ BGE-M3 预加载完成，首次请求将秒响应！")
    except Exception as e:
        logger.error(f"❌ BGE-M3 加载失败: {e}")

    # 3. 连接 Milvus 并预热集合
    try:
        logger.info("⏳ 连接 Milvus 向量数据库...")
        await rag_service.connect_milvus()
        
        # 使用 vector_store 对象上的 collection 属性
        collection = rag_service.vector_store.collection
        
        if collection:
            collection.load()  
            logger.success("✅ Milvus 连接成功，集合已加载到内存，检索零延迟")
        else:
            logger.warning("⚠️ Milvus 连接成功，但 Collection 对象缺失，请检查配置")
    except Exception as e:
        logger.error(f"❌ Milvus 连接或加载失败，RAG 服务将不可用: {e}")

    # 4. 检查本地大模型
    try:
        await llm_service.health_check()
        logger.success(f"✅ 本地模型健康检查通过: {settings.LOCAL_MODEL_URL}")
    except Exception as e:
        logger.error(f"❌ 本地模型连接异常，LLM 服务将不可用: {e}") 

    logger.success("🎉 所有核心服务启动完成，准备接受请求！")
    logger.info(f"📚 API文档: http://{settings.HOST}:{settings.PORT}{settings.API_PREFIX}/docs")

    yield

    # 关闭时清理
    await llm_service.close()
    await session_store.close()
    logger.info("👋 应用已安全关闭")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="聊天机器人",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(api_router, prefix=settings.API_PREFIX)

# 健康检查接口
@app.get("/health")
async def health_check():
    llm_status = await llm_service.health_check(cache_only=True)
    milvus_collection = rag_service.vector_store.collection if rag_service.vector_store else None
    
    return {
        "status": "healthy",
        "llm_model": "ready" if llm_status else "error",
        "milvus": "connected" if milvus_collection else "disconnected",
        "bge_m3": "loaded",
        "redis": "connected" if session_store.client else "disconnected"
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else 2
    )