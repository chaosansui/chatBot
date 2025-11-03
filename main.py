from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.endpoints import router as api_router
from core.config import settings
from services.llm_service import llm_service
import logging

# 配置日志
logging.basicConfig(level=settings.LOG_LEVEL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print(f"🚀 {settings.APP_NAME} v{settings.VERSION} 启动中...")
    
    # 检查模型服务
    await llm_service.health_check()
    
    yield  # 应用运行期间
    
    # 关闭时
    await llm_service.close()
    print("👋 应用关闭完成")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router, prefix=settings.API_PREFIX)

@app.on_event("startup")  # 兼容旧版本FastAPI
async def startup_event():
    """应用启动事件（兼容性）"""
    print(f"📍 服务地址: http://{settings.HOST}:{settings.PORT}")
    print(f"📚 API文档: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"🔧 调试模式: {settings.DEBUG}")
    print(f"🤖 模型地址: {settings.LOCAL_MODEL_URL}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )