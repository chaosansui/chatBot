import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用配置 - 支持环境变量覆盖"""
    
    # 应用基础配置
    APP_NAME: str = "Smart Assistant API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8006
    
    # API配置
    API_PREFIX: str = "/api/v1"
    
    # 本地模型配置
    LOCAL_MODEL_HOST: str = "localhost"
    LOCAL_MODEL_PORT: int = 8002
    
    # Milvus 配置
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: str = "19530"
    MILVUS_COLLECTION_NAME: str = "qwen_rag_docs"

    # RAG 参数
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    RAG_CHUNK_SIZE: int = 800  
    RAG_CHUNK_OVERLAP: int = 80
    RAG_TOP_K: int = 4
    
    
    @property
    def LOCAL_MODEL_URL(self) -> str:
        return f"http://{self.LOCAL_MODEL_HOST}:{self.LOCAL_MODEL_PORT}/v1/chat/completions"
    
    # 模型调用配置
    MODEL_TIMEOUT: int = 30
    MODEL_MAX_TOKENS: int = 1024
    MODEL_TEMPERATURE: float = 0.7
    
    # 基础RAG配置
    ENABLE_RAG: bool = False
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    
    # CORS配置
    CORS_ORIGINS: str = "*"
    
    @property
    def cors_origins_list(self):
        """将CORS_ORIGINS字符串转换为列表"""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    class Config:
        case_sensitive = False
        env_prefix = ""  # 环境变量不需要前缀

# 全局配置实例
settings = Settings()