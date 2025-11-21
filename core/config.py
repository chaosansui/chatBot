import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用配置 - 支持环境变量覆盖"""
    
    # --- 1. 应用基础配置 ---
    APP_NAME: str = "Smart Assistant API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8006
    API_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    
    # --- 2. 本地模型配置 (LLM) ---
    LOCAL_MODEL_HOST: str = "localhost"
    LOCAL_MODEL_PORT: int = 8002
    LOCAL_MODEL_API_PATH: str = "/v1/chat/completions"
    
    @property
    def LOCAL_MODEL_URL(self) -> str:
        return f"http://{self.LOCAL_MODEL_HOST}:{self.LOCAL_MODEL_PORT}{self.LOCAL_MODEL_API_PATH}"
    
    # 模型调用参数
    MODEL_TIMEOUT: int = 60
    MODEL_MAX_TOKENS: int = 2048
    MODEL_TEMPERATURE: float = 0.7
    
    # --- 3. Milvus 配置 ---
    MILVUS_HOST: str = "127.0.0.1"
    MILVUS_PORT: str = "19530"
    MILVUS_COLLECTION_NAME: str = "funki_case"
    MILVUS_USER: str = "" 
    MILVUS_PASSWORD: str = "" 
    MILVUS_SECURE: bool = False 

    @property
    def MILVUS_ADDRESS(self) -> str:
        return f"{self.MILVUS_HOST}:{self.MILVUS_PORT}"
    
    # --- 4. RAG (检索增强生成) ---
    
    # 本地 Embedding 模型 (如 BGE-M3 via vLLM/TEI)
    EMBEDDING_API_HOST: str = "localhost" 
    EMBEDDING_API_PORT: int = 10010       
    EMBEDDING_MODEL_NAME: str = "bge"
    
    @property
    def EMBEDDING_API_URL(self) -> str:
        return f"http://{self.EMBEDDING_API_HOST}:{self.EMBEDDING_API_PORT}/v1/embeddings"
    
    # 切分策略
    RAG_CHUNK_SIZE: int = 800 
    RAG_CHUNK_OVERLAP: int = 100
    RAG_TOP_K: int = 4
    
    ENABLE_RAG: bool = True
    
    # --- 5. Redis Session Store ---
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = "" 
    
    SESSION_TTL: int = 24 * 3600
    HISTORY_LIMIT: int = 10
    
    @property
    def REDIS_URL(self) -> str:
        """拼接 Redis 连接字符串"""
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    # --- 6. CORS 配置 ---
    CORS_ORIGINS: str = "*"
    
    @property
    def cors_origins_list(self) -> List[str]:
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    

# 全局配置实例
settings = Settings()
