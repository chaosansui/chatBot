import os
from typing import Optional, List, Dict, Any
from loguru import logger

from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader

# 引入底层连接管理
from pymilvus import connections, utility

from core.config import settings

class MilvusVectorStore:
    def __init__(self):
        # 1. Embedding 模型初始化
        self.embeddings = OpenAIEmbeddings(
            openai_api_base=settings.EMBEDDING_API_URL.rsplit('/', 1)[0], 
            model=settings.EMBEDDING_MODEL_NAME,
            openai_api_key="sk-not-needed-for-local", 
            check_embedding_ctx_length=False,
        )

        # 集合名称
        self.collection_name = f"{settings.MILVUS_COLLECTION_NAME}_v1"
        self.alias = "default" 
        self._store: Optional[Milvus] = None

    async def connect_milvus(self):
        """
        建立最基础的明文连接 (No TLS, No Auth)
        """
        target_host = settings.MILVUS_HOST
        target_port = settings.MILVUS_PORT
        
        logger.info(f"🔌 正在连接 Milvus ({target_host}:{target_port})...")
        
        try:
            # 1. 强制断开旧连接 (避免残留的配置干扰)
            if connections.has_connection(self.alias):
                connections.disconnect(self.alias)

            # 2. 建立纯净的明文连接
            # secure=False: 禁用 TLS/SSL 握手
            # 也不传递 user/password，强制匿名访问
            connections.connect(
                alias=self.alias,
                host=target_host, 
                port=target_port,
                secure=False 
            )
            
            logger.success(f"✅ Milvus 连接建立成功")

            # 3. 简单检查 (不加载也可以，但检查一下更稳妥)
            if utility.has_collection(self.collection_name, using=self.alias):
                logger.info(f"📚 集合 '{self.collection_name}' 存在")
            else:
                logger.info(f"ℹ️ 集合 '{self.collection_name}' 尚未创建")

        except Exception as e:
            # 遇到错误只打印，不中断程序启动
            # 这种策略允许在网络瞬断时，后续请求仍有机会重试
            logger.warning(f"⚠️ 连接警告: {e}")

    @property
    def vector_store(self) -> Milvus:
        """
        获取 VectorStore 实例
        """
        if self._store is None:
            # 关键：connection_args=None
            # 这告诉 LangChain："不要自己去握手，直接用我上面建立好的 'default' 全局连接"
            self._store = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                connection_args=None, 
                auto_id=True,
                drop_old=False
            )
        return self._store

    @property
    def collection(self):
        try:
            _ = self.vector_store
            return self._store.col
        except:
            return None

    async def index_documents(self, file_paths: List[str], user_name: str, user_id_card: str):
        """
        索引文档，并绑定用户信息
        """
        if not file_paths: return
        await self.connect_milvus()

        all_documents: List[Document] = []
        logger.info(f"📄 正在为用户 [{user_name} - {user_id_card}] 加载文档...")

        for path in file_paths:
            try:
                # ... (加载器的逻辑保持不变) ...
                if loader:
                    docs = loader.load()
                    for doc in docs:
                        # ⭐️ 核心修改在这里：把用户信息打入 Metadata
                        doc.metadata["source"] = os.path.basename(path)
                        doc.metadata["user_name"] = user_name
                        doc.metadata["user_id_card"] = user_id_card
                    
                    all_documents.extend(docs)
            except Exception as e:
                logger.error(f"加载失败 {path}: {e}")

        if not all_documents: return

        logger.info("✂️ 切分文档...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        split_docs = text_splitter.split_documents(all_documents)
        
        logger.info(f"💾 写入数据...")
        self.vector_store.add_documents(split_docs)
        logger.success(f"🎉 用户 {user_name} 的文档索引完成！")

    # 2. 修改 get_retriever，完善过滤逻辑
    def get_retriever(self, user_id_card: Optional[str] = None) -> VectorStoreRetriever:
        """
        获取检索器
        user_id_card: 如果提供，则强制过滤该用户的文档
        """
        search_kwargs = {"k": settings.RAG_TOP_K}
    
        if user_id_card:
            # 语法解释：筛选 metadata 中 user_id_card 等于 传入值 的数据
            search_kwargs["expr"] = f"user_id_card == '{user_id_card}'"
            logger.info(f"🔍 启用权限过滤: 仅检索 ID={user_id_card} 的数据")

        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )
    
vector_store = MilvusVectorStore()