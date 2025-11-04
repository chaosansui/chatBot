from typing import List
from loguru import logger
import os
from core.config import settings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections, utility



class RAGService:
    """RAG 服务：使用 Milvus 作为向量存储，BGE-M3 作为嵌入模型"""
    
    def __init__(self):
        # 1. 初始化嵌入模型 (Embedding Model) - 使用 BGE-M3
        self.embedding_model_name = "BAAI/bge-m3"
        self.collection_name = settings.MILVUS_COLLECTION_NAME # 例如: "rag_documents"
        
        try:
            # BGE-M3 是多语言模型，默认加载方式如下
            self.embeddings = HuggingFaceBgeEmbeddings(model_name=self.embedding_model_name)
            logger.info(f"✅ 嵌入模型加载成功: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"❌ 嵌入模型加载失败，请检查模型下载: {e}")
            self.embeddings = None
        
        # 2. 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.RAG_CHUNK_SIZE,     # 建议 BGE-M3 使用大尺寸，例如 800-1024
            chunk_overlap=settings.RAG_CHUNK_OVERLAP, # 例如: 50-100
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""] 
        )

        # 3. 连接 Milvus 数据库
        self.milvus_host = settings.MILVUS_HOST
        self.milvus_port = settings.MILVUS_PORT
        self.vector_store = None
        self._connect_milvus()


    def _connect_milvus(self):
        """尝试连接 Milvus 并初始化 Milvus 向量存储客户端"""
        if not self.embeddings:
            logger.error("嵌入模型未就绪，无法连接 Milvus。")
            return
            
        try:
            # 创建 Milvus 连接
            connections.connect(
                alias="default", 
                host=self.milvus_host, 
                port=self.milvus_port
            )
            
            # 检查连接是否成功
            if utility.has_connection("default"):
                logger.info(f"✅ 成功连接到 Milvus 服务：{self.milvus_host}:{self.milvus_port}")
                
                # 初始化 LangChain Milvus 客户端
                self.vector_store = Milvus(
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name,
                    connection_args={"host": self.milvus_host, "port": self.milvus_port},
                    auto_id=True, # 使用 Milvus 自动生成的 ID
                    drop_old=False # 启动时不删除旧的 Collection
                )
                logger.info(f"✅ Milvus 集合客户端初始化成功：{self.collection_name}")
            else:
                logger.error("❌ 无法建立 Milvus 连接，请检查服务状态。")
                
        except Exception as e:
            logger.error(f"❌ Milvus 连接或初始化失败: {e}")

    async def process_data(self, file_paths: List[str]):
        """处理一组文件：加载、切分、嵌入并存储到 Milvus。"""
        if not self.vector_store:
            logger.error("Milvus 向量存储未准备就绪，无法处理数据。")
            return
            
        all_documents = []
        
        # 1. 文档加载 (保持不变)
        # ... (加载逻辑)
        for path in file_paths:
            try:
                if path.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                elif path.endswith(".txt"):
                    loader = TextLoader(path, encoding='utf-8')
                else:
                    logger.warning(f"不支持的文件类型，跳过: {path}")
                    continue
                
                documents = loader.load()
                all_documents.extend(documents)
                logger.info(f"📚 成功加载文档: {path}, 页数/块数: {len(documents)}")
            except Exception as e:
                logger.error(f"加载文件 {path} 失败: {e}")

        if not all_documents:
            logger.warning("没有可处理的文档。")
            return

        # 2. 文本切分 (Chunking)
        texts = self.text_splitter.split_documents(all_documents)
        logger.info(f"✂️ 文档切分完成，总计 {len(texts)} 个文本块。")

        # 3. 嵌入并存储到 Milvus
        try:
            # LangChain Milvus 客户端的 add_documents 会自动处理嵌入和插入
            self.vector_store.add_documents(texts)
            logger.info(f"⚡️ {len(texts)} 个文本块已成功嵌入并存储到 Milvus 集合 {self.collection_name}。")
        except Exception as e:
            logger.error(f"❌ 嵌入并存储到 Milvus 失败: {e}")


    def get_retriever(self) -> BaseRetriever:
        """对外提供 Milvus 检索器实例。"""
        if not self.vector_store:
            logger.error("Milvus 向量存储未准备就绪，检索器返回空。")
            from langchain_core.retrievers import create_base_retriever
            return create_base_retriever(lambda x: []) 
        
        # 使用 Milvus 的默认检索器
        return self.vector_store.as_retriever(
            search_type="similarity", # Milvus 检索类型
            search_kwargs={"k": settings.RAG_TOP_K}
        )

# 全局 RAG 服务实例
rag_service = RAGService()