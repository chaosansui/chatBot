# storage/vector_store.py
import os
from typing import Optional, List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_core.vectorstores import VectorStoreRetriever
from pymilvus import connections, utility, Collection
from pymilvus.exceptions import MilvusException
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from core.config import settings
from loguru import logger


class MilvusVectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_base=settings.EMBEDDING_API_URL.rsplit('/', 1)[0], 
            model=settings.EMBEDDING_MODEL_NAME,
            openai_api_key="sk-not-needed-for-local-vllm",
        )

        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        self.alias = "default_milvus"
        self.collection: Optional[Collection] = None

    async def connect_milvus(self):
        try:
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=self.port,
                user=settings.MILVUS_USER,
                password=settings.MILVUS_PASSWORD,
                secure=settings.MILVUS_SECURE
            )

            if not utility.has_collection(self.collection_name, using=self.alias):
                logger.warning(f"Milvus Collection '{self.collection_name}' 不存在，将在索引时尝试创建。")
                self.collection = None
                return
            
            self.collection = Collection(self.collection_name, using=self.alias)
            logger.success(f"✅ Milvus Collection '{self.collection_name}' 已连接。")


        except MilvusException as e:
            logger.error(f"Milvus 连接失败: {e}")
            raise ConnectionError(f"Milvus 连接失败: {e}")
        except Exception as e:
            logger.error(f"Milvus 初始化发生未知错误: {e}")
            raise ConnectionError(f"Milvus 初始化发生未知错误: {e}")

    async def index_documents(self, file_paths: List[str]):
        """
        加载文件 -> 切分文本 -> 嵌入 -> 存储到 Milvus Collection。
        """
        all_documents: List[Document] = []
        
        # 1. 加载阶段 (Loading)
        logger.info("📄 开始加载原始文档...")
        for path in file_paths:
            ext = os.path.splitext(path)[1].lower()
            loader = None
            
            if ext == ".pdf":
                loader = PyPDFLoader(path)
            elif ext == ".txt":
                loader = TextLoader(path)
            elif ext == ".docx":
                loader = Docx2txtLoader(path)
            elif ext == ".md": # ⭐️ 修复点：新增 .md 文件加载器 ⭐️
                # 注意：使用 Unstructured 加载器需要安装 unstructured 库
                loader = UnstructuredMarkdownLoader(path) 
            
            if loader:
                documents = loader.load()
                all_documents.extend(documents)
                logger.info(f"   - 加载 {path} 成功，共 {len(documents)} 文本块/页。")
            else:
                logger.warning(f"   - 警告: 暂不支持文件类型 {ext} ({path})，已跳过。")

        if not all_documents:
            logger.error("所有文件均未加载或内容为空。索引失败。")
            return

        # 2. 切分阶段 (Splitting)
        logger.info("✂️ 开始切分文档片段...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
        split_docs = text_splitter.split_documents(all_documents)
        logger.info(f"文档切分完成，共生成 {len(split_docs)} 个片段。")

        # 3. & 4. 嵌入和存储阶段 (Embedding & Storage)
        logger.info(f"💾 开始嵌入和存储到 Milvus Collection: {self.collection_name}...")
        
        Milvus.from_documents(
             documents=split_docs,
             embedding=self.embeddings,
             collection_name=self.collection_name,
             connection_args={"host": self.host, "port": self.port},
        )
        logger.success("🎉 所有文档片段已成功嵌入并存储到 Milvus Collection！")
        
        self.collection = Collection(self.collection_name, using=self.alias)
    
    def get_retriever(self, user_id_card: Optional[str] = None) -> VectorStoreRetriever:
        
        if not connections.has_connection(self.alias) or not self.collection:
            logger.warning("Milvus 连接未初始化或 Collection 不存在，检索器将无法工作。请检查启动日志。")

        vector_store = Milvus(
            embedding_function=self.embeddings,
            connection_args={"host": self.host, "port": self.port},
            collection_name=self.collection_name,
            auto_id=False,
        )
        
       
        search_kwargs = {"k": settings.RAG_TOP_K}
        
        if user_id_card:
            search_kwargs["expr"] = f"user_id_card == '{user_id_card}'"
            search_kwargs["filter"] = search_kwargs["expr"] 
            
            logger.info(f"为用户 {user_id_card[:4]}*** 启用 RAG 过滤表达式: {search_kwargs['expr']}")

        return vector_store.as_retriever(
            search_kwargs=search_kwargs
        )
    
vector_store = MilvusVectorStore()