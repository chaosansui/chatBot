import os
import asyncio
from typing import Optional, List, Dict, Any
from loguru import logger
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
)

# 引入底层连接管理
from pymilvus import connections, utility, Collection

from core.config import settings

class MilvusVectorStore:
    def __init__(self):
        # 1. Embedding 配置优化
        # 自动处理 URL 后缀，防止 config 配置出错
        base_url = settings.EMBEDDING_API_URL
        if base_url.endswith("/embeddings"):
            base_url = base_url.replace("/embeddings", "")
        elif base_url.endswith("/v1"):
            pass # 或者是 /v1，视具体模型服务而定
            
        self.embeddings = OpenAIEmbeddings(
            openai_api_base=base_url,
            model=settings.EMBEDDING_MODEL_NAME,
            openai_api_key="EMPTY", # 本地模型通常不需要 Key
            check_embedding_ctx_length=False,
        )

        self.collection_name = f"{settings.MILVUS_COLLECTION_NAME}_v1"
        self.alias = "default" 
        self._store: Optional[Milvus] = None
        # 增加一个连接锁，防止并发初始化时的竞争
        self._lock = asyncio.Lock()

    async def connect_milvus(self):
        """
        建立 Milvus 连接 (单例模式优化)
        """
        async with self._lock:
            if connections.has_connection(self.alias):
                # 如果已经连接，直接返回，不再断开重连
                return

            logger.info(f"🔌 正在连接 Milvus ({settings.MILVUS_HOST}:{settings.MILVUS_PORT})...")
            
            try:
                connections.connect(
                    alias=self.alias,
                    host=settings.MILVUS_HOST, 
                    port=settings.MILVUS_PORT,
                    secure=settings.MILVUS_SECURE
                )
                logger.success(f"✅ Milvus 连接成功")
                
                # 连接建立后，检查是否需要创建索引优化
                self._ensure_scalar_index()
                
            except Exception as e:
                logger.error(f"❌ Milvus 连接失败: {e}")
                raise e

    async def index_markdown_content(self, markdown_text: str, metadata: dict):
        """
        专门处理 OCR 转换后的 Markdown 文本
        """
        await self.connect_milvus()

        logger.info(f"✂️ 正在进行 Markdown 结构化切分...")

        # 1. 按标题层级切分 (保留章节结构)
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        # 这一步会把 markdown_text 切成多个 Document，每个都带有 Header metadata
        md_header_splits = markdown_splitter.split_text(markdown_text)

        # 2. 注入业务元数据 (User ID, Name, Source)
        for doc in md_header_splits:
            doc.metadata.update(metadata)

        # 3. 二次切分 (防止某个章节内容过长超过 Embedding 限制)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.RAG_CHUNK_SIZE, 
            chunk_overlap=settings.RAG_CHUNK_OVERLAP
        )
        final_docs = text_splitter.split_documents(md_header_splits)

        logger.info(f"💾 正在写入 Milvus ({len(final_docs)} 个分片)...")
        try:
            self.vector_store.add_documents(final_docs)
            logger.success(f"🎉 索引完成！文档来源: {metadata.get('source')}")
        except Exception as e:
            logger.error(f"❌ Milvus 写入失败: {e}")
            raise
    
    def _ensure_scalar_index(self):
        """
        (高级优化) 确保用于过滤的标量字段有索引
        注意：LangChain 创建的 Collection 默认 metadata 可能是 JSON 动态字段，
        或者 auto_id 模式。这里假设字段作为普通 Scalar 存在。
        """
        if utility.has_collection(self.collection_name, using=self.alias):
            try:
                col = Collection(self.collection_name, using=self.alias)
                # 检查 user_id_card 是否有索引，没有则建立
                # 注意：这取决于 LangChain 首次插入数据时是如何定义 Schema 的
                # 如果是 LangChain 默认行为，metadata 里的字段可能无法直接建索引
                # 这里仅作为后续手动优化 Schema 后的预留接口
                pass 
            except Exception as e:
                logger.warning(f"索引检查跳过: {e}")

    @property
    def vector_store(self) -> Milvus:
        """获取 LangChain VectorStore 实例 (懒加载)"""
        if self._store is None:
            # 确保连接存在（同步环境下可能需要预先 await connect_milvus）
            # 但由于 property 不能是 async，我们假设 initialize 已经调用过
            self._store = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                connection_args={"host": settings.MILVUS_HOST, "port": settings.MILVUS_PORT},
                auto_id=True,
                drop_old=False,
                primary_field="pk",
                text_field="text",
                vector_field="vector"
            )
        return self._store

    @property
    def collection(self):
        try:
            # 触发初始化
            _ = self.vector_store
            return Collection(self.collection_name, using=self.alias)
        except Exception:
            return None

    async def index_documents(self, file_paths: List[str], user_name: str, user_id_card: str):
        """索引文档：完善了 Loader 映射和错误处理"""
        if not file_paths: return
        
        # 确保连接
        await self.connect_milvus()

        all_documents: List[Document] = []
        logger.info(f"📄 正在处理用户 [{user_name}] 的 {len(file_paths)} 个文件...")

        # 定义支持的 Loader 映射
        LOADER_MAPPING = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".mmd": UnstructuredMarkdownLoader,
            ".docx": Docx2txtLoader,
        }

        for path in file_paths:
            ext = os.path.splitext(path)[1].lower()
            loader_cls = LOADER_MAPPING.get(ext)
            
            if not loader_cls:
                logger.warning(f"⚠️ 跳过不支持的文件格式: {path}")
                continue

            try:
                # 实例化 Loader
                loader = loader_cls(path)
                docs = loader.load()
                
                # 清洗和注入 Metadata
                for doc in docs:
                    doc.metadata["source"] = os.path.basename(path)
                    doc.metadata["user_name"] = user_name
                    # 关键：确保这个字段存在，以便后续 filter 使用
                    doc.metadata["user_id_card"] = user_id_card 
                
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"❌ 加载文件失败 {path}: {e}")

        if not all_documents:
            logger.warning("⚠️ 没有有效文档被加载")
            return

        # 优化切分策略
        logger.info(f"✂️ 正在切分 {len(all_documents)} 个文档...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "，", " ", ""] # 针对中文优化
        )
        split_docs = text_splitter.split_documents(all_documents)
        
        logger.info(f"💾 正在写入 Milvus ({len(split_docs)} 个分片)...")
        try:
            # 批量写入
            self.vector_store.add_documents(split_docs)
            logger.success(f"🎉 索引完成！用户: {user_name}, 向量数: {len(split_docs)}")
        except Exception as e:
            logger.error(f"❌ Milvus 写入失败: {e}")
            raise

    def get_retriever(self, user_id_card: Optional[str] = None, k: int = 4) -> VectorStoreRetriever:
        search_kwargs = {
            "k": k,
        }
        
        if user_id_card:
            search_kwargs["expr"] = f"user_id_card == '{user_id_card}'"
            
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )
    
vector_store = MilvusVectorStore()