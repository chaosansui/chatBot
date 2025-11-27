import os
import asyncio
from typing import Optional, List, Dict, Any
from loguru import logger

# LangChain 核心组件
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from pymilvus import connections, utility, Collection

from core.config import settings

class MilvusVectorStore:
    def __init__(self):
        # 1. Embedding 初始化
        base_url = settings.EMBEDDING_API_URL
        if base_url.endswith("/embeddings"):
            base_url = base_url.replace("/embeddings", "")
            
        self.embeddings = OpenAIEmbeddings(
            openai_api_base=base_url,
            model=settings.EMBEDDING_MODEL_NAME,
            openai_api_key="EMPTY", 
            check_embedding_ctx_length=False,
        )

        self.collection_name = f"{settings.MILVUS_COLLECTION_NAME}_v1"
        self.alias = "default" 
        self._store: Optional[Milvus] = None
        self._lock = asyncio.Lock()

    async def connect_milvus(self):
        """建立连接 (单例模式)"""
        async with self._lock:
            if connections.has_connection(self.alias):
                return

            logger.info(f"🔌 连接 Milvus ({settings.MILVUS_HOST}:{settings.MILVUS_PORT})...")
            try:
                connections.connect(
                    alias=self.alias,
                    host=settings.MILVUS_HOST, 
                    port=settings.MILVUS_PORT,
                    secure=settings.MILVUS_SECURE
                )
                logger.success(f"✅ Milvus 连接成功")
            except Exception as e:
                logger.error(f"❌ Milvus 连接失败: {e}")
                raise e

    @property
    def vector_store(self) -> Milvus:
        """获取 LangChain VectorStore 实例"""
        if self._store is None:
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
            _ = self.vector_store
            return Collection(self.collection_name, using=self.alias)
        except Exception:
            return None

    # =========================================================
    # 🔥 核心方法：Markdown 智能索引 (带上下文注入)
    # =========================================================
    async def index_markdown_content(self, markdown_text: str, metadata: dict):
        """
        将 OCR 生成的 Markdown 存入 Milvus。
        关键特性：在每一个切分块(Chunk)的头部，强制注入归属信息。
        """
        await self.connect_milvus()

        # 1. 构造上下文头部 (Context Header)
        # 这一步是为了让每一个切片都“自带名片”
        # 格式示例: "> 文件归属: 张三 (ID:1001) \n> 来源: 工资单.pdf"
        user_name = metadata.get('user_name', '未知用户')
        user_id = metadata.get('user_id_card', '无ID')
        source = metadata.get('source', '未知文件')
        
        context_header = (
            f"> **📄 文件归属**: {user_name} ({user_id})\n"
            f"> **📂 来源文件**: {source}\n"
            f"\n---\n"
        )

        final_docs = []

        # 2. 策略分流
        # 策略 A: 短文档 (证件/单据，< 2000字符) -> 不切分，整块入库
        if len(markdown_text) < 2000:
            logger.info(f"📄 [索引] 短文档 ({len(markdown_text)} chars)，保持完整上下文。")
            # 确保头部存在 (虽然 endpoints 里加过，这里做双重保险)
            if "文件归属" not in markdown_text:
                markdown_text = context_header + markdown_text
            
            final_docs = [Document(page_content=markdown_text, metadata=metadata)]
        
        # 策略 B: 长文档 (手册/合同) -> 结构化切分 + 头部注入
        else:
            logger.info("✂️ [索引] 长文档，执行上下文注入切分...")
            
            # 第一层：Markdown 逻辑切分
            headers_to_split_on = [("#", "Title"), ("##", "Section"), ("###", "Subsection")]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_docs = markdown_splitter.split_text(markdown_text)

            # 第二层：字符级物理切分
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, 
                chunk_overlap=200,
                separators=["\n\n", "\n", "。", "！", "，"]
            )
            
            for md_doc in md_docs:
                splits = text_splitter.split_documents([md_doc])
                for split in splits:
                    # 继承元数据
                    split.metadata.update(metadata)
                    
                    # 🔥 注入动作：如果分片没有头，就给它安一个头
                    if "文件归属" not in split.page_content:
                        split.page_content = context_header + split.page_content
                    
                    final_docs.append(split)

        logger.info(f"💾 [Milvus] 写入 {len(final_docs)} 个向量分片...")
        
        try:
            self.vector_store.add_documents(final_docs)
            logger.success(f"🎉 索引完成！")
        except Exception as e:
            logger.error(f"❌ Milvus 写入失败: {e}")
            raise

    def get_retriever(self, k: int = 15) -> VectorStoreRetriever:
        search_kwargs = {"k": k}
        
        logger.info(f"🔍 [Retriever] 全局语义检索模式 (Smart Search)")

        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )
    
vector_store = MilvusVectorStore()