import asyncio
from typing import List, Optional
from loguru import logger
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from core.config import settings
from core.prompts import get_rewrite_prompt, get_qa_prompt 
from storage.vector_store import vector_store
from services.llm_service import llm_service   
from operator import itemgetter  

class RAGService:
    def __init__(self):
        self.vector_store = vector_store
        self.llm = llm_service.langchain_llm
        self.collection = None

    async def initialize(self):
        logger.info("⚙️ RAG Service 初始化...")
        try:
            await self.vector_store.connect_milvus()
            self.collection = self.vector_store.collection
            # 显式加载 Collection 以加速首次检索
            if self.collection:
                self.collection.load()
                logger.info(f"✅ Milvus Collection '{settings.MILVUS_COLLECTION_NAME}' 已加载")
        except Exception as e:
            logger.error(f"❌ Milvus 初始化失败: {e}")

    def get_rag_chain(self, user_id_card: str) -> Runnable:
        """构建 RAG 执行链"""
        
        # 1. 获取带过滤条件的检索器
        retriever = self.vector_store.get_retriever(user_id_card=user_id_card)

        # 2. 定义文档格式化函数（增强版）
        def format_docs(docs: List[Document]) -> str:
            if not docs:
                return "未找到相关背景信息。"
            
            formatted_docs = []
            for i, doc in enumerate(docs):
                # 尝试获取文件名或页码，如果没有则显示未知
                source_name = doc.metadata.get("filename") or doc.metadata.get("source") or "未知文件"
                # 移除多余换行，节省 Token
                clean_content = doc.page_content.replace('\n', ' ').strip()
                formatted_docs.append(f"<引用 id='{i+1}' source='{source_name}'>\n{clean_content}\n</引用>")
            
            return "\n\n".join(formatted_docs)

        # 3. 问题改写分支 (History Aware)
        # 关键优化：改写时使用低温度 (temperature=0.1)，保证语义不漂移
        rewrite_chain = (
            get_rewrite_prompt()
            | self.llm.bind(temperature=0.1) 
            | StrOutputParser()
        )

        # 路由逻辑：如果有 chat_history，则改写；否则直接用原始 question
        query_transform_branch = RunnableBranch(
            (
                lambda x: len(x.get("chat_history", [])) > 0,
                rewrite_chain
            ),
            itemgetter("question")
        )

        # 4. 检索链 (Retrieval Chain)
        # 先改写问题 -> 再检索 -> 再格式化文档
        retrieval_chain = (
            RunnablePassthrough.assign(query_rewritten=query_transform_branch) 
            | RunnablePassthrough.assign(
                docs=lambda x: retriever.invoke(x["query_rewritten"])
            )
            | RunnablePassthrough.assign(
                context=lambda x: format_docs(x["docs"])
            )
        )

        # 5. 生成链 (Generation Chain)
        rag_chain = (
            retrieval_chain
            | RunnablePassthrough.assign(
                answer=get_qa_prompt() 
                       | self.llm.with_config(run_name="AnswerGenerator") 
                       | StrOutputParser()
            )
        )
        
        return rag_chain

rag_service = RAGService()