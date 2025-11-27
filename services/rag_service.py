import asyncio
from typing import List, Optional
from operator import itemgetter
from loguru import logger

from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# 项目内部依赖
from core.config import settings
from core.prompts import get_rewrite_prompt, get_qa_prompt
from storage.vector_store import vector_store
from services.llm_service import llm_service
from services.rerank_service import rerank_service 

class RAGService:
    def __init__(self):
        self.vector_store = vector_store
        self.llm = llm_service.langchain_llm
        self.collection = None

    async def initialize(self):
        """初始化"""
        logger.info("⚙️ RAG Service 初始化...")
        try:
            # 1. 连接 Milvus
            await self.vector_store.connect_milvus()
            
            # 2. 预加载 Rerank 模型 (修复了之前的报错)
            logger.info("🔥 正在预热 Rerank 模型...")
            _ = rerank_service.model 
            
            # 3. 加载 Collection
            if self.vector_store.collection:
                self.vector_store.collection.load()
                
            logger.success("✅ RAG 服务初始化完成")
        except Exception as e:
            logger.error(f"❌ RAG 服务初始化失败: {e}")

    def get_rag_chain(self) -> Runnable:
        
        base_retriever = self.vector_store.get_retriever(
            k=getattr(settings, "RAG_SEARCH_K", 15)
        )
        
        def rerank_step(inputs):
            query = inputs["query"]
            docs = inputs["docs"]
            return rerank_service.rerank(query, docs)
        def format_docs(docs: List[Document]) -> str:
            if not docs: 
                return "未找到相关背景信息。"
            
            formatted_docs = []
            for i, doc in enumerate(docs):
                source_name = doc.metadata.get("source") or "未知文件"
                score_info = ""
                if 'relevance_score' in doc.metadata:
                    score_info = f" (相关度: {doc.metadata['relevance_score']:.4f})"
                
                # 清洗换行符，保持排版整洁
                clean_content = doc.page_content.replace('\n', ' ').strip()
                formatted_docs.append(f"<引用 id='{i+1}' source='{source_name}'>{score_info}\n{clean_content}\n</引用>")
            
            return "\n\n".join(formatted_docs)

        # --- 3. 构建分支与链 ---

        # 问题改写分支
        rewrite_chain = (
            get_rewrite_prompt()
            | self.llm
            | StrOutputParser()
        )
        
        query_transform_branch = RunnableBranch(
            (lambda x: len(x.get("chat_history", [])) > 0, rewrite_chain),
            itemgetter("question")
        )

        # 组装检索链
        retrieval_chain = (
            # 1. 获取(改写后的)问题
            RunnablePassthrough.assign(query_rewritten=query_transform_branch)
            
            # 2. Milvus 初排 (获取 15 条)
            | RunnablePassthrough.assign(
                raw_docs=lambda x: base_retriever.invoke(x["query_rewritten"])
            )
            
            # 3. Rerank 精排 (筛选 Top 5)
            | RunnablePassthrough.assign(
                docs=lambda x: rerank_step({
                    "query": x["query_rewritten"], 
                    "docs": x["raw_docs"]
                })
            )
            
            # 4. 格式化文本 (现在 format_docs 已经定义了，不会报错了)
            | RunnablePassthrough.assign(
                context=lambda x: format_docs(x["docs"])
            )
        )

        # 最终 RAG 链
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