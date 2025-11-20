# services/rag_service.py
import asyncio
from typing import List, Optional
from loguru import logger

from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever 

from core.config import settings
from storage.vector_store import vector_store
from services.llm_service import llm_service     


class RAGService:
    def __init__(self):
        self.vector_store = vector_store
        self.llm = llm_service.langchain_llm
        self.collection: Optional[object] = None

    async def connect_milvus(self):
        await self.vector_store.connect_milvus()
        self.collection = self.vector_store.collection
    
    async def process_data(self, file_paths: List[str]):
        """
        处理文档索引流程：确保连接 Milvus，并调用 vector_store 执行加载、切分、嵌入和存储。
        """
        if not self.collection:
            await self.connect_milvus()

        logger.info("开始文档加载、切分、嵌入和 Milvus 存储...")
        
        try:
            # 索引文档的逻辑在 vector_store 中
            await self.vector_store.index_documents(file_paths=file_paths)
            await self.connect_milvus() 
            logger.info("文档索引流程已委托给 vector_store 完成。")
        except AttributeError:
            logger.error(f"❌ vector_store 对象缺少 'index_documents' 方法，无法执行索引。")
            raise

    # ⭐️ 关键修复点：get_retriever 方法必须存在 ⭐️
    def get_retriever(self, user_id_card: Optional[str] = None) -> VectorStoreRetriever:
        """
        获取 Milvus 检索器，委托给 vector_store 完成。
        """
        # 实际调用 vector_store 实例的 get_retriever 方法
        return self.vector_store.get_retriever(user_id_card=user_id_card)

    def get_rag_chain(self, user_id_card: str) -> Runnable:
        
        # RAGService 内部调用自身的 get_retriever 方法
        retriever = self.get_retriever(user_id_card=user_id_card)
        
        system_template = (
            "你是一个智能客服机器人，擅长处理中英粤三语提问。\n"
            "请基于提供的背景信息 (context) 简洁、准确地回答问题。\n"
            "如果背景信息不足以回答用户问题，请明确告知用户。\n"
            "背景信息:\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_template),
                MessagesPlaceholder(variable_name="chat_history"), # 历史会话
                HumanMessage(content="{question}"),
            ]
        )
        
        # 这是一个简单的格式化函数，将 Document 对象转换为字符串
        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)
            
        # 链式调用
        rag_chain = (
            {
                "context": retriever | format_docs, # 检索并格式化文档
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain

rag_service = RAGService()