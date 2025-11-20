import asyncio
from typing import List, Optional, Dict, Any
from operator import itemgetter
from loguru import logger

# LangChain 核心组件
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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

    async def initialize(self):
        """初始化 RAG 服务"""
        logger.info("⚙️ 正在初始化 RAG Service...")
        await self.connect_milvus()
        
        if self.collection:
            try:
                self.collection.load()
                logger.info(f"✅ Milvus Collection '{settings.COLLECTION_NAME}' 已加载到内存")
            except Exception as e:
                logger.warning(f"⚠️ Milvus Collection 加载失败: {e}")
        
        try:
            _ = self.vector_store.embeddings
            logger.info("✅ Embedding 模型已就绪")
        except Exception as e:
            logger.error(f"❌ Embedding 模型加载失败: {e}")

    async def connect_milvus(self):
        await self.vector_store.connect_milvus()
        self.collection = self.vector_store.collection
    
    async def process_data(self, file_paths: List[str]):
        if not self.collection:
            await self.connect_milvus()

        logger.info(f"📂 开始处理 {len(file_paths)} 个文件...")
        try:
            await self.vector_store.index_documents(file_paths=file_paths)
            if self.collection:
                self.collection.load()
            logger.success("✅ 文档索引完成并已生效。")
        except Exception as e:
            logger.error(f"❌ 文档处理失败: {e}")
            raise

    def get_retriever(self, user_id_card: Optional[str] = None) -> VectorStoreRetriever:
        return self.vector_store.get_retriever(user_id_card=user_id_card)

    def get_rag_chain(self, user_id_card: str) -> Runnable:
        
        retriever = self.get_retriever(user_id_card=user_id_card)

        # --- 步骤 1: 定义 "问题改写" 分支逻辑 ---
        
        # A. 改写问题的 Prompt
        contextualize_q_system_prompt = (
            "给定一段聊天记录和用户最新的问题（该问题可能引用了上下文），"
            "请将该问题改写为一个独立的、无需上下文即可理解的完整问题。"
            "不要回答问题，只需返回改写后的问题；如果无需改写，原样返回。"
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )
        
        # B. 改写链 (增加 run_name="QuestionRewriter")
        rewrite_chain = (
            contextualize_q_prompt 
            | self.llm.with_config(run_name="QuestionRewriter") 
            | StrOutputParser()
        )

        # C. 分支路由：无历史直接返回问题，有历史则调用改写链
        query_transform_branch = RunnableBranch(
            (
                lambda x: not x.get("chat_history"),
                RunnableLambda(lambda x: x["question"])
            ),
            rewrite_chain
        )

        # D. 组合历史感知检索器
        history_aware_retriever = query_transform_branch | retriever

        # --- 步骤 2: 定义 "回答生成" 逻辑 ---
        
        qa_system_template = (
            "你是一个专业的智能助手。\n"
            "请基于以下检索到的背景信息 (context) 回答问题。\n"
            "如果背景信息里没有答案，请直接说“由于缺乏相关信息，我无法回答这个问题”，不要编造。\n"
            "回答要条理清晰，使用 Markdown 格式。\n\n"
            "背景信息:\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_template),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(f"[资料片段] {doc.page_content}" for doc in docs)

        # --- 步骤 3: 组装检索链 (你之前可能漏掉了这个变量的定义) ---
        
        retrieval_chain = RunnablePassthrough.assign(
            docs=history_aware_retriever,
        ).assign(
            context=lambda x: format_docs(x["docs"]),
            sources=lambda x: x["docs"]
        )

        # --- 步骤 4: 组装最终 RAG 链 ---

        rag_chain = (
            retrieval_chain
            | RunnablePassthrough.assign(
                # 增加 run_name="AnswerGenerator" 以便 API 层过滤
                answer=qa_prompt 
                       | self.llm.with_config(run_name="AnswerGenerator") 
                       | StrOutputParser()
            )
        )
        
        return rag_chain

rag_service = RAGService()