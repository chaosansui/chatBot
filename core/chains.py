# core/chains.py (最终修复版 - 绕过 LCEL 流式聚合)

from typing import List, Dict, Any, Optional
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
# 确保导入 StrOutputParser (尽管我们在链中不再使用它，但 LangChain Core 可能需要)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage 
from langchain_core.documents import Document
from langchain_community.document_transformers import LongContextReorder
import asyncio 

from services.llm_service import llm_service
from services.rag_service import rag_service
from storage.session_store import session_store
from core.config import settings
from loguru import logger
from models.api_models import ChatMessage 

LLM = llm_service.langchain_llm
COMPRESSOR_LLM = llm_service.langchain_llm 


RAG_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """你是一个严谨、专业的知识助手，必须严格遵守以下规则：
1. 你只能使用【检索到的上下文】中的信息来回答问题。
2. 如果上下文无法回答，就直接说“我无法根据已有知识库回答这个问题”，禁止编造。
3. 回答要简洁、专业、结构化，使用中文。
4. 如果答案在多个片段中，请整合后给出完整答案。
5. 引用时使用自然语言，不要出现 [1][2] 这种格式。

【检索到的上下文】：
{context}
"""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def create_rag_retriever(user_id_card: str = None):
    """根据用户ID创建检索器。"""
    base_retriever = rag_service.get_retriever(user_id_card=user_id_card)
    return base_retriever


def format_docs(docs: List[Document]) -> str:
    """格式化检索到的文档，用于注入 Prompt 的 {context} 变量"""
    return "\n\n".join([
        f"来源: {doc.metadata.get('source', 'N/A')}. 内容: {doc.page_content}"
        for doc in docs
    ])

def format_history_for_prompt(history: List[Any]) -> List[BaseMessage]:
    """
    将历史格式 (Dict, BaseMessage, 或 models.api_models.ChatMessage) 转换为 LangChain ChatMessage 格式。
    """
    lc_messages = []
    for msg in history:
        
        if isinstance(msg, BaseMessage):
            lc_messages.append(msg)
            continue
            
        elif isinstance(msg, ChatMessage):
            role = msg.role
            content = msg.content
            
        elif isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
        else:
             logger.warning(f"跳过未知历史消息类型: {type(msg)}")
             continue

        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))

    return lc_messages


# 异步函数，用于在 LCEL 中获取历史记录
async def get_history_messages(input_data: Dict[str, Any]) -> List[Any]: 
    """异步获取并返回聊天历史记录"""
    session_id = input_data.get("session_id", "default")
    limit = settings.HISTORY_LIMIT or 10
    return await session_store.get_session_messages(session_id, limit=limit)


# --- 新增：用于绕过 LCEL 聚合的 RAG Prompt 构建函数 ---
async def get_rag_prompt_messages(question: str, session_id: str, user_id_card: Optional[str] = None) -> List[BaseMessage]:
    """
    异步获取 RAG 所需的上下文和历史记录，并生成完整的 LangChain ChatMessage 列表。
    """
    
    # 1. 获取历史记录
    history_dicts = await get_history_messages({"session_id": session_id})
    chat_history = format_history_for_prompt(history_dicts)

    # 2. 检索并格式化上下文
    base_retriever = create_rag_retriever(user_id_card=user_id_card)
    reorder = LongContextReorder()
    
    retrieved_docs = await base_retriever.ainvoke(question)
    reordered_docs = reorder.transform_documents(retrieved_docs)
    context_str = format_docs(reordered_docs)
    
    # 3. 构建 RAG System Prompt
    # RAG_PROMPT 的第一个消息是 SystemMessagePromptTemplate
    system_template_instance = RAG_PROMPT.messages[0]
    final_system_message = system_template_instance.prompt.template.format(context=context_str)
    
    # 4. 组合最终消息列表
    final_messages = [
        SystemMessagePromptTemplate.from_template(final_system_message).format(),
    ]
    final_messages.extend(chat_history)
    
    # Human 消息
    final_messages.append(HumanMessagePromptTemplate.from_template(question).format())
    
    return final_messages


# --- 4. 暴露调用接口 (修改为直接调用 LLM) ---

async def invoke_rag(question: str, session_id: str, user_id_card: Optional[str] = None):
    """异步同步调用接口 - 直接调用 LLM"""
    
    all_messages = await get_rag_prompt_messages(question, session_id, user_id_card)

    # 直接调用 LLM 的 ainvoke
    result_msg = await LLM.ainvoke(all_messages)
    result = result_msg.content 
    
    await session_store.add_message(session_id, "user", question)
    await session_store.add_message(session_id, "assistant", result)
    return result

async def astream_rag(question: str, session_id: str, user_id_card: Optional[str] = None):
    """异步流式调用接口 - 直接调用 LLM"""
    
    # 1. 准备所有消息 (包括 RAG 增强)
    all_messages = await get_rag_prompt_messages(question, session_id, user_id_card)
    
    # 2. 调用 LLM 的 astream 方法
    await session_store.add_message(session_id, "user", question)
    
    full_response = ""
    # ⭐️ 关键：直接调用 LLM 的 astream ⭐️
    async for chunk in LLM.astream(all_messages):
        # LLM.astream 返回 ChatGenerationChunk 
        content = chunk.content
        if isinstance(content, str):
            full_response += content
            yield content
        
    await session_store.add_message(session_id, "assistant", full_response)