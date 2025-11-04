# core/chains.py

from typing import Dict, Any, List
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from services.llm_service import llm_service 
from services.rag_service import rag_service
from storage.session_store import session_store
from core.config import settings 

LLM = llm_service.langchain_llm
RETRIEVER = rag_service.get_retriever() # ❗ 假设 rag_service.py 中已实现此方法

# ----------------------------------------------------
# 2. RAG Prompt 模板
# ----------------------------------------------------

# 好的 RAG Prompt 应该包含系统指令、检索到的上下文和用户提问
RAG_PROMPT_TEMPLATE = """
你是一个专业的RAG聊天机器人，请根据提供的【上下文】来回答用户的问题。
你的回答必须准确、简洁、专业，且仅基于【上下文】中的信息。
如果【上下文】中没有相关信息，请礼貌地说明你找不到答案，但不要捏造事实。

【上下文】:
{context}

用户问题: {question}

回答:
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=RAG_PROMPT_TEMPLATE),
])


CONDENSE_PROMPT_TEMPLATE = """
根据当前的对话历史和用户的最新提问，将其重写为一个**独立的**、**无歧义**的、**适合信息检索**的查询语句。
如果最新提问本身已经足够清晰，则直接返回该提问。

<对话历史>
{chat_history}

<最新提问>
{question}

独立查询:
"""

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=CONDENSE_PROMPT_TEMPLATE),
])

CONDENSE_LLM = llm_service.langchain_llm 

condense_question_chain = (
    CONDENSE_PROMPT 
    | CONDENSE_LLM 
    | StrOutputParser()
).with_config(run_name="Condense_Question_Chain")


def format_docs(docs: List[Any]) -> str:
    """将检索到的文档列表格式化成一个字符串，用于放入 Prompt 的 {context} 变量中"""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def get_history_by_session_id(input_dict: Dict[str, Any]) -> str:

    session_id = input_dict.get("session_id")
    if not session_id:
        return ""
    
    history_messages = session_store.get_history(session_id) 
    
    # 格式化历史记录为 Prompt 可读的字符串
    formatted_history = "\n".join([
        f"{m['role'].capitalize()}: {m['content']}" for m in history_messages
    ])
    return formatted_history


full_rag_chain = RunnablePassthrough.assign(
    chat_history=RunnableLambda(get_history_by_session_id)
).assign(

    retrieval_question=RunnablePassthrough.assign(
        question_to_rephrase=lambda x: x["question"]
    ) | RunnableLambda(lambda x: condense_question_chain.invoke({
        "question": x["question_to_rephrase"],
        "chat_history": x["chat_history"]
    }) if x["chat_history"] else x["question"])
).assign(
    context=RunnablePassthrough.assign(
        documents=lambda x: RETRIEVER.invoke(x["retrieval_question"])
    ) | RunnableLambda(lambda x: format_docs(x["documents"])),
).assign(
   
    answer=RAG_PROMPT | LLM | StrOutputParser()
).pick("answer", "context")


async def get_streaming_rag_chain(session_id: str = "default_session"):
    return full_rag_chain

async def update_chat_history(session_id: str, question: str, answer: str):
    """更新会话存储中的对话历史记录"""
    if session_id:
        await session_store.add_message(session_id, "user", question)
        await session_store.add_message(session_id, "assistant", answer)