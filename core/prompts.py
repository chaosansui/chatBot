from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CONTEXTUALIZE_Q_SYSTEM = (
    "给定一段聊天记录和用户最新的问题（该问题可能引用了上下文），"
    "请将该问题改写为一个独立的、无需上下文即可理解的完整问题。"
    "不要回答问题，只需返回改写后的问题；如果无需改写，原样返回。"
)

def get_rewrite_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_Q_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

QA_SYSTEM = (
    "你是一个专业的智能助手。\n"
    "请基于以下检索到的背景信息 (context) 回答问题。\n"
    "如果背景信息里没有答案，请直接说“由于缺乏相关信息，我无法回答这个问题”，不要编造。\n"
    "回答要条理清晰，使用 Markdown 格式。\n\n"
    "背景信息:\n"
    "{context}"
)

def get_qa_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])