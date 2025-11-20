#run_indexing.py

import asyncio
import os
from services.rag_service import rag_service
from loguru import logger

# 添加更多文档
DOCUMENT_PATHS = [
    "/mnt/data/AI-chatBot/data/files/shouce.md"
]

async def index_documents():
    """
    执行文档的加载、切分、嵌入和存储到 Milvus 的过程。
    """
    logger.info("📄 开始检查文档并执行 RAG 索引过程...")
    
    # 1. 检查文档是否存在
    valid_paths = [path for path in DOCUMENT_PATHS if os.path.exists(path)]
    if not valid_paths:
        logger.error(f"❌ 找不到任何有效的文档。请确保文档位于指定路径：{DOCUMENT_PATHS}")
        return

    logger.info(f"✅ 找到 {len(valid_paths)} 个文档准备处理。")
    
    try:
        # 2. 调用 rag_service 中的核心处理方法
        await rag_service.process_data(file_paths=valid_paths)
        
        logger.success("🎉 文档索引和 Milvus 存储已完成！")
        
        # 3. 验证数据是否真的进入 Milvus
        retriever = rag_service.get_retriever()
        if retriever:
            test_query = "什么是 RAG Chain 的核心作用？"
            docs = retriever.invoke(test_query) 
            
            logger.info(f"🔍 使用测试查询'{test_query}'检索到 {len(docs)} 个文档片段。")
            if docs:
                logger.info(f"   - 第一个文档片段内容摘要: {docs[0].page_content[:100]}...")
            else:
                logger.warning("⚠️ 测试检索结果为空，请检查 Milvus 连接、集合名称和索引过程。")
        
    except Exception as e:
        logger.error(f"❌ 数据索引过程中发生致命错误: {e}")

if __name__ == "__main__":
    # 需要在异步环境中运行
    asyncio.run(index_documents())