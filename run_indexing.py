import asyncio
import os
import sys
from services.rag_service import rag_service
from loguru import logger

# 1. 配置你的文档路径 (支持绝对路径或相对路径)
DOCUMENT_PATHS = [
    # 请确保这个路径下真的有文件
    "/mnt/data/AI-chatBot/data/files/shouce.md"
]

# 2. 配置索引完成后的测试问题 (一定要改成和你文档相关的问题！)
TEST_QUERY = "在这里填写一个手册里包含的问题，比如：员工怎么请假？" 

async def index_documents():
    """
    执行文档的加载、切分、嵌入和存储到 Milvus 的过程。
    """
    logger.info("🚀 启动索引脚本...")
    
    # --- 步骤 1: 检查文件是否存在 ---
    valid_paths = []
    for path in DOCUMENT_PATHS:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            logger.warning(f"⚠️ 文件未找到: {path}")

    if not valid_paths:
        logger.error("❌ 没有找到任何有效文件，脚本终止。")
        return

    logger.info(f"✅ 找到 {len(valid_paths)} 个待处理文件。")
    
    try:
        # --- 步骤 2: 初始化服务 (确保 Embedding 模型加载) ---
        # 这一步能提前暴露连接问题
        await rag_service.initialize()

        # --- 步骤 3: 执行核心索引逻辑 ---
        # 这会调用我们刚才优化的 vector_store.index_documents (Chunk=800)
        await rag_service.process_data(file_paths=valid_paths)
        
        logger.success("🎉 文档索引流程执行完毕！")
        
        # --- 步骤 4: 验证检索效果 ---
        logger.info("🔍 正在执行自测检索...")
        
        # 使用 rag_service 获取检索器
        retriever = rag_service.get_retriever()
        
        # 执行检索
        docs = await retriever.ainvoke(TEST_QUERY)
        
        if docs:
            logger.success(f"✅ 检索测试通过！共找到 {len(docs)} 条相关片段。")
            logger.info(f"📌 Top 1 结果预览:\n" + "-"*50 + f"\n{docs[0].page_content[:200]}...\n" + "-"*50)
            logger.info(f"📄 来源文件: {docs[0].metadata.get('source', '未知')}")
        else:
            logger.warning(f"⚠️ 检索结果为空！可能原因：\n1. 测试问题 '{TEST_QUERY}' 与文档无关\n2. 向量嵌入失败")
        
    except Exception as e:
        logger.error(f"❌ 索引过程中发生错误: {e}")
        # 打印详细堆栈以便调试
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
        
    asyncio.run(index_documents())