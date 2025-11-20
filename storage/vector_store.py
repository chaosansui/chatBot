import os
from typing import Optional, List, Dict, Any
from loguru import logger

# --- 1. 引入新版官方库 ---
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载器
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader

from core.config import settings

class MilvusVectorStore:
    def __init__(self):
        # 初始化 Embedding 模型
        # 注意：这里假设你有一个兼容 OpenAI 接口的本地 Embedding 服务 (如 vLLM/Ollama)
        self.embeddings = OpenAIEmbeddings(
            openai_api_base=settings.EMBEDDING_API_URL.rsplit('/', 1)[0], 
            model=settings.EMBEDDING_MODEL_NAME,
            openai_api_key="sk-not-needed-for-local", # 本地模型通常不需要 Key
            check_embedding_ctx_length=False, # 关闭长度检查以避免报错
        )

        self.collection_name = settings.MILVUS_COLLECTION_NAME
        
        # Milvus 连接参数
        self.connection_args = {
            "host": settings.MILVUS_HOST,
            "port": settings.MILVUS_PORT,
            "user": settings.MILVUS_USER,
            "password": settings.MILVUS_PASSWORD,
            "secure": settings.MILVUS_SECURE
        }
        
        # 缓存 store 实例
        self._store: Optional[Milvus] = None

    @property
    def vector_store(self) -> Milvus:
        """
        懒加载获取 Milvus 实例。
        LangChain 的 Milvus 类会自动处理连接复用。
        """
        if self._store is None:
            self._store = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                connection_args=self.connection_args,
                auto_id=True,  # 让 Milvus 自动生成 ID
                drop_old=False # 默认不删除旧数据
            )
        return self._store

    @property
    def collection(self):
        """暴露底层的 pymilvus Collection 对象，供 main.py 做 health check 或 load"""
        # 触发一次初始化
        _ = self.vector_store
        return self._store.col if self._store else None
    
    @property
    def embeddings_model(self):
        """暴露 embedding 对象供外部预热"""
        return self.embeddings

    async def connect_milvus(self):
        """
        显式连接测试。
        在 LangChain 新版中，实例化 Milvus 对象即建立了连接。
        这里主要用于检查连接是否通畅。
        """
        try:
            # 访问一下集合属性来触发连接
            col = self.vector_store.col
            if col:
                logger.info(f"✅ Milvus 已连接，当前集合: {self.collection_name}")
            else:
                logger.warning(f"⚠️ Milvus 已连接，但集合 {self.collection_name} 尚未创建")
        except Exception as e:
            logger.error(f"❌ Milvus 连接失败: {e}")
            # 这里不 raise，允许应用降级启动，但在调用检索时会报错
    
    async def index_documents(self, file_paths: List[str]):
        """
        加载文件 -> 切分 -> 存入 Milvus
        """
        if not file_paths:
            logger.warning("没有文件需要索引")
            return

        all_documents: List[Document] = []
        logger.info(f"📄 开始加载 {len(file_paths)} 个文档...")

        for path in file_paths:
            try:
                ext = os.path.splitext(path)[1].lower()
                loader = None
                
                if ext == ".pdf":
                    loader = PyPDFLoader(path)
                elif ext == ".txt":
                    loader = TextLoader(path, encoding='utf-8')
                elif ext == ".docx":
                    loader = Docx2txtLoader(path)
                elif ext == ".md":
                    try:
                        loader = UnstructuredMarkdownLoader(path)
                    except ImportError:
                        logger.warning("未安装 unstructured，降级使用 TextLoader 加载 Markdown")
                        loader = TextLoader(path, encoding='utf-8')
                
                if loader:
                    docs = loader.load()
                    # 补充元数据 source，防止 loader 没加
                    for doc in docs:
                        if "source" not in doc.metadata:
                            doc.metadata["source"] = os.path.basename(path)
                    
                    all_documents.extend(docs)
                    logger.info(f"   - {os.path.basename(path)}: 加载成功 ({len(docs)} 页/块)")
                else:
                    logger.warning(f"   - 跳过不支持的文件类型: {path}")

            except Exception as e:
                logger.error(f"   - 加载文件 {path} 失败: {e}")

        if not all_documents:
            return

        # --- 2. 优化切分策略 ---
        logger.info("✂️ 开始切分文档...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,      # [优化] 增大到 800，保留更多上下文
            chunk_overlap=100,   # [优化] 增大重叠，防止句子被截断
            separators=["\n\n", "\n", "。", "！", "？", " ", ""], # 针对中文优化
            length_function=len,
        )
        split_docs = text_splitter.split_documents(all_documents)
        logger.info(f"切分完成，共生成 {len(split_docs)} 个向量片段")

        # --- 3. 存入 Milvus ---
        logger.info(f"💾 正在写入 Milvus Collection: {self.collection_name}...")
        
        # 直接使用 vector_store 实例的 add_documents 方法
        self.vector_store.add_documents(split_docs)
        
        logger.success(f"🎉 成功索引 {len(split_docs)} 条数据！")

    def get_retriever(self, user_id_card: Optional[str] = None) -> VectorStoreRetriever:
        """
        获取检索器，支持 MMR 和 元数据过滤
        """
        search_kwargs: Dict[str, Any] = {
            "k": settings.RAG_TOP_K, # 比如 4
        }

        # [优化] 使用 MMR (最大边际相关性) 而不是默认的 Similarity
        # MMR 会尽量找 "既相关又不同" 的文档，避免找到 4 段完全一样的话
        search_type = "mmr" 
        
        # 如果需要元数据过滤 (Metadata Filtering)
        if user_id_card:
            # 注意: 你的文档必须在 index_documents 时就存入了 user_id_card 字段
            # 否则这里过滤会导致查不到任何数据。
            # 这里的 expr 是 Milvus 特有的过滤语法
            search_kwargs["expr"] = f"user_id_card == '{user_id_card}'"
            logger.debug(f"启用 RAG 过滤: {search_kwargs['expr']}")

        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
# 单例实例
vector_store = MilvusVectorStore()