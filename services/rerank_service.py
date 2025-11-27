import torch
from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from loguru import logger
from core.config import settings
import setproctitle   
setproctitle.setproctitle("reranker")

class RerankService:
    def __init__(self):
        self._model = None
        self.model_name = getattr(settings, "RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
        self.device = getattr(settings, "RERANK_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = getattr(settings, "RAG_RERANK_TOP_K", 5)

    @property
    def model(self):
        """懒加载模型"""
        if self._model is None:
            logger.info(f"🚀 [Rerank] 正在加载模型: {self.model_name} (Device: {self.device})...")
            try:
                self._model = CrossEncoder(
                    self.model_name, 
                    device=self.device,
                    automodel_args={"torch_dtype": "auto"}
                )
                logger.success("✅ [Rerank] 模型加载完成")
            except Exception as e:
                logger.error(f"❌ [Rerank] 模型加载失败: {e}")
                raise e
        return self._model

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """
        核心逻辑：接收查询和一组文档，返回排序后的 Top-K 文档
        """
        if not docs:
            return []

        # 1. 准备模型输入 pairs: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc.page_content] for doc in docs]

        # 2. 模型打分
        # scores 是一个浮点数列表，分数越高越相关
        scores = self.model.predict(pairs)

        # 3. 将文档和分数绑定
        docs_with_scores = list(zip(docs, scores))

        # 4. 按分数倒序排列 (从高到低)
        sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)

        # 5. 截取 Top K
        top_docs = []
        for doc, score in sorted_docs[:self.top_k]:
            # 把分数写回 metadata，方便调试查看
            doc.metadata["relevance_score"] = float(score)
            top_docs.append(doc)

        logger.info(f"⚖️ [Rerank] 重排序完成: 输入 {len(docs)} -> 输出 {len(top_docs)}")
        return top_docs

rerank_service = RerankService()