"""Embeddings, Milvus vector store, and cross-encoder reranker."""

import logging
from functools import lru_cache

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from sentence_transformers import CrossEncoder

from src.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    s = get_settings()
    logger.info("Loading embeddings: %s (device=%s)", s.embedding_model, s.embedding_device)
    return HuggingFaceEmbeddings(
        model_name=s.embedding_model,
        model_kwargs={"device": s.embedding_device},
        encode_kwargs={"normalize_embeddings": True})


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    s = get_settings()
    logger.info("Loading reranker: %s", s.reranker_model)
    return CrossEncoder(s.reranker_model)


def rerank(docs: list[Document], query: str, top_k: int = 4) -> list[Document]:
    if not docs:
        return []
    scores = get_reranker().predict([[query, d.page_content] for d in docs])
    return [d for d, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]]


class MilvusStore:
    def __init__(self, drop_old=False):
        self._settings = get_settings()
        self._drop_old = drop_old
        self._vs = None

    def _get_vs(self):
        if self._vs is None:
            s = self._settings
            self._vs = Milvus(
                embedding_function=get_embeddings(),
                collection_name=s.milvus_collection,
                connection_args={"uri": s.milvus_uri},
                drop_old=self._drop_old, auto_id=True)
        return self._vs

    def add_documents(self, docs: list[Document]) -> list[str]:
        logger.info("Storing %d chunks", len(docs))
        return self._get_vs().add_documents(docs)

    def search(self, query: str, k: int | None = None) -> list[Document]:
        k = k or self._settings.retrieval_top_k
        docs = self._get_vs().similarity_search(query, k=min(k * 3, 20))
        return rerank(docs, query, top_k=k)

    def health_check(self) -> bool:
        try:
            self._get_vs()
            return True
        except Exception:
            return False
