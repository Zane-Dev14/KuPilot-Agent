"""Application settings — loaded from .env via Pydantic BaseSettings."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "k8s_failures"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    embedding_device: str = "mps"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    simple_model: str = "llama3.1:8b-instruct-q8_0"
    complex_model: str = "Qwen3-coder:30b"
    ollama_base_url: str = "http://localhost:11434"
    query_complexity_threshold: float = 0.7
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 4
    log_level: str = "INFO"
    memory_path: str = "data/chat_memory.json"
    memory_max_messages: int = 40
    memory_max_sessions: int = 256

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
