from functools import lru_cache
from urllib.parse import urlparse

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import get_settings


@lru_cache
def get_vector_store() -> PGVectorStore:
    s = get_settings()
    url = urlparse(s.database_url.replace("postgresql+psycopg", "postgresql"))
    return PGVectorStore.from_params(
        database=url.path.lstrip("/"),
        host=url.hostname,
        port=str(url.port or 5432),
        user=url.username,
        password=url.password,
        table_name=s.vector_table_name,  # llama-index stores it as data_rag_chunks
        embed_dim=s.embed_dim,
        hybrid_search=True,
        text_search_config="english",
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )


@lru_cache
def get_embed_model() -> OllamaEmbedding:
    s = get_settings()
    return OllamaEmbedding(model_name=s.embed_model, base_url=s.ollama_base_url)
