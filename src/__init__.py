"""Semantic caching library.

Quick start::

    from src.config import CacheConfig, EmbeddingConfig, VectorStoreConfig
    from src.embedding import OpenAIEmbeddingProvider, EmbeddingService
    from src.vector_store import InMemoryVectorStore
    from src.query_normalizer import QueryNormalizer
    from src.models import CacheEntry

    # 1. Build embedding service (text-embedding-3-small)
    provider = OpenAIEmbeddingProvider(api_key="sk-...")
    embedding_service = EmbeddingService(primary=provider)

    # 2. Build vector store
    store = InMemoryVectorStore(max_size=10_000)

    # 3. Cache a prompt
    normalizer = QueryNormalizer()
    raw_query = "What is semantic caching?"
    normalized = normalizer.normalize(raw_query)
    embedding = embedding_service.embed_one(normalized)

    hits = store.search(embedding, k=1, threshold=0.85)
    if hits:
        entry, score = hits[0]
        response = entry.response
    else:
        response = call_llm(raw_query)           # your LLM call
        entry = CacheEntry(
            query=raw_query,
            normalized_query=normalized,
            response=response,
            embedding=embedding,
        )
        store.add(entry)
"""

from .config import CacheConfig, EmbeddingConfig, VectorStoreConfig
from .embedding import EmbeddingProvider, EmbeddingService, OpenAIEmbeddingProvider
from .exceptions import ConfigError, EmbeddingError, SemanticCacheError, VectorStoreError
from .models import CacheEntry
from .query_normalizer import QueryNormalizer
from .similarity import batch_cosine_similarity, cosine_similarity, find_most_similar
from .vector_store import InMemoryVectorStore, VectorStore

__all__ = [
    # Config
    "CacheConfig",
    "EmbeddingConfig",
    "VectorStoreConfig",
    # Embedding
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "EmbeddingService",
    # Vector store
    "VectorStore",
    "InMemoryVectorStore",
    # Domain model
    "CacheEntry",
    # Utilities
    "QueryNormalizer",
    "cosine_similarity",
    "batch_cosine_similarity",
    "find_most_similar",
    # Exceptions
    "SemanticCacheError",
    "EmbeddingError",
    "VectorStoreError",
    "ConfigError",
]

__version__ = "1.0.0"
