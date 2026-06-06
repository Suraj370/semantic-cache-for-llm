"""Semantic caching library.

Cache key strategy
------------------
There is no traditional string key.  Every incoming prompt goes through a
two-part keying process before hitting the vector store:

1. **Context key** (exact match) — ``CacheKeyBuilder`` hashes the LLM
   context (system_prompt, model, temperature, max_tokens) into a short hex
   digest.  Only entries stored under the *same* context key are eligible
   as hits.  This prevents cross-contamination between different system
   prompts or generation parameters.

2. **Semantic key** (similarity match) — the *normalised* query text is
   embedded via ``text-embedding-3-small`` and compared to stored embeddings
   using cosine similarity.  A result must exceed ``threshold`` (default
   0.85) to count as a hit.

Lookup flow::

    raw query
        → QueryNormalizer.normalize()
        → EmbeddingService.embed_one()          # text-embedding-3-small
        → VectorStore.search(
              embedding,
              context_key=key,                  # hard filter (exact)
              threshold=0.85,                   # soft filter (similarity)
          )
        → hit → return cached response
          miss → call LLM, store CacheEntry, return response

Quick start::

    from src import (
        OpenAIEmbeddingProvider, EmbeddingService,
        InMemoryVectorStore,
        QueryNormalizer,
        LLMContext, CacheKeyBuilder,
        CacheEntry,
    )

    # 1. Build subsystems
    provider   = OpenAIEmbeddingProvider(api_key="sk-...")
    embeddings = EmbeddingService(primary=provider)
    store      = InMemoryVectorStore(max_size=10_000)
    normalizer = QueryNormalizer()
    key_builder = CacheKeyBuilder()

    # 2. Define the LLM context (anything that affects the response)
    context = LLMContext(
        system_prompt="You are a concise assistant.",
        model="gpt-4o",
        temperature=0.2,
        max_tokens=512,
    )
    context_key = key_builder.build(context)

    # 3. Lookup
    raw_query  = "What is semantic caching?"
    normalized = normalizer.normalize(raw_query)
    embedding  = embeddings.embed_one(normalized)

    hits = store.search(embedding, k=1, threshold=0.85, context_key=context_key)
    if hits:
        entry, score = hits[0]
        response = entry.response
    else:
        response = call_llm(raw_query)           # your LLM call
        store.add(CacheEntry(
            query=raw_query,
            normalized_query=normalized,
            response=response,
            embedding=embedding,
            context_key=context_key,
        ))
"""

from .cache_key import CacheKeyBuilder, LLMContext, default_key_builder
from .config import CacheConfig, EmbeddingConfig, VectorStoreConfig
from .embedding import EmbeddingProvider, EmbeddingService, OpenAIEmbeddingProvider
from .exceptions import ConfigError, EmbeddingError, SemanticCacheError, VectorStoreError
from .models import CacheEntry
from .query_normalizer import QueryNormalizer
from .similarity import batch_cosine_similarity, cosine_similarity, find_most_similar
from .ttl_classifier import TtlClassifier, TtlTier, default_classifier
from .vector_store import InMemoryVectorStore, VectorStore

__all__ = [
    # Config
    "CacheConfig",
    "EmbeddingConfig",
    "VectorStoreConfig",
    # Cache key
    "LLMContext",
    "CacheKeyBuilder",
    "default_key_builder",
    # Embedding
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "EmbeddingService",
    # Vector store
    "VectorStore",
    "InMemoryVectorStore",
    # Domain model
    "CacheEntry",
    # TTL classification
    "TtlTier",
    "TtlClassifier",
    "default_classifier",
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
