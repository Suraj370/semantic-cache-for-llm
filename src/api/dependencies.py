"""Shared service singletons injected via FastAPI Depends.

All heavy objects (OpenAI client, vector store, embedding service) are
created once at startup and reused across requests.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from ..cache_key import CacheKeyBuilder
from ..embedding import EmbeddingService, OpenAIEmbeddingProvider
from ..query_normalizer import QueryNormalizer
from ..vector_store import InMemoryVectorStore


@lru_cache(maxsize=1)
def _embedding_service() -> EmbeddingService:
    api_key = os.environ.get("OPENAI_API_KEY")
    provider = OpenAIEmbeddingProvider(api_key=api_key)
    return EmbeddingService(primary=provider)


@lru_cache(maxsize=1)
def _vector_store() -> InMemoryVectorStore:
    max_size = int(os.environ.get("SEMANTIC_CACHE_MAX_SIZE", "10000"))
    return InMemoryVectorStore(max_size=max_size)


@lru_cache(maxsize=1)
def _normalizer() -> QueryNormalizer:
    return QueryNormalizer()


@lru_cache(maxsize=1)
def _key_builder() -> CacheKeyBuilder:
    return CacheKeyBuilder()


@lru_cache(maxsize=1)
def _similarity_threshold() -> float:
    return float(os.environ.get("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", "0.85"))


# ---------------------------------------------------------------------------
# FastAPI dependency callables (one indirection keeps signatures clean)
# ---------------------------------------------------------------------------

def get_embedding_service() -> EmbeddingService:
    return _embedding_service()


def get_vector_store() -> InMemoryVectorStore:
    return _vector_store()


def get_normalizer() -> QueryNormalizer:
    return _normalizer()


def get_key_builder() -> CacheKeyBuilder:
    return _key_builder()


def get_similarity_threshold() -> float:
    return _similarity_threshold()


# ---------------------------------------------------------------------------
# Annotated type aliases — use these in route signatures
# ---------------------------------------------------------------------------

EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
VectorStoreDep = Annotated[InMemoryVectorStore, Depends(get_vector_store)]
NormalizerDep = Annotated[QueryNormalizer, Depends(get_normalizer)]
KeyBuilderDep = Annotated[CacheKeyBuilder, Depends(get_key_builder)]
ThresholdDep = Annotated[float, Depends(get_similarity_threshold)]
