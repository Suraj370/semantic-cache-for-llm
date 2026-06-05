"""Central configuration for the semantic caching library.

All parameters can be overridden via ``SEMANTIC_CACHE_*`` environment
variables, making the library 12-factor-app compatible out of the box.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .exceptions import ConfigError

# text-embedding-3-small produces 1536-dimensional vectors by default.
# The `dimensions` API parameter can reduce this (e.g. to 512) for faster
# search at a small quality cost.
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_DEFAULT_EMBEDDING_DIM = 1536


@dataclass
class EmbeddingConfig:
    """Configuration for the OpenAI embedding provider."""

    api_key: Optional[str] = None
    model: str = _DEFAULT_EMBEDDING_MODEL
    dimensions: Optional[int] = None  # None → use model default (1536)
    timeout: float = 30.0
    max_retries: int = 3
    batch_size: int = 100

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ConfigError(f"embedding timeout must be positive, got {self.timeout}")
        if self.max_retries < 0:
            raise ConfigError(f"embedding max_retries must be >= 0, got {self.max_retries}")
        if self.dimensions is not None and self.dimensions < 1:
            raise ConfigError(f"embedding dimensions must be >= 1, got {self.dimensions}")

    @property
    def effective_dim(self) -> int:
        return self.dimensions if self.dimensions is not None else _DEFAULT_EMBEDDING_DIM


@dataclass
class VectorStoreConfig:
    """Configuration for the in-memory vector store."""

    similarity_threshold: float = 0.85
    max_size: int = 10_000
    default_ttl: Optional[int] = 3600  # seconds; None = never expire
    use_faiss: bool = True
    faiss_threshold: int = 500  # switch to FAISS after this many entries

    def __post_init__(self) -> None:
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ConfigError(
                f"similarity_threshold must be in [0, 1], got {self.similarity_threshold}"
            )
        if self.max_size < 0:
            raise ConfigError(f"max_size must be >= 0, got {self.max_size}")


@dataclass
class CacheConfig:
    """Top-level configuration object for the semantic cache.

    Compose an :class:`EmbeddingConfig` and :class:`VectorStoreConfig` or
    use :meth:`from_env` to load everything from environment variables.
    """

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    logger_name: str = "semantic_cache"

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Construct a :class:`CacheConfig` from environment variables."""
        raw_ttl = os.getenv("SEMANTIC_CACHE_TTL")
        default_ttl: Optional[int] = int(raw_ttl) if raw_ttl is not None else 3600

        raw_dim = os.getenv("SEMANTIC_CACHE_EMBEDDING_DIM")
        dimensions: Optional[int] = int(raw_dim) if raw_dim is not None else None

        embedding = EmbeddingConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("SEMANTIC_CACHE_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL),
            dimensions=dimensions,
            timeout=float(os.getenv("SEMANTIC_CACHE_EMBEDDING_TIMEOUT", "30.0")),
            max_retries=int(os.getenv("SEMANTIC_CACHE_EMBEDDING_MAX_RETRIES", "3")),
            batch_size=int(os.getenv("SEMANTIC_CACHE_EMBEDDING_BATCH_SIZE", "100")),
        )

        vector_store = VectorStoreConfig(
            similarity_threshold=float(
                os.getenv("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", "0.85")
            ),
            max_size=int(os.getenv("SEMANTIC_CACHE_MAX_SIZE", "10000")),
            default_ttl=default_ttl,
            use_faiss=os.getenv("SEMANTIC_CACHE_USE_FAISS", "true").lower()
            in {"true", "1", "yes"},
            faiss_threshold=int(os.getenv("SEMANTIC_CACHE_FAISS_THRESHOLD", "500")),
        )

        return cls(
            embedding=embedding,
            vector_store=vector_store,
            logger_name=os.getenv("SEMANTIC_CACHE_LOGGER_NAME", "semantic_cache"),
        )
