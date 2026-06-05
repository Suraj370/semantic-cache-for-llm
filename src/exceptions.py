"""Custom exception hierarchy for the semantic caching library."""

from __future__ import annotations


class SemanticCacheError(Exception):
    """Base for all semantic-cache exceptions."""


class EmbeddingError(SemanticCacheError):
    """Raised when an embedding provider fails to generate embeddings."""


class VectorStoreError(SemanticCacheError):
    """Raised when a vector-store operation (add / search / remove) fails."""


class ConfigError(SemanticCacheError):
    """Raised when the cache configuration is invalid or incomplete."""
