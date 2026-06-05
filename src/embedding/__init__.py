"""Embedding sub-package."""

from .base import EmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider
from .service import EmbeddingService

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "EmbeddingService",
]
