"""High-level embedding service with primary + optional fallback provider."""

from __future__ import annotations

import logging
from typing import List, Optional

from ..exceptions import EmbeddingError
from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wrap one or two :class:`EmbeddingProvider` instances.

    If the primary provider raises :class:`~src.exceptions.EmbeddingError`
    the fallback is tried before re-raising.  This lets callers combine a
    remote OpenAI provider with a local sentence-transformers fallback for
    offline resilience.

    Args:
        primary: Primary provider (typically
            :class:`~src.embedding.openai_provider.OpenAIEmbeddingProvider`).
        fallback: Optional fallback provider, called only when primary fails.
    """

    def __init__(
        self,
        primary: EmbeddingProvider,
        fallback: Optional[EmbeddingProvider] = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed *texts*, falling back to secondary provider on failure."""
        try:
            return self._primary.embed(texts)
        except EmbeddingError as exc:
            if self._fallback is not None:
                logger.warning("Primary embedding provider failed, using fallback: %s", exc)
                return self._fallback.embed(texts)
            raise

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    # ------------------------------------------------------------------
    # Async
    # ------------------------------------------------------------------

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        """Embed *texts* asynchronously, falling back on failure."""
        try:
            return await self._primary.aembed(texts)
        except EmbeddingError as exc:
            if self._fallback is not None:
                logger.warning(
                    "Primary embedding provider failed (async), using fallback: %s", exc
                )
                return await self._fallback.aembed(texts)
            raise

    async def aembed_one(self, text: str) -> List[float]:
        return (await self.aembed([text]))[0]
