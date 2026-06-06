"""Abstract interface for semantic-cache vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from ..models import CacheEntry


class VectorStore(ABC):
    """Contract that all vector-store backends must satisfy.

    Every mutating method must be thread-safe in concrete implementations.
    Callers never interact with raw arrays — they always work with
    :class:`~src.models.CacheEntry` objects and float similarity scores.
    """

    @abstractmethod
    def add(self, entry: CacheEntry) -> None:
        """Persist *entry* and index its embedding.

        Raises:
            VectorStoreError: On storage failure.
        """

    @abstractmethod
    def search(
        self,
        embedding: List[float],
        k: int = 1,
        threshold: float = 0.85,
        context_key: Optional[str] = None,
    ) -> List[Tuple[CacheEntry, float]]:
        """Find the *k* most similar live (non-expired) entries.

        Args:
            embedding: Query vector produced by the embedding service.
            k: Maximum number of results to return.
            threshold: Minimum cosine similarity for a result to qualify.
            context_key: When supplied, only entries whose ``context_key``
                exactly matches are eligible.  Entries stored without a
                context key (``None``) are never returned when a key is
                provided, preventing cross-contamination between different
                system prompts, models, or generation parameters.

        Returns:
            ``[(entry, similarity), ...]`` sorted by similarity descending.
            Empty list when nothing meets the threshold or context filter.

        Raises:
            VectorStoreError: On search failure.
        """

    @abstractmethod
    def remove(self, entry_id: str) -> bool:
        """Remove the entry identified by *entry_id*.

        Returns:
            ``True`` if the entry existed and was removed.
        """

    @abstractmethod
    def rebuild(self) -> None:
        """Prune expired entries and rebuild the underlying index.

        Call periodically in long-running services to reclaim memory.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the store to *path*.

        Raises:
            VectorStoreError: On I/O failure.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "VectorStore":
        """Load a previously saved store from *path*.

        Raises:
            VectorStoreError: On I/O or deserialisation failure.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries."""

    @abstractmethod
    def size(self) -> int:
        """Return the total number of stored entries (including expired)."""

    @abstractmethod
    def __iter__(self) -> Iterator[CacheEntry]:
        """Iterate over all stored entries (including expired)."""

    # ------------------------------------------------------------------
    # Bulk invalidation helpers (concrete — backends get them for free)
    # ------------------------------------------------------------------

    def invalidate_by_context_key(self, context_key: str) -> int:
        """Remove all entries whose ``context_key`` matches exactly.

        Use this when a system prompt changes: pass the old context key
        (SHA-256 hex digest produced by :class:`~src.cache_key.CacheKeyBuilder`)
        to wipe every cached response that was generated under it.

        Returns:
            Number of entries removed.
        """
        ids = [e.id for e in self if e.context_key == context_key]
        return sum(1 for eid in ids if self.remove(eid))

    def invalidate_by_model(self, model: str) -> int:
        """Remove all entries whose ``metadata["model"]`` equals *model*.

        Use this after a model upgrade: pass the old model name (e.g.
        ``"gpt-4o"``). Entries for the new model will be populated fresh.

        Returns:
            Number of entries removed.
        """
        ids = [e.id for e in self if e.metadata.get("model") == model]
        return sum(1 for eid in ids if self.remove(eid))

    def invalidate_by_query_prefix(self, prefix: str) -> int:
        """Remove all entries whose original query starts with *prefix*.

        Useful for topic-scoped manual invalidation (e.g. all queries
        beginning with ``"Tell me about our pricing"``).

        Returns:
            Number of entries removed.
        """
        lower = prefix.lower()
        ids = [e.id for e in self if e.query.lower().startswith(lower)]
        return sum(1 for eid in ids if self.remove(eid))

    def invalidate_by_tag(self, tag: str) -> int:
        """Remove all entries that carry *tag* in ``metadata["tags"]``.

        Tags are written as a list of strings stored under the ``"tags"``
        key in :attr:`~src.models.CacheEntry.metadata`.

        Returns:
            Number of entries removed.
        """
        ids = [
            e.id for e in self
            if tag in e.metadata.get("tags", [])
        ]
        return sum(1 for eid in ids if self.remove(eid))
