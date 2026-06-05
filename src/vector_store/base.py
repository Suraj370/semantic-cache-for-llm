"""Abstract interface for semantic-cache vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Tuple

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
    ) -> List[Tuple[CacheEntry, float]]:
        """Find the *k* most similar live (non-expired) entries.

        Args:
            embedding: Query vector produced by the embedding service.
            k: Maximum number of results to return.
            threshold: Minimum cosine similarity for a result to qualify.

        Returns:
            ``[(entry, similarity), ...]`` sorted by similarity descending.
            Empty list when nothing meets the threshold.

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
