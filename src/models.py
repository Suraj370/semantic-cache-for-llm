"""Domain model for cached entries."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CacheEntry:
    """A single record stored in the semantic cache.

    The embedding field holds the vector for *normalized_query*, which is
    what the vector store uses to compute cosine similarity at lookup time.
    There is no separate string-based cache key — semantic identity is
    determined entirely by embedding distance.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    normalized_query: str = ""
    response: str = ""
    embedding: List[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def is_expired(self) -> bool:
        """Return True if the entry's TTL has elapsed."""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl

    def touch(self) -> None:
        """Record a cache hit: bump access_count and refresh last_accessed."""
        self.last_accessed = time.time()
        self.access_count += 1

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "normalized_query": self.normalized_query,
            "response": self.response,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "ttl": self.ttl,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        return cls(
            id=data["id"],
            query=data["query"],
            normalized_query=data["normalized_query"],
            response=data["response"],
            embedding=data["embedding"],
            created_at=data["created_at"],
            last_accessed=data["last_accessed"],
            access_count=data["access_count"],
            ttl=data.get("ttl"),
            metadata=data.get("metadata", {}),
        )
