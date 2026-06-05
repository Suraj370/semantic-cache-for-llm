"""Thread-safe in-memory vector store backed by numpy (+ optional FAISS).

Cache key strategy (mirroring src-old design):
    There is no traditional string key.  The *embedding* of the normalised
    query is the key.  Lookup is a nearest-neighbour search: entries whose
    embedding is within ``threshold`` cosine similarity of the query
    embedding are considered hits.  This makes the cache resilient to
    paraphrasing, capitalisation changes, and filler-word variation.

Index strategy:
    - < ``faiss_threshold`` entries  → brute-force numpy cosine similarity
      (fast enough for small stores; no extra dependency required).
    - ≥ ``faiss_threshold`` entries  → FAISS ``IndexFlatIP`` on L2-normalised
      vectors (inner-product ≡ cosine similarity after normalisation).
      Built lazily and rebuilt when the store changes (``_index_dirty``).

Eviction:
    When ``max_size > 0`` and the store is full, the 10 % least-recently-used
    entries are evicted before each ``add``.  Expired entries are filtered at
    search time and compacted during :meth:`rebuild`.
"""

from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..exceptions import VectorStoreError
from ..models import CacheEntry
from ..similarity import batch_cosine_similarity
from .base import VectorStore

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore[import]

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.debug("faiss not installed; InMemoryVectorStore will use numpy search.")


class InMemoryVectorStore(VectorStore):
    """Thread-safe, numpy-backed vector store with optional FAISS acceleration.

    Args:
        use_faiss: Enable FAISS when available. Default ``True``.
        faiss_threshold: Switch from numpy to FAISS above this many entries.
        max_size: Hard cap on stored entries; ``0`` = unlimited.
    """

    def __init__(
        self,
        use_faiss: bool = True,
        faiss_threshold: int = 500,
        max_size: int = 10_000,
    ) -> None:
        self._use_faiss = use_faiss and _FAISS_AVAILABLE
        self._faiss_threshold = faiss_threshold
        self._max_size = max_size
        self._lock = threading.RLock()

        # id → CacheEntry
        self._entries: Dict[str, CacheEntry] = {}
        # insertion-ordered id list — needed to map FAISS row indices back to ids
        self._id_list: List[str] = []
        # (n, d) float32 embedding matrix; None when store is empty
        self._embeddings: Optional[np.ndarray] = None
        # FAISS index; rebuilt lazily
        self._index: Any = None
        self._index_dirty: bool = False

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add(self, entry: CacheEntry) -> None:
        """Add or update *entry* in the store.

        If the store is at capacity the oldest-accessed entries are evicted
        before the new entry is inserted.
        """
        with self._lock:
            if entry.id in self._entries:
                self._update_existing(entry)
                return

            if self._max_size > 0 and len(self._entries) >= self._max_size:
                self._evict_lru(count=max(1, self._max_size // 10))

            vec = np.array(entry.embedding, dtype=np.float32)
            self._entries[entry.id] = entry
            self._id_list.append(entry.id)
            self._embeddings = (
                vec.reshape(1, -1)
                if self._embeddings is None
                else np.vstack([self._embeddings, vec])
            )
            self._index_dirty = True

    def search(
        self,
        embedding: List[float],
        k: int = 1,
        threshold: float = 0.85,
        context_key: Optional[str] = None,
    ) -> List[Tuple[CacheEntry, float]]:
        """Return up to *k* non-expired entries with similarity ≥ *threshold*.

        When *context_key* is provided only entries whose stored
        ``context_key`` matches exactly are eligible.  This prevents hits
        across different system prompts, models, or generation parameters.
        """
        with self._lock:
            if self._embeddings is None or not self._entries:
                return []

            query = np.array(embedding, dtype=np.float32)

            # Fetch more candidates than k so context/expiry filtering
            # doesn't leave us short of results.
            raw = (
                self._faiss_search(query, k=k * 4)
                if self._use_faiss and len(self._id_list) >= self._faiss_threshold
                else self._numpy_search(query, k=k * 4)
            )

            results: List[Tuple[CacheEntry, float]] = []
            for idx, sim in raw:
                if sim < threshold:
                    break  # sorted descending — no point continuing
                entry_id = self._id_list[idx]
                entry = self._entries.get(entry_id)
                if entry is None or entry.is_expired():
                    continue
                # Hard filter: context keys must match exactly.
                # An entry stored without a key (None) is never returned
                # when the caller supplies a key, and vice-versa.
                if context_key != entry.context_key:
                    continue
                results.append((entry, float(sim)))
                if len(results) >= k:
                    break

            return results

    def remove(self, entry_id: str) -> bool:
        """Remove an entry by ID. Returns ``True`` if it existed."""
        with self._lock:
            if entry_id not in self._entries:
                return False

            idx = self._id_list.index(entry_id)
            del self._entries[entry_id]
            self._id_list.pop(idx)

            if self._embeddings is not None:
                self._embeddings = np.delete(self._embeddings, idx, axis=0)
                if len(self._embeddings) == 0:
                    self._embeddings = None

            self._index_dirty = True
            return True

    def rebuild(self) -> None:
        """Prune expired entries and rebuild the FAISS index."""
        with self._lock:
            expired = [eid for eid, e in self._entries.items() if e.is_expired()]
            for eid in expired:
                self.remove(eid)
            self._index = None
            self._index_dirty = True
            logger.info(
                "Rebuilt vector store: removed %d expired entries, %d remain",
                len(expired),
                len(self._entries),
            )

    def save(self, path: Path) -> None:
        """Pickle the store to *path*.

        Raises:
            VectorStoreError: On I/O failure.
        """
        path = Path(path)
        try:
            with self._lock:
                state = {
                    "entries": self._entries,
                    "id_list": self._id_list,
                    "embeddings": self._embeddings,
                    "use_faiss": self._use_faiss,
                    "faiss_threshold": self._faiss_threshold,
                    "max_size": self._max_size,
                }
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as fh:
                pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Saved InMemoryVectorStore to %s", path)
        except Exception as exc:
            raise VectorStoreError(f"Failed to save store: {exc}") from exc

    @classmethod
    def load(cls, path: Path) -> "InMemoryVectorStore":
        """Load a store from a pickle file created by :meth:`save`.

        Raises:
            VectorStoreError: On I/O or deserialisation failure.
        """
        path = Path(path)
        try:
            with path.open("rb") as fh:
                state = pickle.load(fh)
            store = cls(
                use_faiss=state.get("use_faiss", True),
                faiss_threshold=state.get("faiss_threshold", 500),
                max_size=state.get("max_size", 10_000),
            )
            store._entries = state["entries"]
            store._id_list = state["id_list"]
            store._embeddings = state["embeddings"]
            store._index_dirty = True
            logger.info(
                "Loaded InMemoryVectorStore from %s (%d entries)",
                path,
                len(store._entries),
            )
            return store
        except Exception as exc:
            raise VectorStoreError(f"Failed to load store: {exc}") from exc

    def clear(self) -> None:
        """Remove all entries and reset the index."""
        with self._lock:
            self._entries.clear()
            self._id_list.clear()
            self._embeddings = None
            self._index = None
            self._index_dirty = False

    def size(self) -> int:
        """Return the total number of stored entries (including expired)."""
        with self._lock:
            return len(self._entries)

    def __iter__(self) -> Iterator[CacheEntry]:
        """Iterate over a snapshot of all stored entries."""
        with self._lock:
            snapshot = list(self._entries.values())
        yield from snapshot

    # ------------------------------------------------------------------
    # Private — search backends
    # ------------------------------------------------------------------

    def _numpy_search(
        self, query: np.ndarray, k: int
    ) -> List[Tuple[int, float]]:
        """Brute-force cosine similarity via BLAS matrix-vector multiply."""
        assert self._embeddings is not None
        sims = batch_cosine_similarity(query, self._embeddings)
        n = min(k, len(sims))
        top_idx = np.argpartition(sims, -n)[-n:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [(int(i), float(sims[i])) for i in top_idx]

    def _faiss_search(
        self, query: np.ndarray, k: int
    ) -> List[Tuple[int, float]]:
        """FAISS IndexFlatIP search (cosine via L2-normalised vectors)."""
        assert _FAISS_AVAILABLE and self._embeddings is not None

        if self._index_dirty or self._index is None:
            self._build_faiss_index()

        norm = np.linalg.norm(query)
        q_norm = (query / (norm + 1e-10)).reshape(1, -1).astype(np.float32)
        actual_k = min(k, len(self._id_list))
        distances, indices = self._index.search(q_norm, actual_k)  # type: ignore[union-attr]
        return [
            (int(idx), float(dist))
            for idx, dist in zip(indices[0], distances[0])
            if idx >= 0
        ]

    def _build_faiss_index(self) -> None:
        """Build / rebuild the FAISS IndexFlatIP from the current embeddings."""
        assert _FAISS_AVAILABLE and self._embeddings is not None

        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = (self._embeddings / norms).astype(np.float32)

        dim = normed.shape[1]
        index = faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]
        index.add(normed)
        self._index = index
        self._index_dirty = False
        logger.debug("Built FAISS index: %d vectors, dim=%d", len(normed), dim)

    # ------------------------------------------------------------------
    # Private — eviction
    # ------------------------------------------------------------------

    def _update_existing(self, entry: CacheEntry) -> None:
        """In-place update of an entry that already exists (caller holds lock)."""
        self._entries[entry.id] = entry
        idx = self._id_list.index(entry.id)
        if self._embeddings is not None:
            self._embeddings[idx] = np.array(entry.embedding, dtype=np.float32)
        self._index_dirty = True

    def _evict_lru(self, count: int = 1) -> None:
        """Evict *count* least-recently-used entries (caller holds lock)."""
        if not self._entries:
            return
        sorted_ids = sorted(
            self._entries.keys(),
            key=lambda eid: self._entries[eid].last_accessed,
        )
        for eid in sorted_ids[:count]:
            self.remove(eid)
