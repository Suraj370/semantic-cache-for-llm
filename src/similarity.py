"""Cosine similarity utilities with a batched NumPy implementation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors.

    Returns 0.0 when either vector is the zero vector (undefined case).
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(
        np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b)
    )


def batch_cosine_similarity(
    query: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between one query and many candidates.

    Uses a single BLAS matrix-vector multiply — significantly faster than
    looping over :func:`cosine_similarity`.

    Args:
        query: Shape ``(d,)``.
        candidates: Shape ``(n, d)``.

    Returns:
        Float32 array of shape ``(n,)``.  Zero-norm candidates get 0.0.

    Raises:
        ValueError: If dimensions are incompatible.
    """
    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)

    if query.shape[0] != candidates.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query has {query.shape[0]} dims, "
            f"candidates have {candidates.shape[1]} dims."
        )

    query_f64 = query.astype(np.float64)
    cands_f64 = candidates.astype(np.float64)

    query_norm = float(np.linalg.norm(query_f64))
    if query_norm == 0.0:
        return np.zeros(len(candidates), dtype=np.float32)

    query_unit = query_f64 / query_norm

    candidate_norms = np.linalg.norm(cands_f64, axis=1)
    zero_mask = candidate_norms == 0.0
    safe_norms = np.where(zero_mask, 1.0, candidate_norms)
    candidates_unit = cands_f64 / safe_norms[:, np.newaxis]

    similarities = candidates_unit @ query_unit
    similarities[zero_mask] = 0.0

    return similarities.astype(np.float32)


def find_most_similar(
    query: np.ndarray,
    candidates: np.ndarray,
    threshold: float = 0.85,
    k: int = 1,
) -> List[Tuple[int, float]]:
    """Return the top-k candidate indices above threshold, sorted descending.

    Args:
        query: Shape ``(d,)``.
        candidates: Shape ``(n, d)``.
        threshold: Minimum cosine similarity to include.
        k: Maximum number of results.

    Returns:
        List of ``(index, similarity)`` tuples.
    """
    if len(candidates) == 0:
        return []

    similarities = batch_cosine_similarity(query, candidates)
    sorted_indices = np.argsort(similarities)[::-1]

    results: List[Tuple[int, float]] = []
    for idx in sorted_indices:
        sim = float(similarities[idx])
        if sim < threshold:
            break
        results.append((int(idx), sim))
        if len(results) >= k:
            break

    return results
