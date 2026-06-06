"""Similarity threshold tuner.

POST /v1/cache/tune-threshold

Uses a leave-one-out simulation over the current cache contents: each stored
entry is treated as an incoming query and we record what would have happened
at each candidate threshold.

Metric definitions
------------------
hit_rate          — fraction of queries that would be served from cache.
miss_rate         — fraction that would call the LLM  (1 - hit_rate).
borderline_hit_rate — hits whose best-match similarity falls within
                    ``borderline_margin`` above the threshold.  These are
                    the "slightly wrong answer" risk zone: the cache would
                    serve a response, but the semantic match is marginal.
confident_hit_rate  — hits whose similarity clearly exceeds the threshold
                    (hit_rate - borderline_hit_rate).

The recommended threshold is the one that maximises confident_hit_rate.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..models import CacheEntry
from .dependencies import VectorStoreDep

logger = logging.getLogger(__name__)

tuner_router = APIRouter(tags=["threshold-tuner"])

_DEFAULT_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class TuneRequest(BaseModel):
    thresholds: List[float] = Field(
        default=_DEFAULT_THRESHOLDS,
        description="Thresholds to simulate. Values must be in (0, 1].",
    )
    borderline_margin: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description=(
            "Hits with similarity in [threshold, threshold + margin) are "
            "flagged as borderline — high risk of a subtly wrong answer."
        ),
    )
    max_sample: int = Field(
        default=1000,
        ge=2,
        le=5000,
        description=(
            "Maximum entries to include. When the store is larger, the most "
            "recently accessed entries are sampled (hottest cache lines)."
        ),
    )


class ThresholdResult(BaseModel):
    threshold: float
    hits: int
    misses: int
    borderline_hits: int
    confident_hits: int
    hit_rate: float
    miss_rate: float
    borderline_hit_rate: float
    confident_hit_rate: float


class Recommendation(BaseModel):
    threshold: float
    hit_rate: float
    confident_hit_rate: float
    borderline_hit_rate: float
    reason: str


class TuneResponse(BaseModel):
    sample_size: int
    context_groups: int
    results: List[ThresholdResult]
    recommendation: Optional[Recommendation] = None


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@tuner_router.post("/v1/cache/tune-threshold", response_model=TuneResponse)
def tune_threshold(body: TuneRequest, store: VectorStoreDep) -> TuneResponse:
    """Simulate hit/miss behaviour at multiple similarity thresholds.

    Only non-expired entries with valid embeddings are included.  Entries are
    grouped by ``context_key`` so comparisons are restricted to entries that
    would actually compete at search time.
    """
    entries = [
        e for e in store
        if not e.is_expired() and e.embedding
    ]

    if len(entries) > body.max_sample:
        entries.sort(key=lambda e: e.last_accessed, reverse=True)
        entries = entries[: body.max_sample]

    if len(entries) < 2:
        return TuneResponse(
            sample_size=len(entries),
            context_groups=0,
            results=[],
            recommendation=None,
        )

    # ------------------------------------------------------------------
    # Group by context_key: entries from different groups can never match
    # ------------------------------------------------------------------
    groups: Dict[Optional[str], List[CacheEntry]] = {}
    for e in entries:
        groups.setdefault(e.context_key, []).append(e)

    # For each entry, compute the highest cosine similarity to any *other*
    # entry in the same context group.  Single-entry groups always get 0.0.
    all_best: List[float] = []

    for group in groups.values():
        if len(group) == 1:
            all_best.append(0.0)
            continue

        mat = np.array([e.embedding for e in group], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normed = mat / norms                    # (n, d) unit vectors

        sim = normed @ normed.T                 # (n, n) cosine similarity
        np.fill_diagonal(sim, -np.inf)          # exclude self-similarity

        all_best.extend(float(sim[i].max()) for i in range(len(group)))

    n = len(all_best)
    best = np.array(all_best, dtype=np.float32)

    # ------------------------------------------------------------------
    # Evaluate each threshold
    # ------------------------------------------------------------------
    results: List[ThresholdResult] = []
    for t in sorted(body.thresholds):
        t = round(t, 6)
        hit_mask = best >= t
        borderline_mask = hit_mask & (best < t + body.borderline_margin)

        hits = int(hit_mask.sum())
        borderline = int(borderline_mask.sum())
        confident = hits - borderline
        misses = n - hits

        results.append(ThresholdResult(
            threshold=t,
            hits=hits,
            misses=misses,
            borderline_hits=borderline,
            confident_hits=confident,
            hit_rate=round(hits / n, 4),
            miss_rate=round(misses / n, 4),
            borderline_hit_rate=round(borderline / n, 4),
            confident_hit_rate=round(confident / n, 4),
        ))

    return TuneResponse(
        sample_size=n,
        context_groups=len(groups),
        results=results,
        recommendation=_recommend(results),
    )


# ---------------------------------------------------------------------------
# Recommendation heuristic
# ---------------------------------------------------------------------------


def _recommend(results: List[ThresholdResult]) -> Optional[Recommendation]:
    """Pick the threshold with the best confident_hit_rate.

    Tiebreak: lower threshold wins (more hits at the same quality).
    Returns None when there are no hits at any threshold.
    """
    candidates = [r for r in results if r.confident_hits > 0]
    if not candidates:
        return None

    best = max(candidates, key=lambda r: (r.confident_hit_rate, -r.threshold))

    reason = (
        f"At threshold {best.threshold:.2f}: "
        f"{best.hit_rate:.0%} hit rate — "
        f"{best.borderline_hit_rate:.0%} borderline (risky) and "
        f"{best.confident_hit_rate:.0%} confident hits. "
        f"Raises threshold until risky hits drop below the borderline margin."
    )
    return Recommendation(
        threshold=best.threshold,
        hit_rate=best.hit_rate,
        confident_hit_rate=best.confident_hit_rate,
        borderline_hit_rate=best.borderline_hit_rate,
        reason=reason,
    )
