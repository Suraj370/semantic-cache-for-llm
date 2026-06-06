"""Near-miss tracker: queries that fell just below the similarity threshold.

A near-miss is a cache lookup that returned a candidate with similarity in
``[threshold - margin, threshold)``.  Tracking them reveals:

* Which queries are repeatedly *almost* hitting the cache — normalisation or
  threshold tuning could promote them to full hits.
* Whether the current threshold is too conservative (many near-misses with
  high similarity).
* Query patterns that would benefit from synonym expansion or filler-word
  stripping before embedding.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class NearMiss:
    """A single near-miss record."""

    query: str
    normalized_query: str
    similarity: float
    threshold: float            # adaptive threshold that was active
    request_type: str
    context_key: Optional[str]
    timestamp: float = field(default_factory=time.time)

    @property
    def gap(self) -> float:
        """Distance below the threshold (positive = below threshold)."""
        return round(self.threshold - self.similarity, 4)


class NearMissTracker:
    """Ring buffer of recent near-misses with analysis helpers.

    Args:
        capacity: Maximum number of records to keep.  Oldest are dropped
            when the buffer is full (LRU ring).
        margin: Half-open band ``[threshold - margin, threshold)`` that
            qualifies as a near-miss.  Queries further below the threshold
            are genuine misses and not tracked.
    """

    def __init__(self, capacity: int = 500, margin: float = 0.10) -> None:
        self.margin = margin
        self._buffer: deque[NearMiss] = deque(maxlen=capacity)
        self._total_seen = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, near_miss: NearMiss) -> None:
        with self._lock:
            self._buffer.append(near_miss)
            self._total_seen += 1

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self, top_n: int = 20) -> Dict[str, Any]:
        """Return a structured analysis of buffered near-misses.

        Keys
        ----
        total_recorded     — entries in the current buffer.
        total_seen         — all near-misses since startup (including evicted).
        by_request_type    — per-type counts and average similarity.
        top_queries        — most frequently near-missed queries.
        similarity_stats   — distribution of near-miss similarity scores.
        threshold_advice   — suggested threshold reduction (or None).
        """
        with self._lock:
            data: List[NearMiss] = list(self._buffer)
            total_seen = self._total_seen

        if not data:
            return {
                "total_recorded": 0,
                "total_seen": total_seen,
                "by_request_type": {},
                "top_queries": [],
                "similarity_stats": {},
                "threshold_advice": None,
            }

        # Per-type breakdown
        by_type: Dict[str, List[float]] = defaultdict(list)
        for nm in data:
            by_type[nm.request_type].append(nm.similarity)

        # Top near-missed normalised queries
        query_counter: Counter = Counter(nm.normalized_query for nm in data)
        top_queries = []
        for nq, count in query_counter.most_common(top_n):
            matching = [nm for nm in data if nm.normalized_query == nq]
            top_queries.append({
                "normalized_query": nq,
                "count": count,
                "avg_similarity": round(float(np.mean([m.similarity for m in matching])), 4),
                "avg_gap": round(float(np.mean([m.gap for m in matching])), 4),
                "request_types": list({m.request_type for m in matching}),
            })

        # Overall similarity distribution
        sims = np.array([nm.similarity for nm in data], dtype=np.float64)
        sim_stats = {
            "count": len(sims),
            "mean": round(float(np.mean(sims)), 4),
            "p50": round(float(np.percentile(sims, 50)), 4),
            "p90": round(float(np.percentile(sims, 90)), 4),
            "min": round(float(np.min(sims)), 4),
            "max": round(float(np.max(sims)), 4),
        }

        # Threshold advice: if >= 30 % of near-misses have gap <= 0.02,
        # a small threshold reduction would convert them to hits.
        close_count = int(np.sum(sims >= (sims.max() - 0.02)))
        pct_close = close_count / len(sims) if sims.size > 0 else 0.0
        threshold_advice = None
        if pct_close >= 0.30:
            # Group by request type for targeted advice
            by_type_advice = {}
            for rt, rt_sims in by_type.items():
                rt_arr = np.array(rt_sims, dtype=np.float64)
                rt_close = int(np.sum(rt_arr >= rt_arr.max() - 0.02))
                if rt_close / len(rt_arr) >= 0.30:
                    suggested = round(float(rt_arr.max() - 0.005), 4)
                    by_type_advice[rt] = {
                        "close_miss_pct": round(rt_close / len(rt_arr), 2),
                        "suggested_threshold": suggested,
                    }
            threshold_advice = {
                "message": (
                    f"{pct_close:.0%} of near-misses are within 0.02 of the "
                    "threshold. A small threshold reduction may improve hit rate."
                ),
                "by_request_type": by_type_advice,
            }

        return {
            "total_recorded": len(data),
            "total_seen": total_seen,
            "by_request_type": {
                rt: {
                    "count": len(ss),
                    "avg_similarity": round(float(np.mean(ss)), 4),
                    "p90_similarity": round(float(np.percentile(ss, 90)), 4),
                }
                for rt, ss in by_type.items()
            },
            "top_queries": top_queries,
            "similarity_stats": sim_stats,
            "threshold_advice": threshold_advice,
        }

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the *limit* most recent near-misses as plain dicts."""
        with self._lock:
            data = list(self._buffer)[-limit:]
        return [
            {
                "query": nm.query[:200],
                "similarity": nm.similarity,
                "threshold": nm.threshold,
                "gap": nm.gap,
                "request_type": nm.request_type,
                "timestamp": nm.timestamp,
            }
            for nm in reversed(data)
        ]


# Module-level singleton
default_near_miss_tracker = NearMissTracker(capacity=500, margin=0.10)
