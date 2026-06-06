"""Prometheus metrics + in-memory summary for the semantic cache.

All Prometheus objects are module-level singletons registered in the default
registry.  Call the module-level helpers (``record_hit``, ``record_miss``,
``record_near_miss``, ``update_cache_size``) from request handlers.

In-memory summary
-----------------
``summary()`` returns a plain-dict snapshot suitable for the JSON monitoring
API without requiring a Prometheus scrape.  It keeps per-model and
per-request-type breakdowns in addition to the global totals.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Prometheus metric objects (module-level singletons)
# ---------------------------------------------------------------------------

HITS = Counter(
    "semantic_cache_hits_total",
    "Total cache hits",
    ["model", "request_type"],
)
MISSES = Counter(
    "semantic_cache_misses_total",
    "Total cache misses",
    ["model", "request_type"],
)
TOKENS_SAVED = Counter(
    "semantic_cache_tokens_saved_total",
    "Estimated LLM tokens saved by cache hits",
    ["model"],
)
COST_SAVED = Counter(
    "semantic_cache_cost_saved_usd_total",
    "Estimated USD saved by cache hits",
    ["model"],
)
NEAR_MISSES = Counter(
    "semantic_cache_near_misses_total",
    "Queries that fell just below the similarity threshold",
    ["request_type"],
)
EVICTIONS = Counter(
    "semantic_cache_evictions_total",
    "LRU evictions from the vector store",
)

CACHE_SIZE = Gauge(
    "semantic_cache_size",
    "Current number of entries in the vector store",
)

# Latency in seconds; separate result label keeps hit/miss latencies comparable
REQUEST_LATENCY = Histogram(
    "semantic_cache_request_latency_seconds",
    "End-to-end request latency",
    ["result", "model"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

SIMILARITY_SCORE = Histogram(
    "semantic_cache_similarity_score",
    "Cosine similarity of cache hits",
    ["request_type"],
    buckets=[0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 1.0],
)

NEAR_MISS_SIMILARITY = Histogram(
    "semantic_cache_near_miss_similarity",
    "Cosine similarity of near-miss candidates (below threshold)",
    ["request_type"],
    buckets=[0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95],
)


# ---------------------------------------------------------------------------
# In-memory summary (for the JSON monitoring API)
# ---------------------------------------------------------------------------


class _Stats:
    """Fast in-memory accumulator — no Prometheus dependency for the JSON API."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._near_misses = 0
        self._evictions = 0
        self._hit_latencies_ms: List[float] = []
        self._miss_latencies_ms: List[float] = []
        self._similarities: List[float] = []
        self._near_miss_similarities: List[float] = []
        self._tokens_saved = 0
        self._cost_saved_usd = 0.0
        self._by_model: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"hits": 0, "misses": 0}
        )
        self._by_request_type: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"hits": 0, "misses": 0, "near_misses": 0}
        )
        self._start = time.time()

    def record_hit(
        self,
        model: str,
        request_type: str,
        latency_ms: float,
        similarity: float,
        tokens_saved: int,
        cost_saved_usd: float,
    ) -> None:
        with self._lock:
            self._hits += 1
            self._hit_latencies_ms.append(latency_ms)
            self._similarities.append(similarity)
            self._tokens_saved += tokens_saved
            self._cost_saved_usd += cost_saved_usd
            self._by_model[model]["hits"] += 1
            self._by_request_type[request_type]["hits"] += 1

    def record_miss(self, model: str, request_type: str, latency_ms: float) -> None:
        with self._lock:
            self._misses += 1
            self._miss_latencies_ms.append(latency_ms)
            self._by_model[model]["misses"] += 1
            self._by_request_type[request_type]["misses"] += 1

    def record_near_miss(self, request_type: str, similarity: float) -> None:
        with self._lock:
            self._near_misses += 1
            self._near_miss_similarities.append(similarity)
            self._by_request_type[request_type]["near_misses"] += 1

    def record_evictions(self, count: int) -> None:
        with self._lock:
            self._evictions += count

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            hit_lats = np.array(self._hit_latencies_ms, dtype=np.float64)
            miss_lats = np.array(self._miss_latencies_ms, dtype=np.float64)
            sims = np.array(self._similarities, dtype=np.float64)
            nm_sims = np.array(self._near_miss_similarities, dtype=np.float64)

            return {
                "hits": self._hits,
                "misses": self._misses,
                "near_misses": self._near_misses,
                "total_queries": total,
                "hit_rate": round(hit_rate, 4),
                "evictions": self._evictions,
                "tokens_saved": self._tokens_saved,
                "cost_saved_usd": round(self._cost_saved_usd, 6),
                "uptime_seconds": round(time.time() - self._start, 1),
                "latency": {
                    "hit":  _percentiles(hit_lats),
                    "miss": _percentiles(miss_lats),
                },
                "similarity": {
                    "hits":       _sim_percentiles(sims),
                    "near_misses": _sim_percentiles(nm_sims),
                },
                "by_model": dict(self._by_model),
                "by_request_type": dict(self._by_request_type),
            }


def _percentiles(arr: np.ndarray) -> Dict[str, float]:
    if len(arr) == 0:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "count": int(len(arr)),
        "mean":  round(float(np.mean(arr)), 2),
        "p50":   round(float(np.percentile(arr, 50)), 2),
        "p95":   round(float(np.percentile(arr, 95)), 2),
        "p99":   round(float(np.percentile(arr, 99)), 2),
    }


def _sim_percentiles(arr: np.ndarray) -> Dict[str, float]:
    if len(arr) == 0:
        return {"count": 0, "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "count": int(len(arr)),
        "mean":  round(float(np.mean(arr)), 4),
        "p10":   round(float(np.percentile(arr, 10)), 4),
        "p50":   round(float(np.percentile(arr, 50)), 4),
        "p90":   round(float(np.percentile(arr, 90)), 4),
    }


# Module-level singleton
_stats = _Stats()


# ---------------------------------------------------------------------------
# Public helpers — call these from request handlers
# ---------------------------------------------------------------------------


def record_hit(
    model: str,
    request_type: str,
    latency_s: float,
    similarity: float,
    tokens_saved: int = 0,
    cost_saved_usd: float = 0.0,
) -> None:
    """Instrument a cache HIT."""
    HITS.labels(model=model, request_type=request_type).inc()
    REQUEST_LATENCY.labels(result="hit", model=model).observe(latency_s)
    SIMILARITY_SCORE.labels(request_type=request_type).observe(similarity)
    if tokens_saved:
        TOKENS_SAVED.labels(model=model).inc(tokens_saved)
    if cost_saved_usd:
        COST_SAVED.labels(model=model).inc(cost_saved_usd)
    _stats.record_hit(model, request_type, latency_s * 1000, similarity, tokens_saved, cost_saved_usd)


def record_miss(model: str, request_type: str, latency_s: float) -> None:
    """Instrument a cache MISS."""
    MISSES.labels(model=model, request_type=request_type).inc()
    REQUEST_LATENCY.labels(result="miss", model=model).observe(latency_s)
    _stats.record_miss(model, request_type, latency_s * 1000)


def record_near_miss(request_type: str, similarity: float) -> None:
    """Instrument a near-miss (below threshold but within the margin)."""
    NEAR_MISSES.labels(request_type=request_type).inc()
    NEAR_MISS_SIMILARITY.labels(request_type=request_type).observe(similarity)
    _stats.record_near_miss(request_type, similarity)


def record_evictions(count: int) -> None:
    """Instrument LRU evictions from the vector store."""
    if count > 0:
        EVICTIONS.inc(count)
        _stats.record_evictions(count)


def update_cache_size(size: int) -> None:
    """Update the cache size gauge."""
    CACHE_SIZE.set(size)


def summary() -> Dict[str, Any]:
    """Return all in-memory metrics as a plain dict (for the JSON API)."""
    return _stats.snapshot()
