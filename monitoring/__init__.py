"""Monitoring package: Prometheus metrics, cost estimation, and near-miss analysis."""

from .cost import estimate_savings, estimate_tokens_saved, pricing_table
from .metrics import (
    record_evictions,
    record_hit,
    record_miss,
    record_near_miss,
    summary,
    update_cache_size,
)
from .near_miss import NearMiss, NearMissTracker, default_near_miss_tracker

__all__ = [
    # Metrics
    "record_hit",
    "record_miss",
    "record_near_miss",
    "record_evictions",
    "update_cache_size",
    "summary",
    # Cost
    "estimate_savings",
    "estimate_tokens_saved",
    "pricing_table",
    # Near-miss
    "NearMiss",
    "NearMissTracker",
    "default_near_miss_tracker",
]
