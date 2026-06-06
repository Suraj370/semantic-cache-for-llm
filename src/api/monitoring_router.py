"""Monitoring REST endpoints.

GET  /v1/monitoring/summary      — all metrics as JSON (no Prometheus needed)
GET  /v1/monitoring/near-misses  — recent near-misses + analysis
GET  /v1/monitoring/cost         — model pricing table
"""

from __future__ import annotations

from fastapi import APIRouter, Query

import monitoring.metrics as m
from monitoring.cost import pricing_table
from monitoring.near_miss import default_near_miss_tracker

monitoring_router = APIRouter(tags=["monitoring"])


@monitoring_router.get("/v1/monitoring/summary")
def metrics_summary() -> dict:
    """Full metrics snapshot: hit rate, latency percentiles, cost, near-misses.

    Equivalent to scraping the Prometheus ``/metrics`` endpoint but returns
    structured JSON.  Useful for dashboards that don't use Prometheus.
    """
    return m.summary()


@monitoring_router.get("/v1/monitoring/near-misses")
def near_misses(
    limit: int = Query(default=20, ge=1, le=200, description="Max recent records to return"),
    analyze: bool = Query(default=True, description="Include full analysis (patterns, advice)"),
) -> dict:
    """Queries that fell just below the similarity threshold.

    Use this to tune thresholds and identify normalisation opportunities:
    queries that appear repeatedly here would become cache hits with a
    slightly lower threshold or better pre-processing.
    """
    result: dict = {"recent": default_near_miss_tracker.recent(limit=limit)}
    if analyze:
        result["analysis"] = default_near_miss_tracker.analyze()
    return result


@monitoring_router.get("/v1/monitoring/cost")
def cost_info() -> dict:
    """Model pricing table used for cost-savings estimation."""
    return {"pricing_usd_per_1k_tokens": pricing_table()}
