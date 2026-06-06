"""Feedback and adaptive-threshold endpoints.

POST /v1/cache/feedback          — submit a good/bad signal on a cache hit
GET  /v1/cache/thresholds        — view current per-type thresholds + drift
POST /v1/cache/thresholds/reset  — reset one or all types to defaults
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..adaptive_threshold import default_adaptive_threshold_manager
from ..request_type import RequestType

logger = logging.getLogger(__name__)

feedback_router = APIRouter(tags=["adaptive-thresholds"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    """Feedback on a single cache hit.

    Obtain ``entry_id``, ``similarity``, and ``request_type`` from the
    response headers of a cache HIT:

        X-Cache: HIT
        X-Cache-Entry-Id: <uuid>
        X-Cache-Similarity: 0.9134
        X-Cache-Request-Type: factual
    """

    entry_id: str = Field(description="ID of the cache entry that was served.")
    similarity: float = Field(ge=0.0, le=1.0, description="Similarity score of the hit.")
    request_type: RequestType = Field(description="Request type reported in X-Cache-Request-Type.")
    was_good: bool = Field(description="True if the cached response was helpful/correct.")


class FeedbackResponse(BaseModel):
    request_type: str
    was_good: bool
    similarity: float
    updated_threshold: float
    feedback_count: int


class ThresholdEntry(BaseModel):
    request_type: str
    current_threshold: float
    default_threshold: float
    drift: float
    feedback_count: int
    good_hits: int
    bad_hits: int


class ThresholdsResponse(BaseModel):
    thresholds: list[ThresholdEntry]


class ResetRequest(BaseModel):
    request_type: Optional[RequestType] = Field(
        default=None,
        description="Request type to reset.  Omit to reset all types.",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@feedback_router.post("/v1/cache/feedback", response_model=FeedbackResponse)
def submit_feedback(body: FeedbackRequest) -> FeedbackResponse:
    """Record whether a cache hit was helpful.

    The adaptive threshold manager uses this signal to raise or lower the
    similarity threshold for the given request type so that future hits
    maintain the configured quality floor (default: 90 % good hits).

    **How to use:** when you receive a cache HIT, check whether the answer
    was correct.  If not, POST this endpoint with ``was_good: false``.  After
    enough feedback the threshold for that type will automatically rise to
    exclude near-misses.
    """
    mgr = default_adaptive_threshold_manager
    new_threshold = mgr.record_feedback(
        request_type=body.request_type,
        similarity=body.similarity,
        was_good=body.was_good,
    )
    stats = mgr.get_stats()[body.request_type.value]
    logger.info(
        "Feedback recorded  type=%s  sim=%.4f  good=%s  threshold=%.4f",
        body.request_type.value,
        body.similarity,
        body.was_good,
        new_threshold,
    )
    return FeedbackResponse(
        request_type=body.request_type.value,
        was_good=body.was_good,
        similarity=body.similarity,
        updated_threshold=new_threshold,
        feedback_count=stats["feedback_count"],
    )


@feedback_router.get("/v1/cache/thresholds", response_model=ThresholdsResponse)
def get_thresholds() -> ThresholdsResponse:
    """Return current adaptive thresholds for all request types.

    ``drift`` is the difference between the learned threshold and the
    factory default.  Positive drift means the system raised the bar due to
    bad-hit feedback; negative drift means it relaxed after consistent
    good hits.
    """
    stats = default_adaptive_threshold_manager.get_stats()
    return ThresholdsResponse(
        thresholds=[
            ThresholdEntry(request_type=rt, **data)
            for rt, data in stats.items()
        ]
    )


@feedback_router.post("/v1/cache/thresholds/reset")
def reset_thresholds(body: ResetRequest) -> dict:
    """Reset learned thresholds (and feedback history) to factory defaults.

    Pass ``request_type`` to reset a single type; omit it to reset all.
    """
    default_adaptive_threshold_manager.reset(body.request_type)
    scope = body.request_type.value if body.request_type else "all"
    return {"reset": scope}
