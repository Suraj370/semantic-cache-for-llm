"""Adaptive per-request-type similarity threshold learning.

Each :class:`~src.request_type.RequestType` starts at its default threshold.
As feedback arrives (was the cached answer actually helpful?) the threshold
is adjusted so that the fraction of good hits stays at or above
``target_precision``.

Algorithm
---------
After every feedback record the manager re-evaluates the optimal threshold
for that request type using a precision-maximising sweep over recent history:

1. Sort all (similarity, was_good) records by similarity descending.
2. Walk the sorted list; track running precision as entries are included.
3. The optimal threshold is the LOWEST similarity value at which the running
   cumulative precision still meets ``target_precision``.  This maximises
   recall (cache hits) subject to a quality floor.
4. The new threshold is blended toward the optimal via an EMA:
       new = (1 - alpha) * current + alpha * optimal
   capped to move at most ``max_step`` per update.

Thread safety: all mutable state is protected by a reentrant lock.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from .request_type import RequestType


class AdaptiveThresholdManager:
    """Learn per-:class:`~src.request_type.RequestType` similarity thresholds.

    Args:
        target_precision: Minimum fraction of hits that should be "good".
            Default 0.90 (90 % of served cache hits must be helpful).
        min_feedback: Minimum feedback records required before adjusting
            a threshold.  Prevents over-fitting on tiny samples.
        max_history: Maximum feedback records kept per request type.
            Older records are discarded (sliding window).
        smoothing: EMA weight applied to the newly computed optimal
            threshold.  0.3 = blend 30 % toward new value per update.
        max_step: Maximum absolute change in threshold per update.
        floor: Global lower bound — no threshold will drop below this.
    """

    def __init__(
        self,
        target_precision: float = 0.90,
        min_feedback: int = 10,
        max_history: int = 200,
        smoothing: float = 0.3,
        max_step: float = 0.05,
        floor: float = 0.50,
    ) -> None:
        self._target_precision = target_precision
        self._min_feedback = min_feedback
        self._smoothing = smoothing
        self._max_step = max_step
        self._floor = floor
        self._lock = threading.RLock()

        self._thresholds: Dict[RequestType, float] = {
            rt: rt.default_threshold for rt in RequestType
        }
        self._history: Dict[RequestType, Deque[Tuple[float, bool]]] = {
            rt: deque(maxlen=max_history) for rt in RequestType
        }
        self._feedback_counts: Dict[RequestType, int] = {
            rt: 0 for rt in RequestType
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_threshold(self, request_type: RequestType, global_floor: float = 0.0) -> float:
        """Return the current threshold for *request_type*.

        The result is at least ``max(self._floor, global_floor)`` so
        the global env-var floor is always respected.
        """
        with self._lock:
            return max(self._thresholds[request_type], self._floor, global_floor)

    def record_feedback(
        self,
        request_type: RequestType,
        similarity: float,
        was_good: bool,
    ) -> float:
        """Record one feedback signal and return the updated threshold.

        Args:
            request_type: The request type of the cache hit that was judged.
            similarity: The cosine similarity score of the hit (0–1).
            was_good: ``True`` if the cached response was helpful/correct.

        Returns:
            The new threshold for *request_type* after this update.
        """
        with self._lock:
            self._history[request_type].append((similarity, was_good))
            self._feedback_counts[request_type] += 1
            self._recompute(request_type)
            return self._thresholds[request_type]

    def get_stats(self) -> Dict[str, Any]:
        """Snapshot of all per-type thresholds and feedback counts."""
        with self._lock:
            return {
                rt.value: {
                    "current_threshold": self._thresholds[rt],
                    "default_threshold": rt.default_threshold,
                    "drift": round(self._thresholds[rt] - rt.default_threshold, 4),
                    "feedback_count": self._feedback_counts[rt],
                    "good_hits": sum(1 for _, g in self._history[rt] if g),
                    "bad_hits":  sum(1 for _, g in self._history[rt] if not g),
                }
                for rt in RequestType
            }

    def reset(self, request_type: Optional[RequestType] = None) -> None:
        """Reset thresholds (and history) to defaults.

        If *request_type* is given, reset only that type; otherwise reset all.
        """
        with self._lock:
            types = [request_type] if request_type else list(RequestType)
            for rt in types:
                self._thresholds[rt] = rt.default_threshold
                self._history[rt].clear()
                self._feedback_counts[rt] = 0

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _recompute(self, rt: RequestType) -> None:
        """Re-derive the optimal threshold and EMA-blend toward it."""
        data = list(self._history[rt])
        if len(data) < self._min_feedback:
            return

        optimal = _find_optimal_threshold(data, self._target_precision)
        if optimal is None:
            return

        current = self._thresholds[rt]
        raw_delta = optimal - current
        # Clamp to max_step per update, then smooth via EMA
        clamped_delta = max(-self._max_step, min(self._max_step, raw_delta))
        new_threshold = current + self._smoothing * clamped_delta
        self._thresholds[rt] = max(self._floor, min(0.99, round(new_threshold, 4)))


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def _find_optimal_threshold(
    data: List[Tuple[float, bool]],
    target_precision: float,
) -> Optional[float]:
    """Return the lowest similarity threshold achieving ``target_precision``.

    Scans sorted data from highest to lowest similarity, tracking running
    precision.  Records the lowest similarity value at which cumulative
    precision was still >= target.  This maximises recall (more cache hits)
    subject to the quality floor.

    Returns ``None`` when the target cannot be met at any threshold (e.g.
    all feedback is negative).
    """
    sorted_desc = sorted(data, key=lambda x: x[0], reverse=True)
    good = 0
    optimal: Optional[float] = None

    for i, (sim, is_good) in enumerate(sorted_desc):
        good += int(is_good)
        if good / (i + 1) >= target_precision:
            optimal = sim   # keep updating — we want the LOWEST qualifying sim

    return optimal


# Module-level singleton shared across the application.
default_adaptive_threshold_manager = AdaptiveThresholdManager()
