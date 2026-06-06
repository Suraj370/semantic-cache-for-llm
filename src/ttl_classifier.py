"""Rule-based TTL tier classifier for semantic cache entries.

Assigns a :class:`TtlTier` to an incoming prompt based on how quickly the
best answer is likely to become stale.  Classification is purely regex-based
— no model calls, no I/O, negligible latency.

Tiers (checked in priority order):
    NO_CACHE  — real-time / live-data queries; bypass the cache entirely.
    SHORT     — time-sensitive queries (today, latest, news …); TTL = 1 h.
    PERMANENT — math, code, definitions, historical facts; TTL = None.
    LONG      — general factual questions; TTL = 24 h (default fallback).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional

# ---------------------------------------------------------------------------
# Tier definition
# ---------------------------------------------------------------------------


class TtlTier(str, Enum):
    NO_CACHE  = "no_cache"   # skip caching entirely
    SHORT     = "short"      # 1 h  = 3 600 s
    LONG      = "long"       # 24 h = 86 400 s  (default)
    PERMANENT = "permanent"  # never expires (ttl=None)

    @property
    def seconds(self) -> Optional[int]:
        """Return the TTL in seconds, or ``None`` for PERMANENT.

        Callers should treat :attr:`NO_CACHE` as a signal to skip writing
        to the cache; the return value (``0``) is not stored in an entry.
        """
        _MAP = {
            TtlTier.NO_CACHE:  0,
            TtlTier.SHORT:     3_600,
            TtlTier.LONG:      86_400,
            TtlTier.PERMANENT: None,
        }
        return _MAP[self]


# ---------------------------------------------------------------------------
# Pattern lists
# ---------------------------------------------------------------------------

def _c(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.I)


# Tier 1: real-time / live data — bypass the cache entirely
_NO_CACHE: List[re.Pattern] = [
    _c(r"\bright\s+now\b"),
    _c(r"\bat\s+(this\s+)?moment\b"),
    _c(r"\breal[- ]?time\b"),
    _c(r"\blive\s+(price|score|update|feed|stream|data)\b"),
    # Financial live quotes
    _c(r"\b(stock|crypto|bitcoin|btc|ethereum|eth|forex|exchange)\s+(price|rate|value|quote)\b"),
    _c(r"\b(market\s+cap|spot\s+price|bid\s+price|ask\s+price)\b"),
    # Weather — current conditions only (not "how does weather form")
    _c(r"\bwhat('?s| is)\s+(the\s+)?(weather|temperature|forecast)\b"),
    _c(r"\bweather\s+(forecast|report|today|right\s+now|currently)\b"),
    _c(r"\b(current\s+)?(weather|temperature)\s+(in|at|for|today|right\s+now)\b"),
    # Clock / calendar
    _c(r"\bwhat\s+(time|date|day)\s+is\s+it\b"),
    _c(r"\bwhat('?s| is)\s+(today'?s?\s+)?(date|time|day)\b"),
    # Live scores / matches
    _c(r"\b(live|current|ongoing|in[- ]progress)\s+(score|game|match|result)\b"),
    # Service / system status
    _c(r"\bis\s+\S+\s+(down|offline|unavailable)\b"),
    _c(r"\b(server|service|site|website|api|system)\s+status\b"),
    _c(r"\bstatus\s+(page|check)\b"),
    # Breaking news
    _c(r"\bbreaking\s+news\b"),
]

# Tier 2: time-sensitive but not necessarily second-by-second
_SHORT: List[re.Pattern] = [
    # Today / this period
    _c(r"\btoday\b"),
    _c(r"\bthis\s+(morning|afternoon|evening|night|week)\b"),
    _c(r"\btonight\b"),
    # Latest / just-released
    _c(r"\b(latest|most\s+recent|newest)\b"),
    _c(r"\bjust\s+(released|announced|launched|published|dropped)\b"),
    # News and trending
    _c(r"\bnews\b"),
    _c(r"\bheadlines?\b"),
    _c(r"\btrending\b"),
    _c(r"\bviral\b"),
    # Temporal adverbs that imply recency
    _c(r"\bcurrently\b"),
    _c(r"\bnowadays\b"),
    _c(r"\bthese\s+days\b"),
    # Current events
    _c(r"\bcurrent\s+events?\b"),
    _c(r"\bwhat('?s| is)\s+(happening|going\s+on)\b"),
    # Recent year references (suggest recent/changing content)
    _c(r"\bin\s+20[2-9]\d\b"),
]

# Tier 3: mathematically / logically / historically fixed
_PERMANENT: List[re.Pattern] = [
    # Mathematics
    _c(r"\b(calculate|compute|solve|evaluate|simplify|integrate|differentiate)\b"),
    _c(r"\b(equation|formula|theorem|proof|lemma|corollary|axiom)\b"),
    _c(r"\b(integral|derivative|gradient|matrix|determinant|eigenvalue)\b"),
    # CS theory / algorithms
    _c(r"\b(algorithm|complexity|big[- ]o notation|time\s+complexity|space\s+complexity)\b"),
    _c(r"\b(sorting|searching|tree\s+traversal|dynamic\s+programming|memoization)\b"),
    # Code generation / transformation
    _c(r"\b(write|implement|create|build|code|generate)\s+(a\s+|an\s+)?(function|class|method|program|script|snippet|module)\b"),
    _c(r"\b(refactor|optimise|optimize|debug|fix)\s+(this|the|my)\s+(code|function|class|script)\b"),
    # Definitions / etymology
    _c(r"\b(define|definition\s+of|meaning\s+of|etymology\s+of)\b"),
    # Historical facts with fixed outcomes
    _c(r"\b(who|when)\s+(invented|discovered|founded|created|designed)\b"),
    _c(r"\bhistory\s+of\b"),
    _c(r"\bin\s+the\s+\d{1,2}(th|st|nd|rd)\s+century\b"),
    _c(r"\bborn\s+in\s+\d{4}\b"),
    # Language / grammar
    _c(r"\b(translate|grammar|spelling|pronunciation|conjugat)\b"),
    # Versioned / immutable specs
    _c(r"\b(iso|ieee|rfc|ecma)\s+\d+\b"),
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TtlClassifier:
    """Classify a prompt string into a :class:`TtlTier`.

    Usage::

        classifier = TtlClassifier()
        tier = classifier.classify("what's the weather today?")
        # → TtlTier.NO_CACHE

        ttl = tier.seconds   # → 0  (caller interprets as bypass)
    """

    def classify(self, prompt: str) -> TtlTier:
        """Return the :class:`TtlTier` for *prompt*.

        Pattern sets are checked in priority order so that the most
        restrictive rule wins.  An empty prompt defaults to LONG.
        """
        if not prompt or not prompt.strip():
            return TtlTier.LONG

        text = prompt.strip()

        if _matches_any(text, _NO_CACHE):
            return TtlTier.NO_CACHE
        if _matches_any(text, _SHORT):
            return TtlTier.SHORT
        if _matches_any(text, _PERMANENT):
            return TtlTier.PERMANENT
        return TtlTier.LONG


def _matches_any(text: str, patterns: List[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


# Module-level singleton — import and use directly in other modules.
default_classifier = TtlClassifier()
