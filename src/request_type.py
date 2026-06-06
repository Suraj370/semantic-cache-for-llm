"""Request-type classifier for per-type similarity thresholds.

Each request type has a distinct tolerance for approximate cache matches:

    CLASSIFICATION  — constrained answer space; 0.90 is safe.
    FACTUAL         — general knowledge; 0.85 default.
    SUMMARIZATION   — structural task; 0.88.
    EXTRACTION      — structural task; 0.90.
    CODE            — precision matters; 0.92.
    CREATIVE        — highly prompt-specific; 0.98 (near-exact match only).
    CONVERSATIONAL  — open-ended fallback; 0.85.

Thresholds are starting points.  :class:`~src.adaptive_threshold.AdaptiveThresholdManager`
adjusts them per-type as feedback accumulates.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List


class RequestType(str, Enum):
    CLASSIFICATION  = "classification"
    FACTUAL         = "factual"
    SUMMARIZATION   = "summarization"
    EXTRACTION      = "extraction"
    CODE            = "code"
    CREATIVE        = "creative"
    CONVERSATIONAL  = "conversational"   # fallback

    @property
    def default_threshold(self) -> float:
        return _DEFAULTS[self]


_DEFAULTS: dict = {}   # populated after class definition to avoid forward-ref issues


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

_DEFAULTS.update({
    RequestType.CLASSIFICATION: 0.90,
    RequestType.FACTUAL:        0.85,
    RequestType.SUMMARIZATION:  0.88,
    RequestType.EXTRACTION:     0.90,
    RequestType.CODE:           0.92,
    RequestType.CREATIVE:       0.98,
    RequestType.CONVERSATIONAL: 0.85,
})


# ---------------------------------------------------------------------------
# Classification patterns (checked in declaration order)
# ---------------------------------------------------------------------------

def _c(p: str) -> re.Pattern:
    return re.compile(p, re.I)


# Sentiment, labelling, yes/no, multiple-choice
_CLASSIFICATION: List[re.Pattern] = [
    _c(r"\b(classify|categorize|categorise|label|tag)\b"),
    _c(r"\b(sentiment|tone|emotion)\b"),
    _c(r"\bis\s+(this|it|that)\s+(a|an|the)?\b"),
    _c(r"\b(true|false)\s*(or|\/)\s*(false|true)\b"),
    _c(r"\byes\s+or\s+no\b"),
    _c(r"\bwhich\s+(category|class|group|type)\b"),
    _c(r"\bpositive|negative|neutral\b"),
    _c(r"\bmulti[- ]?label\b"),
]

# Story, poem, creative writing, ideation
_CREATIVE: List[re.Pattern] = [
    _c(r"\b(write|compose|create|generate|draft|craft)\s+(a\s+|an\s+)?(story|poem|essay|song|haiku|limerick|sonnet|narrative|script|dialogue|fiction|blog\s+post|creative)\b"),
    _c(r"\b(brainstorm|ideate|imagine|invent|think\s+of|come\s+up\s+with)\b"),
    _c(r"\b(creative|imaginative|original)\s+(ideas?|content|writing|text)\b"),
    _c(r"\bcontinue\s+(the\s+)?(story|narrative|poem)\b"),
    _c(r"\bmake\s+(it\s+)?(more\s+)?(creative|interesting|engaging|fun|poetic)\b"),
    _c(r"\bfree[- ]?form\b"),
]

# Write / fix / explain code
_CODE: List[re.Pattern] = [
    _c(r"\b(write|implement|create|build|code|generate)\s+(a\s+|an\s+)?(function|class|method|program|script|snippet|module|api|endpoint)\b"),
    _c(r"\b(debug|fix|refactor|optimise|optimize|review)\s+(this|the|my)?\s*(code|function|class|script|bug)\b"),
    _c(r"\b(what|why)\s+(does|is)\s+.{1,40}\s+code\b"),
    _c(r"\b(sql|regex|bash|shell|python|javascript|typescript|golang|rust|java)\s+(query|script|snippet|code|function)\b"),
    _c(r"\bunit\s+test(s|ing)?\b"),
    _c(r"\btime\s+complexity\b"),
    _c(r"```"),   # fenced code in the prompt → treat as code task
]

# Summarise a body of text or document
_SUMMARIZATION: List[re.Pattern] = [
    _c(r"\b(summarize|summarise|summary|summarization)\b"),
    _c(r"\btl;?dr\b"),
    _c(r"\bkey\s+(points?|takeaways?|insights?)\b"),
    _c(r"\bmain\s+(points?|ideas?|findings?)\b"),
    _c(r"\b(brief|short)\s+(overview|recap|summary)\b"),
    _c(r"\bcondense\b"),
    _c(r"\bin\s+a\s+few\s+(words|sentences|lines|bullet\s*points?)\b"),
]

# Extract specific data from text
_EXTRACTION: List[re.Pattern] = [
    _c(r"\b(extract|pull\s+out|parse|scrape)\b"),
    _c(r"\bfind\s+all\b"),
    _c(r"\blist\s+(all|every|each)\b"),
    _c(r"\bidentify\s+(all|every)\b"),
    _c(r"\bnamed\s+entit(y|ies)\b"),
    _c(r"\b(phone\s+numbers?|email\s+addresses?|dates?|prices?)\s+(from|in|within)\b"),
]

# General knowledge, explanation, definition
_FACTUAL: List[re.Pattern] = [
    _c(r"\bwhat\s+is\b"),
    _c(r"\bwho\s+is\b"),
    _c(r"\bhow\s+does\b"),
    _c(r"\bexplain\b"),
    _c(r"\bdefine\b"),
    _c(r"\bdescribe\b"),
    _c(r"\bwhen\s+(did|was|were)\b"),
    _c(r"\bwhy\s+(is|does|do|did)\b"),
    _c(r"\bwhat\s+are\s+the\b"),
]


def _matches(text: str, patterns: List[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class RequestTypeClassifier:
    """Classify a prompt string into a :class:`RequestType`.

    Pattern sets are checked in priority order (most-specific first).
    Falls back to :attr:`~RequestType.CONVERSATIONAL` when nothing matches.
    """

    def classify(self, prompt: str) -> RequestType:
        if not prompt or not prompt.strip():
            return RequestType.CONVERSATIONAL

        text = prompt.strip()

        if _matches(text, _CREATIVE):
            return RequestType.CREATIVE
        if _matches(text, _CODE):
            return RequestType.CODE
        if _matches(text, _CLASSIFICATION):
            return RequestType.CLASSIFICATION
        if _matches(text, _SUMMARIZATION):
            return RequestType.SUMMARIZATION
        if _matches(text, _EXTRACTION):
            return RequestType.EXTRACTION
        if _matches(text, _FACTUAL):
            return RequestType.FACTUAL
        return RequestType.CONVERSATIONAL


default_request_type_classifier = RequestTypeClassifier()
