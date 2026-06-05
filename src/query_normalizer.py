"""Query normalisation — the semantic cache key strategy.

The cache has no traditional string key.  Instead, the *normalized* query
text is embedded and the resulting vector is compared via cosine similarity
against stored embeddings.  Normalisation is therefore the first step in
deriving a "semantic key": it reduces superficial variation (casing, filler
words, polite prefixes) so that queries with the same intent produce
embeddings that are close together in vector space.

Flow for every incoming prompt:
    raw query
        → QueryNormalizer.normalize()   ← this module
        → EmbeddingService.embed_one()  ← src/embedding/service.py
        → VectorStore.search()          ← src/vector_store/
        → hit (return cached response) or miss (call LLM, store result)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default filler words
# ---------------------------------------------------------------------------

_DEFAULT_FILLERS: Set[str] = {
    "um", "uh", "er", "ah", "hmm",
    "like", "you know", "basically", "literally", "actually",
    "just", "so", "well", "right",
    "okay", "ok", "alright",
    "yeah", "yep",
    "please",
    "could you", "can you", "would you", "will you",
    "i was wondering", "i would like to know", "i need to know",
    "tell me", "help me",
}

# ---------------------------------------------------------------------------
# Intent-extraction regex patterns  (applied after lowercasing)
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: List[tuple] = [
    (re.compile(r"^(could|can|would|will)\s+you\s+", re.I), ""),
    (re.compile(r"^please\s+", re.I), ""),
    (re.compile(r"^tell\s+me\s+", re.I), ""),
    (re.compile(r"^explain(\s+to\s+me)?\s+", re.I), ""),
    (re.compile(r"^i\s+(want|need|would\s+like)\s+to\s+know(\s+about)?\s+", re.I), ""),
    (re.compile(r"^i\s+was\s+wondering(\s+about)?\s+", re.I), ""),
]


class QueryNormalizer:
    """Normalise a raw query string to improve semantic cache hit rates.

    Steps (each independently toggleable):
    1. Unicode NFKC normalisation.
    2. Lowercase.
    3. Optional intent extraction (strip polite/conversational prefixes).
    4. Filler-word removal.
    5. Optional punctuation stripping.
    6. Whitespace collapse.
    7. Optional length truncation.

    Args:
        lowercase: Lowercase the query. Default ``True``.
        strip_punctuation: Remove non-alphanumeric, non-whitespace chars.
            Default ``False`` (question marks can carry semantic weight).
        remove_fillers: Remove filler words. Default ``True``.
        normalize_whitespace: Collapse runs of whitespace. Default ``True``.
        extract_intent: Strip polite prefixes to expose core intent.
            Default ``False``.
        max_length: Truncate to this many chars after all other steps.
            ``0`` means no truncation.
        custom_fillers: Additional filler words to merge with the defaults.
    """

    def __init__(
        self,
        lowercase: bool = True,
        strip_punctuation: bool = False,
        remove_fillers: bool = True,
        normalize_whitespace: bool = True,
        extract_intent: bool = False,
        max_length: int = 0,
        custom_fillers: Optional[Set[str]] = None,
    ) -> None:
        self._lowercase = lowercase
        self._strip_punctuation = strip_punctuation
        self._remove_fillers = remove_fillers
        self._normalize_whitespace = normalize_whitespace
        self._extract_intent = extract_intent
        self._max_length = max_length

        fillers = _DEFAULT_FILLERS.copy()
        if custom_fillers:
            fillers.update(f.lower() for f in custom_fillers)

        # Sort longest-first so multi-word fillers match before sub-phrases
        sorted_fillers = sorted(fillers, key=len, reverse=True)
        self._filler_patterns = [
            re.compile(r"(?<!\w)" + re.escape(f) + r"(?!\w)", re.I)
            for f in sorted_fillers
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, query: str) -> str:
        """Normalise a single query string.

        Returns:
            Normalised string.  May be empty if the query was all fillers.
        """
        if not query:
            return ""

        text = unicodedata.normalize("NFKC", query).strip()

        if self._lowercase:
            text = text.lower()

        # Intent extraction before filler removal so regex patterns match cleanly
        if self._extract_intent:
            text = self._apply_intent_patterns(text)

        if self._remove_fillers:
            text = self._strip_fillers(text)

        if self._strip_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)

        if self._normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        if self._max_length and len(text) > self._max_length:
            text = text[: self._max_length].rstrip()

        return text

    def normalize_batch(self, queries: List[str]) -> List[str]:
        """Normalise a list of queries."""
        return [self.normalize(q) for q in queries]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _strip_fillers(self, text: str) -> str:
        for pattern in self._filler_patterns:
            text = pattern.sub(" ", text)
        return text

    def _apply_intent_patterns(self, text: str) -> str:
        for pattern, replacement in _INTENT_PATTERNS:
            new_text = pattern.sub(replacement, text, count=1)
            if new_text != text:
                return new_text.strip()
        return text
