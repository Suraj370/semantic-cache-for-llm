"""Context-aware cache key generation.

Two prompts that are semantically identical but issued under *different*
LLM configurations (system prompt, model, temperature, max_tokens) must
never share a cache entry — the responses would be meaningfully different.

``CacheKeyBuilder`` hashes the LLM context into a short, deterministic
``context_key`` string.  This key is stored on every ``CacheEntry`` and
used as a hard filter in ``VectorStore.search`` before cosine similarity
is even considered.

Lookup flow:
    raw query
        → QueryNormalizer.normalize()
        → EmbeddingService.embed_one()
        → VectorStore.search(embedding, context_key=key)   ← hard filter here
            • discard candidates whose context_key != key
            • rank remaining candidates by cosine similarity
        → hit / miss
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMContext:
    """Immutable snapshot of LLM call parameters that affect the response.

    All fields are optional so callers can supply only what they know.
    Two ``LLMContext`` objects are equal when every supplied field matches.

    Args:
        system_prompt: The system prompt / instruction prefix sent to the LLM.
        model: Model identifier (e.g. ``"gpt-4o"``, ``"claude-opus-4-8"``).
        temperature: Sampling temperature.
        max_tokens: Maximum output token limit.
    """

    system_prompt: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class CacheKeyBuilder:
    """Build a deterministic context key from an :class:`LLMContext`.

    The key is the first ``digest_length`` hex characters of the SHA-256
    hash of a canonicalised JSON representation of the context.  Callers
    store this on ``CacheEntry.context_key`` and pass it to
    ``VectorStore.search`` so entries are partitioned by LLM configuration.

    Args:
        digest_length: Number of hex characters to use from the SHA-256
            digest.  Default ``16`` (64-bit prefix) gives a collision
            probability low enough for any realistic cache size.
    """

    def __init__(self, digest_length: int = 16) -> None:
        if digest_length < 8 or digest_length > 64:
            raise ValueError("digest_length must be between 8 and 64")
        self._digest_length = digest_length

    def build(self, context: LLMContext) -> str:
        """Return a short hex digest that uniquely identifies *context*.

        ``None`` fields are normalised to a sentinel so that
        ``LLMContext(model="gpt-4o")`` and
        ``LLMContext(model="gpt-4o", temperature=None)`` produce the same key.

        Args:
            context: The LLM context to hash.

        Returns:
            Hex string of length ``digest_length``.
        """
        payload = {
            "system_prompt": context.system_prompt or "",
            "model": context.model or "",
            # Use a string sentinel for None so JSON is stable across
            # languages / tools that may deserialise differently.
            "temperature": context.temperature if context.temperature is not None else "__none__",
            "max_tokens": context.max_tokens if context.max_tokens is not None else "__none__",
        }
        # sort_keys guarantees field order is deterministic regardless of
        # Python dict insertion order or future dataclass field reordering.
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return digest[: self._digest_length]

    def build_from_parts(
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Convenience wrapper — build key from individual parameters.

        Args:
            system_prompt: System prompt string.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Token limit.

        Returns:
            Hex context key.
        """
        return self.build(
            LLMContext(
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )


# Module-level default instance — use this unless you need a custom digest length.
default_key_builder = CacheKeyBuilder()
