"""Model name → LLM provider routing.

Routing rules (checked in order):
    gpt-*, o1-*, o3-*, o4-*          → OpenAI
    claude-*, us.anthropic.*          → Anthropic
    everything else                   → Ollama

The rules can be extended without touching the router or cache logic.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from .anthropic_provider import AnthropicProvider
from .base import LLMProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

# Prefixes that identify each provider — checked in declaration order.
_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-", "chatgpt-")
_ANTHROPIC_PREFIXES = ("claude-", "us.anthropic.", "eu.anthropic.", "ap.anthropic.")


@lru_cache(maxsize=1)
def _openai() -> OpenAIProvider:
    return OpenAIProvider()


@lru_cache(maxsize=1)
def _anthropic() -> AnthropicProvider:
    return AnthropicProvider()


@lru_cache(maxsize=1)
def _ollama() -> OllamaProvider:
    return OllamaProvider()


def get_provider(model: str) -> LLMProvider:
    """Return the appropriate :class:`LLMProvider` for *model*.

    Args:
        model: Model identifier from the incoming request (e.g.
            ``"gpt-4o"``, ``"claude-opus-4-8"``, ``"llama3"``).

    Returns:
        A cached :class:`LLMProvider` singleton.
    """
    model_lower = model.lower()

    if any(model_lower.startswith(p) for p in _OPENAI_PREFIXES):
        logger.debug("Routing model=%r to OpenAI", model)
        return _openai()

    if any(model_lower.startswith(p) for p in _ANTHROPIC_PREFIXES):
        logger.debug("Routing model=%r to Anthropic", model)
        return _anthropic()

    logger.debug("Routing model=%r to Ollama (no prefix match)", model)
    return _ollama()
