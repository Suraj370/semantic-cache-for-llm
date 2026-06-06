"""LLM cost-savings estimator.

Ported and updated from src-old/cost_calculator.py with current model
pricing (June 2025) and Claude 4 models added.

Pricing tiers
-------------
Each model entry is (input, cached_input, output) in USD per 1 000 000 tokens.
``cached_input`` is OpenAI/Anthropic's *server-side* prompt-cache discount —
a separate mechanism from this semantic cache.  When a request's system prompt
was cached by the provider, input tokens are billed at the lower rate.

Usage::

    savings = estimate_savings("gpt-4o-mini", input_tokens=200, output_tokens=350)
    # → 0.000225  (USD saved for one avoided LLM call)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# (input, cached_input, output) — USD per 1 000 000 tokens, prices as of June 2025.
# cached_input = None means the provider does not publish a cached-input rate.
# fmt: off
_PRICING_TABLE: Dict[str, Tuple[float, Optional[float], float]] = {
    # OpenAI
    "gpt-4o":                      (2.50,  1.25,  10.00),
    "gpt-4o-mini":                 (0.15,  0.075,  0.60),
    "gpt-4-turbo":                 (10.00, None,  30.00),
    "gpt-4":                       (30.00, None,  60.00),
    "gpt-3.5-turbo":               (0.50,  None,   1.50),
    "o1":                          (15.00, 7.50,  60.00),
    "o1-mini":                     (3.00,  1.50,  12.00),
    "o3-mini":                     (1.10,  0.55,   4.40),
    # Anthropic — Claude 4
    "claude-opus-4-8":             (15.00, 1.50,  75.00),
    "claude-sonnet-4-6":           (3.00,  0.30,  15.00),
    "claude-haiku-4-5":            (0.80,  0.08,   4.00),
    "claude-haiku-4-5-20251001":   (0.80,  0.08,   4.00),
    # Anthropic — Claude 3.x
    "claude-3-5-sonnet-20241022":  (3.00,  0.30,  15.00),
    "claude-3-5-haiku-20241022":   (0.80,  0.08,   4.00),
    "claude-3-opus-20240229":      (15.00, 1.50,  75.00),
    "claude-3-haiku-20240307":     (0.25,  0.03,   1.25),
}
# fmt: on

# text-embedding-3-small: $0.02 per 1M tokens.
# A typical query is ~100 tokens → ~$0.000000002 per lookup — effectively zero.
_EMBEDDING_COST_PER_CALL: float = 100 * 0.02 / 1_000_000


@dataclass(frozen=True)
class ModelPricing:
    model: str
    input_cost_per_1m: float          # USD per 1M input tokens (standard)
    cached_input_cost_per_1m: Optional[float]  # USD per 1M (provider prompt cache)
    output_cost_per_1m: float         # USD per 1M output tokens

    def call_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        use_cached_input: bool = False,
    ) -> float:
        """Compute USD cost for one LLM call.

        Args:
            input_tokens: Number of prompt tokens.
            output_tokens: Number of completion tokens.
            use_cached_input: Apply the provider's cached-input rate if
                available (e.g. when the system prompt was already in the
                provider's KV cache).
        """
        if use_cached_input and self.cached_input_cost_per_1m is not None:
            in_rate = self.cached_input_cost_per_1m
        else:
            in_rate = self.input_cost_per_1m

        return (
            input_tokens  * in_rate                  / 1_000_000
            + output_tokens * self.output_cost_per_1m / 1_000_000
        )


def _get_pricing(model: str) -> Optional[ModelPricing]:
    """Return pricing for *model*, trying progressively shorter prefixes."""
    if model in _PRICING_TABLE:
        inp, cached, out = _PRICING_TABLE[model]
        return ModelPricing(model, inp, cached, out)
    # Strip date suffixes: "gpt-4o-2024-08-06" → "gpt-4o"
    parts = model.split("-")
    for n in range(len(parts) - 1, 0, -1):
        key = "-".join(parts[:n])
        if key in _PRICING_TABLE:
            inp, cached, out = _PRICING_TABLE[key]
            return ModelPricing(key, inp, cached, out)
    return None


def estimate_savings(
    model: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    avg_input_tokens: int = 200,
    avg_output_tokens: int = 350,
    use_cached_input: bool = False,
) -> float:
    """Estimate USD saved by a single semantic-cache hit for *model*.

    The saving is the full LLM call cost that was avoided, minus the
    negligible embedding-lookup cost.

    Returns 0.0 when the model has no known pricing.
    """
    pricing = _get_pricing(model)
    if pricing is None:
        logger.debug("No pricing for model %r; cost savings will be 0.0", model)
        return 0.0

    inp = input_tokens if input_tokens is not None else avg_input_tokens
    out = output_tokens if output_tokens is not None else avg_output_tokens
    llm_cost = pricing.call_cost(inp, out, use_cached_input=use_cached_input)
    return max(0.0, llm_cost - _EMBEDDING_COST_PER_CALL)


def estimate_tokens_saved(
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    avg_input_tokens: int = 200,
    avg_output_tokens: int = 350,
) -> int:
    """Estimate total LLM tokens saved by a single semantic-cache hit."""
    inp = input_tokens if input_tokens is not None else avg_input_tokens
    out = output_tokens if output_tokens is not None else avg_output_tokens
    return inp + out


def pricing_table() -> Dict[str, Dict[str, object]]:
    """Return the full pricing table as a plain dict (for API responses)."""
    return {
        model: {
            "input_per_1m":        inp,
            "cached_input_per_1m": cached,
            "output_per_1m":       out,
        }
        for model, (inp, cached, out) in _PRICING_TABLE.items()
    }
