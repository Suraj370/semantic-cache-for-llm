"""Anthropic provider — converts OpenAI format ↔ Anthropic Messages API.

The Anthropic Messages API differs from OpenAI in three key ways:
1. ``system`` is a top-level field, not a message role.
2. ``max_tokens`` is required (no default).
3. Response shape uses ``content[].text`` and ``stop_reason`` instead of
   ``choices[].message.content`` and ``finish_reason``.

This provider handles both translations so the cache layer and the router
remain completely provider-agnostic.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from ..api.models import ChatCompletionRequest
from .base import LLMProvider

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 1024

# Anthropic stop_reason → OpenAI finish_reason
_STOP_REASON_MAP: Dict[str, str] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


class AnthropicProvider(LLMProvider):
    """Forward requests to the Anthropic Messages API.

    Args:
        api_key: Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def complete(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise HTTPException(
                status_code=500, detail="anthropic package not installed."
            ) from exc

        client = anthropic.AsyncAnthropic(api_key=self._api_key)

        system, messages = self._split_messages(request)

        kwargs: Dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or _DEFAULT_MAX_TOKENS,
        }
        if system:
            kwargs["system"] = system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop:
            kwargs["stop_sequences"] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )

        try:
            response = await client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            logger.error("Anthropic API error: %s", exc)
            raise HTTPException(status_code=502, detail=f"Anthropic error: {exc}") from exc

        return self._to_openai_format(response, request.model)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_messages(
        request: ChatCompletionRequest,
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Separate the system prompt from user/assistant turns."""
        system: Optional[str] = None
        messages: List[Dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == "system":
                # Anthropic only supports a single system prompt string
                system = (system + "\n" + (msg.content or "")).strip() if system else msg.content
            else:
                messages.append({"role": msg.role, "content": msg.content or ""})
        return system, messages

    @staticmethod
    def _to_openai_format(response: Any, requested_model: str) -> Dict[str, Any]:
        """Convert an Anthropic ``Message`` object to an OpenAI response dict."""
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        finish_reason = _STOP_REASON_MAP.get(
            response.stop_reason or "end_turn", "stop"
        )

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.model or requested_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
            "usage": usage,
            "system_fingerprint": None,
        }
