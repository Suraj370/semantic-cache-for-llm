"""Anthropic provider — converts OpenAI format ↔ Anthropic Messages API."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import HTTPException

from ..api.models import ChatCompletionRequest
from .base import LLMProvider

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 1024

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

    def _client(self):  # type: ignore[return]
        try:
            import anthropic  # noqa: PLC0415
            return anthropic.AsyncAnthropic(api_key=self._api_key)
        except ImportError as exc:
            raise HTTPException(
                status_code=500, detail="anthropic package not installed."
            ) from exc

    def _build_kwargs(self, request: ChatCompletionRequest) -> Dict[str, Any]:
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
        return kwargs

    async def complete(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        import anthropic  # noqa: PLC0415
        try:
            response = await self._client().messages.create(**self._build_kwargs(request))
            return self._to_openai_format(response, request.model)
        except anthropic.APIError as exc:
            logger.error("Anthropic error: %s", exc)
            raise HTTPException(status_code=502, detail=f"Anthropic error: {exc}") from exc

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """Stream from Anthropic and yield OpenAI-format SSE chunks."""
        import anthropic  # noqa: PLC0415

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        try:
            async with self._client().messages.stream(**self._build_kwargs(request)) as stream:
                async for event in stream:
                    chunk = self._event_to_openai_chunk(
                        event, chunk_id, created, request.model
                    )
                    if chunk is not None:
                        yield f"data: {json.dumps(chunk)}\n\n"

            # Final chunk with finish_reason
            final_message = await stream.get_final_message()
            finish_reason = _STOP_REASON_MAP.get(
                final_message.stop_reason or "end_turn", "stop"
            )
            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except anthropic.APIError as exc:
            logger.error("Anthropic streaming error: %s", exc)
            raise HTTPException(status_code=502, detail=f"Anthropic error: {exc}") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_messages(
        request: ChatCompletionRequest,
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        system: Optional[str] = None
        messages: List[Dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == "system":
                system = (system + "\n" + (msg.content or "")).strip() if system else msg.content
            else:
                messages.append({"role": msg.role, "content": msg.content or ""})
        return system, messages

    @staticmethod
    def _to_openai_format(response: Any, requested_model: str) -> Dict[str, Any]:
        content = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        finish_reason = _STOP_REASON_MAP.get(response.stop_reason or "end_turn", "stop")
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

    @staticmethod
    def _event_to_openai_chunk(
        event: Any,
        chunk_id: str,
        created: int,
        model: str,
    ) -> Optional[Dict[str, Any]]:
        """Convert an Anthropic stream event to an OpenAI chunk dict, or None to skip."""
        event_type = getattr(event, "type", None)

        if event_type == "content_block_delta":
            delta_text = getattr(getattr(event, "delta", None), "text", None)
            if delta_text:
                return {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta_text},
                            "finish_reason": None,
                        }
                    ],
                }

        if event_type == "message_start":
            # Emit the role delta once at the start
            return {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }

        return None
