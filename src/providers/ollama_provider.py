"""Ollama provider — forwards to Ollama's OpenAI-compatible endpoint.

Ollama exposes ``POST /v1/chat/completions`` at ``http://localhost:11434``
by default, which is wire-compatible with the OpenAI SDK.  No request or
response translation is needed — just a base URL swap.

Set ``OLLAMA_BASE_URL`` to override the default host (useful for remote
or Docker-networked Ollama instances).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from fastapi import HTTPException

from ..api.models import ChatCompletionRequest
from .base import LLMProvider

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OllamaProvider(LLMProvider):
    """Forward requests to a locally running Ollama instance.

    Args:
        base_url: Ollama base URL. Falls back to ``OLLAMA_BASE_URL`` env
            var, then ``http://localhost:11434/v1``.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or _DEFAULT_OLLAMA_BASE_URL
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def complete(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise HTTPException(
                status_code=500, detail="openai package not installed."
            ) from exc

        # Ollama's OpenAI-compat endpoint doesn't validate the API key
        client = openai.AsyncOpenAI(
            base_url=self._base_url,
            api_key="ollama",
        )
        try:
            response = await client.chat.completions.create(
                **request.model_dump(exclude_none=True)
            )
            return response.model_dump()
        except openai.OpenAIError as exc:
            logger.error("Ollama error: %s", exc)
            raise HTTPException(status_code=502, detail=f"Ollama error: {exc}") from exc
