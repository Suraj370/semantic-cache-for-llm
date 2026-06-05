"""Ollama provider — forwards to Ollama's OpenAI-compatible endpoint."""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Dict

from fastapi import HTTPException

from ..api.models import ChatCompletionRequest
from .base import LLMProvider

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OllamaProvider(LLMProvider):
    """Forward requests to a locally running Ollama instance.

    Ollama exposes an OpenAI-compatible API so no request / response
    translation is needed — just a base URL swap.

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

    def _client(self):  # type: ignore[return]
        try:
            import openai  # noqa: PLC0415
            return openai.AsyncOpenAI(base_url=self._base_url, api_key="ollama")
        except ImportError as exc:
            raise HTTPException(status_code=500, detail="openai package not installed.") from exc

    def _build_kwargs(self, request: ChatCompletionRequest, stream: bool) -> Dict[str, Any]:
        kwargs = request.model_dump(exclude_none=True)
        kwargs["stream"] = stream
        return kwargs

    async def complete(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        import openai  # noqa: PLC0415
        try:
            response = await self._client().chat.completions.create(
                **self._build_kwargs(request, stream=False)
            )
            return response.model_dump()
        except openai.OpenAIError as exc:
            logger.error("Ollama error: %s", exc)
            raise HTTPException(status_code=502, detail=f"Ollama error: {exc}") from exc

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        import openai  # noqa: PLC0415
        try:
            response = await self._client().chat.completions.create(
                **self._build_kwargs(request, stream=True)
            )
            async for chunk in response:
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        except openai.OpenAIError as exc:
            logger.error("Ollama streaming error: %s", exc)
            raise HTTPException(status_code=502, detail=f"Ollama error: {exc}") from exc
