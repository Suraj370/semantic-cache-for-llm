"""OpenAI provider — forwards to the OpenAI chat completions API."""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Dict

from fastapi import HTTPException

from ..api.models import ChatCompletionRequest
from .base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """Forward requests to the OpenAI chat completions API.

    Args:
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        base_url: Override base URL (Azure OpenAI, local proxy, etc.).
            Falls back to ``OPENAI_BASE_URL`` env var.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")

    @property
    def provider_name(self) -> str:
        return "openai"

    def _client(self):  # type: ignore[return]
        try:
            import openai  # noqa: PLC0415
            return openai.AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
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
            logger.error("OpenAI error: %s", exc)
            raise HTTPException(status_code=502, detail=f"OpenAI error: {exc}") from exc

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
            logger.error("OpenAI streaming error: %s", exc)
            raise HTTPException(status_code=502, detail=f"OpenAI error: {exc}") from exc
