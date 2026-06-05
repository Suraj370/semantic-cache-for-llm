"""OpenAI provider — forwards the request as-is using the OpenAI SDK."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from fastapi import HTTPException

from ..api.models import ChatCompletionRequest
from .base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """Forward requests to the OpenAI chat completions API.

    The request is already in OpenAI format so no translation is needed.

    Args:
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        base_url: Override the OpenAI base URL (useful for Azure OpenAI or
            local proxies). Falls back to ``OPENAI_BASE_URL`` env var.
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

    async def complete(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise HTTPException(status_code=500, detail="openai package not installed.") from exc

        client = openai.AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )
        try:
            response = await client.chat.completions.create(
                **request.model_dump(exclude_none=True)
            )
            return response.model_dump()
        except openai.OpenAIError as exc:
            logger.error("OpenAI API error: %s", exc)
            raise HTTPException(status_code=502, detail=f"OpenAI error: {exc}") from exc
