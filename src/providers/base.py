"""Abstract LLM provider interface.

Every concrete provider receives an OpenAI-format request and must return
an OpenAI-format response dict.  The cache layer never sees provider
internals — it only works with this normalised contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..api.models import ChatCompletionRequest


class LLMProvider(ABC):
    """Forward a chat completion request to an upstream LLM.

    Implementations are responsible for:
    - Translating ``ChatCompletionRequest`` to the provider's native format.
    - Calling the provider's API.
    - Translating the response back to an OpenAI-format dict so the cache
      layer and the HTTP response remain provider-agnostic.
    """

    @abstractmethod
    async def complete(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Send *request* to the upstream provider.

        Args:
            request: OpenAI-format chat completion request.

        Returns:
            OpenAI-format response dict (same shape as
            ``ChatCompletionResponse.model_dump()``).

        Raises:
            ``fastapi.HTTPException`` with an appropriate status code on
            upstream API errors.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier, e.g. ``"openai"``."""
