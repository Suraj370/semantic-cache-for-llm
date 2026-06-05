"""Abstract LLM provider interface.

Every concrete provider receives an OpenAI-format request and must return
an OpenAI-format response dict.  The cache layer never sees provider
internals — it only works with this normalised contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict

from ..api.models import ChatCompletionRequest


class LLMProvider(ABC):
    """Forward a chat completion request to an upstream LLM.

    Two calling modes are supported:

    ``complete()``
        Non-streaming.  Returns a full OpenAI-format response dict.
        Used for cache misses on non-streaming requests and for
        buffering full responses before caching.

    ``stream()``
        Streaming.  Yields SSE lines in OpenAI chunk format
        (``"data: {...}\\n\\n"`` followed by ``"data: [DONE]\\n\\n"``).
        The caller is responsible for forwarding chunks to the client
        and accumulating them for cache storage.

    Both methods must always behave as non-streaming / streaming
    respectively, regardless of ``request.stream`` — the router sets the
    appropriate mode via which method it calls.
    """

    @abstractmethod
    async def complete(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Send *request* to the upstream provider (non-streaming).

        Returns:
            OpenAI-format response dict.

        Raises:
            ``fastapi.HTTPException`` on upstream API errors.
        """

    @abstractmethod
    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """Stream *request* to the upstream provider.

        Yields:
            SSE lines in OpenAI chunk format, e.g.::

                "data: {\\"id\\":\\"chatcmpl-...\\", \\"choices\\":[...]}\n\n"

            Terminated by ``"data: [DONE]\n\n"``.

        Raises:
            ``fastapi.HTTPException`` on upstream API errors.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier, e.g. ``"openai"``."""
