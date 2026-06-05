"""OpenAI embedding provider using ``text-embedding-3-small``.

Every incoming prompt is embedded via this provider before being looked up
in or stored to the vector store.  The model default is text-embedding-3-small
which produces 1536-dimensional vectors; pass ``dimensions`` to reduce size
(e.g. 512) for faster search at a small quality cost.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..exceptions import EmbeddingError
from .base import EmbeddingProvider

logger = logging.getLogger(__name__)

_MODEL = "text-embedding-3-small"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embed text using OpenAI's ``text-embedding-3-small`` model.

    Supports batching (up to 2048 inputs per call) and exponential back-off
    retry for transient API errors.

    Args:
        api_key: OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var
            when the ``openai`` client is constructed.
        model: Embedding model name.  Defaults to ``"text-embedding-3-small"``.
        dimensions: Output vector size.  ``None`` uses the model default
            (1536 for text-embedding-3-small).  Valid range: 1 – 1536.
        timeout: Per-request timeout in seconds.
        max_retries: Number of retry attempts before raising
            :class:`~src.exceptions.EmbeddingError`.
        batch_size: Number of texts per API call.

    Raises:
        EmbeddingError: If the ``openai`` package is not installed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _MODEL,
        dimensions: Optional[int] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        batch_size: int = 100,
    ) -> None:
        try:
            import openai  # noqa: PLC0415

            self._client = openai.OpenAI(api_key=api_key, timeout=timeout)
            self._async_client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout)
        except ImportError as exc:
            raise EmbeddingError(
                "openai package not installed. Run: pip install openai"
            ) from exc

        self._model = model
        self._dimensions = dimensions
        self._max_retries = max_retries
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed *texts* in batches with exponential back-off retry."""
        results: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            results.extend(self._embed_batch_sync(texts[i : i + self._batch_size]))
        return results

    def _embed_batch_sync(self, batch: List[str]) -> List[List[float]]:
        for attempt in Retrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(Exception),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                try:
                    kwargs = {"input": batch, "model": self._model}
                    if self._dimensions is not None:
                        kwargs["dimensions"] = self._dimensions
                    response = self._client.embeddings.create(**kwargs)
                    return [item.embedding for item in response.data]
                except Exception as exc:
                    raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc
        raise EmbeddingError("OpenAI embedding exhausted retries")  # pragma: no cover

    # ------------------------------------------------------------------
    # Async
    # ------------------------------------------------------------------

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        """Embed *texts* asynchronously in batches with back-off retry."""
        results: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            results.extend(await self._embed_batch_async(texts[i : i + self._batch_size]))
        return results

    async def _embed_batch_async(self, batch: List[str]) -> List[List[float]]:
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception_type(Exception),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    try:
                        kwargs = {"input": batch, "model": self._model}
                        if self._dimensions is not None:
                            kwargs["dimensions"] = self._dimensions
                        response = await self._async_client.embeddings.create(**kwargs)
                        return [item.embedding for item in response.data]
                    except Exception as exc:
                        raise EmbeddingError(
                            f"OpenAI async embedding failed: {exc}"
                        ) from exc
        except RetryError as exc:
            raise EmbeddingError("OpenAI async embedding exhausted retries") from exc
        raise EmbeddingError("Unexpected: async embedding loop exited")  # pragma: no cover
