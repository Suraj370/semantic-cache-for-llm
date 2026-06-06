"""Streaming helpers: stream from a provider while buffering for cache storage.

Design
------
When a streaming request results in a cache miss:

1. The provider yields SSE chunks (``data: {...}\\n\\n``) in OpenAI format.
2. Each chunk is forwarded to the client immediately — zero added latency.
3. Content deltas are accumulated in a local buffer.
4. When ``data: [DONE]`` arrives the buffer is assembled into a complete
   response and written to the cache — but only if the stream finished
   without error.  A partial response is never cached.

Cache hits with ``stream=True``
--------------------------------
Hits are returned as a regular JSON response (no streaming needed).
This is intentional: the response is already complete, so chunking it
adds latency without benefit.  Clients that require strict SSE must handle
both ``application/json`` and ``text/event-stream`` content types, which
is standard practice for proxy-aware code.
"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator, List, Optional

import monitoring.metrics as metrics
from ..models import CacheEntry
from ..providers.base import LLMProvider
from ..query_normalizer import QueryNormalizer
from ..vector_store import InMemoryVectorStore
from .models import ChatCompletionRequest

logger = logging.getLogger(__name__)


async def stream_and_cache(
    request: ChatCompletionRequest,
    provider: LLMProvider,
    store: InMemoryVectorStore,
    normalizer: QueryNormalizer,
    embedding: Optional[List[float]],
    context_key: Optional[str],
    ttl: Optional[int] = None,
    tags: Optional[List[str]] = None,
    request_type: Optional[str] = None,
    start_monotonic: Optional[float] = None,
) -> AsyncIterator[str]:
    """Stream provider chunks to the caller and cache the assembled response.

    Yields raw SSE lines (e.g. ``"data: {...}\\n\\n"``).  The cache entry
    is written after the final ``[DONE]`` sentinel is received; if the
    stream errors before that point nothing is cached.

    Args:
        request: Original chat completion request.
        provider: Provider selected by the registry for this model.
        store: Live vector store instance.
        normalizer: Query normalizer for the cache entry.
        embedding: Pre-computed embedding of the normalised query, or
            ``None`` if embedding failed (cache write is skipped).
        context_key: Context partition key, or ``None`` on bypass.
    """
    content_buffer: List[str] = []
    finish_reason: str = "stop"
    usage: Optional[dict] = None
    response_model: str = request.model
    stream_complete = False

    try:
        async for chunk_line in provider.stream(request):
            yield chunk_line

            stripped = chunk_line.strip()

            if stripped == "data: [DONE]":
                stream_complete = True
                break

            if not stripped.startswith("data: "):
                continue

            try:
                data = json.loads(stripped[len("data: "):])
            except json.JSONDecodeError:
                continue

            # Accumulate content deltas
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    content_buffer.append(delta["content"])
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

            # Some providers send usage on the final chunk
            if data.get("usage"):
                usage = data["usage"]
            if data.get("model"):
                response_model = data["model"]

    except Exception as exc:
        # Stream failed mid-flight — log but do not cache the partial response.
        logger.error(
            "Streaming error from provider=%s, response not cached: %s",
            provider.provider_name,
            exc,
        )
        raise

    # Record miss latency now that the full stream has completed (or failed)
    if start_monotonic is not None:
        metrics.record_miss(
            model=request.model,
            request_type=request_type or "conversational",
            latency_s=time.monotonic() - start_monotonic,
        )
        metrics.update_cache_size(store.size())

    # Only cache when the stream finished cleanly and we have an embedding
    if stream_complete and embedding is not None and content_buffer:
        full_content = "".join(content_buffer)
        last_user = request.last_user_content() or ""
        meta = {
            "finish_reason": finish_reason,
            "usage": usage,
            "model": response_model,
            "provider": provider.provider_name,
        }
        if tags:
            meta["tags"] = tags
        if request_type:
            meta["request_type"] = request_type
        store.add(CacheEntry(
            query=last_user,
            normalized_query=normalizer.normalize(last_user),
            response=full_content,
            embedding=embedding,
            context_key=context_key,
            ttl=ttl,
            metadata=meta,
        ))
        logger.debug(
            "Cached streamed response  provider=%s  tokens=%d",
            provider.provider_name,
            len(content_buffer),
        )
