"""POST /v1/chat/completions — OpenAI-compatible semantic cache proxy.

Streaming behaviour
-------------------
Cache HIT  + stream=true  → regular JSON response returned instantly.
                            No chunking needed; the response is complete.
Cache MISS + stream=true  → SSE stream forwarded chunk-by-chunk to the
                            client while content is buffered in memory.
                            Once ``[DONE]`` is received the assembled
                            response is written to the cache.  If the
                            stream errors mid-flight nothing is cached.
Cache HIT  + stream=false → regular JSON response (unchanged).
Cache MISS + stream=false → forward, cache, return JSON (unchanged).

Response headers
----------------
X-Cache: HIT | MISS
X-Cache-Similarity: 0.9312   (on HIT only)
X-Cache-Provider: openai | anthropic | ollama   (on MISS only)
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from ..cache_key import LLMContext
from ..exceptions import EmbeddingError
from ..models import CacheEntry
from ..providers.registry import get_provider
from ..ttl_classifier import TtlTier, default_classifier
from .dependencies import (
    EmbeddingServiceDep,
    KeyBuilderDep,
    NormalizerDep,
    ThresholdDep,
    VectorStoreDep,
)
from .models import ChatCompletionRequest, ChatCompletionResponse
from .streaming import stream_and_cache

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    embeddings: EmbeddingServiceDep,
    store: VectorStoreDep,
    normalizer: NormalizerDep,
    key_builder: KeyBuilderDep,
    threshold: ThresholdDep,
) -> JSONResponse | StreamingResponse:
    """Semantic cache proxy with multi-provider routing and streaming support."""

    # n > 1 can't be meaningfully cached (multiple distinct choices per request)
    bypass_cache = (request.n or 1) > 1

    # ------------------------------------------------------------------
    # 1. Extract query and context
    # ------------------------------------------------------------------
    last_user_content = request.last_user_content()
    if not last_user_content:
        raise HTTPException(status_code=422, detail="No user message found in messages.")

    # Classify TTL tier from the user prompt.
    # NO_CACHE (live/real-time queries) bypasses the cache entirely.
    ttl_tier = default_classifier.classify(last_user_content)
    if ttl_tier is TtlTier.NO_CACHE:
        bypass_cache = True
    entry_ttl: Optional[int] = ttl_tier.seconds if ttl_tier is not TtlTier.NO_CACHE else None
    logger.debug("TTL tier=%s ttl=%s query=%r", ttl_tier.value, entry_ttl, last_user_content[:60])

    context_key: Optional[str] = None
    embedding: Optional[List[float]] = None
    normalized = normalizer.normalize(last_user_content)

    if not bypass_cache:
        context = LLMContext(
            system_prompt=request.system_prompt(),
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        context_key = key_builder.build(context)

        # ------------------------------------------------------------------
        # 2. Normalize → embed
        # ------------------------------------------------------------------
        try:
            embedding = await embeddings.aembed_one(normalized)
        except EmbeddingError as exc:
            logger.warning("Embedding failed, bypassing cache: %s", exc)

        # ------------------------------------------------------------------
        # 3. Search cache (same path for streaming and non-streaming)
        # ------------------------------------------------------------------
        if embedding is not None:
            hits = store.search(embedding, k=1, threshold=threshold, context_key=context_key)
            if hits:
                entry, similarity = hits[0]
                entry.touch()
                logger.debug(
                    "Cache HIT  stream=%s  sim=%.4f  query=%r",
                    request.stream,
                    similarity,
                    last_user_content[:60],
                )
                cached_response = ChatCompletionResponse.from_cache(
                    content=entry.response,
                    model=request.model,
                    finish_reason=entry.metadata.get("finish_reason", "stop"),
                    stored_usage=entry.metadata.get("usage"),
                )
                # HIT: always return instantly as JSON regardless of stream flag.
                # The response is already complete — no benefit to chunking it.
                return JSONResponse(
                    content=cached_response.model_dump(),
                    headers={
                        "X-Cache": "HIT",
                        "X-Cache-Similarity": f"{similarity:.4f}",
                    },
                )

    # ------------------------------------------------------------------
    # 4. Cache miss — route to the correct provider
    # ------------------------------------------------------------------
    provider = get_provider(request.model)
    logger.debug(
        "Cache MISS  stream=%s  provider=%s  query=%r",
        request.stream,
        provider.provider_name,
        last_user_content[:60],
    )

    miss_headers = {
        "X-Cache": "MISS",
        "X-Cache-Provider": provider.provider_name,
    }

    # ------------------------------------------------------------------
    # 5a. Streaming miss — forward chunks, buffer, cache on completion
    # ------------------------------------------------------------------
    if request.stream:
        return StreamingResponse(
            stream_and_cache(
                request=request,
                provider=provider,
                store=store,
                normalizer=normalizer,
                embedding=embedding,
                context_key=context_key,
                ttl=entry_ttl,
                tags=request.cache_tags or None,
            ),
            media_type="text/event-stream",
            headers=miss_headers,
        )

    # ------------------------------------------------------------------
    # 5b. Non-streaming miss — forward, cache, return
    # ------------------------------------------------------------------
    upstream_response = await provider.complete(request)

    if not bypass_cache and embedding is not None:
        assistant_content = (
            upstream_response["choices"][0]["message"].get("content") or ""
        )
        meta: Dict[str, Any] = {
            "finish_reason": upstream_response["choices"][0].get("finish_reason", "stop"),
            "usage": upstream_response.get("usage"),
            "model": upstream_response.get("model", request.model),
            "provider": provider.provider_name,
            "ttl_tier": ttl_tier.value,
        }
        if request.cache_tags:
            meta["tags"] = request.cache_tags
        store.add(CacheEntry(
            query=last_user_content,
            normalized_query=normalized,
            response=assistant_content,
            embedding=embedding,
            context_key=context_key,
            ttl=entry_ttl,
            metadata=meta,
        ))

    return JSONResponse(content=upstream_response, headers=miss_headers)
