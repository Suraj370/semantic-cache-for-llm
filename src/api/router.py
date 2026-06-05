"""POST /v1/chat/completions — OpenAI-compatible semantic cache proxy.

Request / response shapes are identical to the OpenAI API so any client
can switch to this service by changing only its ``base_url``.

Provider routing (based on the ``model`` field):
    gpt-*, o1-*, o3-*, o4-*  →  OpenAI
    claude-*, us.anthropic.*  →  Anthropic  (translated to Messages API)
    everything else           →  Ollama

The cache layer is provider-agnostic.  ``context_key`` already includes
the model name so a cached OpenAI response is never served for an Anthropic
request — no extra logic required.

Cache hit responses include two extra headers:
    X-Cache: HIT
    X-Cache-Similarity: 0.9312

Cache miss responses include:
    X-Cache: MISS
    X-Cache-Provider: openai | anthropic | ollama
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..cache_key import LLMContext
from ..exceptions import EmbeddingError
from ..models import CacheEntry
from ..providers.registry import get_provider
from .dependencies import (
    EmbeddingServiceDep,
    KeyBuilderDep,
    NormalizerDep,
    ThresholdDep,
    VectorStoreDep,
)
from .models import ChatCompletionRequest, ChatCompletionResponse

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
) -> JSONResponse:
    """Semantic cache proxy with multi-provider routing.

    - Streaming requests (``stream=True``) and multi-choice requests
      (``n > 1``) bypass the cache and are forwarded directly.
    - On embedding failure the request is forwarded transparently; the
      cache is skipped, not surfaced as an error to the caller.
    """
    bypass_cache = bool(request.stream) or (request.n or 1) > 1

    # ------------------------------------------------------------------
    # 1. Extract query and context
    # ------------------------------------------------------------------
    last_user_content = request.last_user_content()
    if not last_user_content:
        raise HTTPException(status_code=422, detail="No user message found in messages.")

    context_key: Optional[str] = None
    embedding = None
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
        # 3. Search cache
        # ------------------------------------------------------------------
        if embedding is not None:
            hits = store.search(embedding, k=1, threshold=threshold, context_key=context_key)
            if hits:
                entry, similarity = hits[0]
                entry.touch()
                logger.debug(
                    "Cache HIT  provider=%s  sim=%.4f  query=%r",
                    entry.metadata.get("provider", "unknown"),
                    similarity,
                    last_user_content[:60],
                )
                cached_response = ChatCompletionResponse.from_cache(
                    content=entry.response,
                    model=request.model,
                    finish_reason=entry.metadata.get("finish_reason", "stop"),
                    stored_usage=entry.metadata.get("usage"),
                )
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
        "Cache MISS  provider=%s  query=%r",
        provider.provider_name,
        last_user_content[:60],
    )

    upstream_response = await provider.complete(request)

    # ------------------------------------------------------------------
    # 5. Store result (skip on bypass or embedding failure)
    # ------------------------------------------------------------------
    if not bypass_cache and embedding is not None:
        assistant_content = (
            upstream_response["choices"][0]["message"].get("content") or ""
        )
        finish_reason = upstream_response["choices"][0].get("finish_reason", "stop")
        usage: Optional[Dict[str, Any]] = upstream_response.get("usage")

        store.add(CacheEntry(
            query=last_user_content,
            normalized_query=normalized,
            response=assistant_content,
            embedding=embedding,
            context_key=context_key,
            metadata={
                "finish_reason": finish_reason,
                "usage": usage,
                "model": upstream_response.get("model", request.model),
                "provider": provider.provider_name,
            },
        ))

    return JSONResponse(
        content=upstream_response,
        headers={
            "X-Cache": "MISS",
            "X-Cache-Provider": provider.provider_name,
        },
    )
