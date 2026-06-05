"""POST /v1/chat/completions — OpenAI-compatible semantic cache proxy.

Request / response shapes are identical to the OpenAI API so any client
can switch to this service by changing only its ``base_url``.

Cache hit responses include two extra headers:
    X-Cache: HIT
    X-Cache-Similarity: 0.9312

Cache miss responses include:
    X-Cache: MISS
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..cache_key import LLMContext
from ..exceptions import EmbeddingError
from ..models import CacheEntry
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
    """Semantic cache proxy for OpenAI chat completions.

    - Streaming requests (``stream=True``) and multi-choice requests
      (``n > 1``) bypass the cache and are forwarded directly — these
      response shapes can't be meaningfully cached entry-per-entry.
    - On embedding failure the request is forwarded transparently; the
      cache is simply skipped, not surfaced as an error.
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
        normalized = normalizer.normalize(last_user_content)
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
                logger.debug("Cache HIT  sim=%.4f  query=%r", similarity, last_user_content[:60])

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
    # 4. Cache miss — forward to OpenAI
    # ------------------------------------------------------------------
    logger.debug("Cache MISS  query=%r", last_user_content[:60])
    oai_response = await _forward_to_openai(request)

    # ------------------------------------------------------------------
    # 5. Store result (skip on bypass or embedding failure)
    # ------------------------------------------------------------------
    if not bypass_cache and embedding is not None:
        assistant_content = (
            oai_response["choices"][0]["message"].get("content") or ""
        )
        finish_reason = oai_response["choices"][0].get("finish_reason", "stop")
        usage: Optional[Dict[str, Any]] = oai_response.get("usage")

        store.add(CacheEntry(
            query=last_user_content,
            normalized_query=normalizer.normalize(last_user_content),
            response=assistant_content,
            embedding=embedding,
            context_key=context_key,
            metadata={
                "finish_reason": finish_reason,
                "usage": usage,
                "model": oai_response.get("model", request.model),
            },
        ))

    return JSONResponse(
        content=oai_response,
        headers={"X-Cache": "MISS"},
    )


# ---------------------------------------------------------------------------
# Private — OpenAI forwarding
# ---------------------------------------------------------------------------

async def _forward_to_openai(request: ChatCompletionRequest) -> Dict[str, Any]:
    """Forward the request to the real OpenAI API and return the raw dict."""
    try:
        import openai  # noqa: PLC0415
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="openai package not installed.",
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    client = openai.AsyncOpenAI(api_key=api_key)

    try:
        response = await client.chat.completions.create(
            **request.model_dump(exclude_none=True)
        )
        return response.model_dump()
    except openai.OpenAIError as exc:
        logger.error("OpenAI API error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Upstream OpenAI error: {exc}") from exc
