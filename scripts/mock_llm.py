"""Minimal mock LLM for benchmarking.

Simulates an OpenAI-compatible /v1/chat/completions endpoint with realistic
latency and token counts so benchmark results reflect real cache speedup.

Start with:
    uvicorn scripts.mock_llm:app --host 0.0.0.0 --port 9999

Then set in .env:
    OPENAI_BASE_URL=http://localhost:9999/v1      # local run
    OPENAI_BASE_URL=http://mock-llm:9999/v1       # docker-compose
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Mock LLM", docs_url=None, redoc_url=None)

# Canned responses with realistic token counts (~40-180 completion tokens each)
_RESPONSES = [
    "That's a great question. Based on the available information, there are several important factors to consider when approaching this topic.",
    "The answer involves a few key concepts. First, it's important to understand the underlying principles before diving into the specifics.",
    "There are multiple perspectives on this. The most widely accepted view is that the fundamentals need to be mastered before advanced concepts.",
    "This is well-documented in the literature. The core idea is straightforward once you break it down into its component parts.",
    "Excellent question. The short answer is yes, though the complete picture requires understanding a few prerequisite ideas first.",
    "From a technical standpoint, the solution involves applying established principles in a systematic way to arrive at the correct result.",
    "The historical context is important here. Over time, our understanding has evolved significantly, leading to the current consensus.",
    "In practice, this works by leveraging the underlying mechanisms in a way that is both efficient and reliable for most use cases.",
    "The key insight is that the problem can be decomposed into smaller sub-problems, each of which has a well-known solution.",
    "Research has shown that the most effective approach combines theoretical understanding with practical application and iteration.",
    "To summarize the main points: the concept is foundational, the applications are broad, and the implementation is straightforward.",
    "The best way to think about this is through analogy. Imagine the system as a series of interconnected components working in harmony.",
    "This depends on context, but the general principle holds across most scenarios you are likely to encounter in practice.",
    "The standard approach, which is supported by extensive evidence, is to start with the basics and build up systematically.",
    "A common misconception is that this is more complex than it actually is. In reality, the core mechanism is quite elegant.",
    "From first principles, we can derive that the optimal solution balances simplicity, correctness, and performance requirements.",
]


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    body = await request.json()

    # Simulate realistic LLM call latency (150–450 ms)
    await asyncio.sleep(random.uniform(0.15, 0.45))

    prompt_tokens = random.randint(20, 90)
    completion_tokens = random.randint(35, 160)
    content = random.choice(_RESPONSES)
    model = body.get("model", "gpt-4o-mini")

    return JSONResponse({
        "id": f"mock-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "mock-llm"}
