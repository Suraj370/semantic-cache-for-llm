"""Pydantic schemas that exactly mirror the OpenAI chat completions contract.

Any client that already calls ``POST /v1/chat/completions`` on OpenAI can
point its ``base_url`` at this service with zero code changes.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Helpers used by the router
    # ------------------------------------------------------------------

    def last_user_content(self) -> Optional[str]:
        """Return the content of the last user-role message, or None."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    def system_prompt(self) -> Optional[str]:
        """Return the first system-role message content, or None."""
        for msg in self.messages:
            if msg.role == "system":
                return msg.content
        return None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None


class Choice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: Optional[str] = "stop"
    logprobs: Optional[Any] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None

    @classmethod
    def from_cache(
        cls,
        content: str,
        model: str,
        finish_reason: str = "stop",
        stored_usage: Optional[Dict[str, Any]] = None,
    ) -> "ChatCompletionResponse":
        """Reconstruct a response from a cached entry.

        A fresh ``id`` and ``created`` timestamp are generated so the
        response is indistinguishable from a live OpenAI reply.
        Prompt tokens are zeroed out to reflect that no LLM call was made.
        """
        usage = None
        if stored_usage:
            usage = Usage(
                prompt_tokens=0,
                completion_tokens=stored_usage.get("completion_tokens", 0),
                total_tokens=stored_usage.get("completion_tokens", 0),
            )
        return cls(
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=ResponseMessage(content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )
