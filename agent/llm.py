"""Stream one chat completion from the local vLLM server, with hard time limits (PLAN M3.1).

A caller is waiting on the phone, so a slow or broken LLM must turn into a typed LLMUnavailable
quickly; the caller side then plays a fallback line. Rules (Cyril, M3.1):
- Separate limits for connecting, for the first token, between chunks, and for the whole call.
- First token = the first chunk with text or a tool-call delta. vLLM's role-only first chunk is
  recorded as first_chunk_ns (diagnostic) and does not count.
- At most one retry, and only if nothing was received; never once a token has streamed, so a
  half-heard reply is never repeated. The SDK's own retries are off (max_retries=0).
- A cancellation (barge-in) propagates as CancelledError and closes the stream. It is never turned
  into LLMUnavailable (CLAUDE.md rule 7).
Tool-call deltas are joined by the SDK's ChatCompletionStreamState, not by code here.
"""

import asyncio
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import httpx2
from openai import APIConnectionError, APIError, APIStatusError, AsyncOpenAI, omit
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam

from agent import vllm_server

# All five are initial; revisit after M4.5 (baseline latency).
CONNECT_TIMEOUT_S = 1.0
FIRST_TOKEN_TIMEOUT_S = 2.0  # from sending the request to the first text or tool-call delta
INTER_CHUNK_TIMEOUT_S = 2.0
TOTAL_TIMEOUT_S = 8.0  # the whole call, retry included
MAX_TOKENS = 256
ATTEMPTS = 2  # one retry, only when nothing was received
MODEL = "llm"  # --served-model-name in agent/vllm_server.py

Reason = Literal["connect", "first_token", "stalled", "total", "dropped", "http_error"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text as the model wrote it; the tool layer validates it


@dataclass(frozen=True)
class LLMReply:
    text: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    first_chunk_ns: int | None  # perf_counter_ns of the first chunk of any kind (diagnostic)
    first_token_ns: int | None  # perf_counter_ns of llm_first_token (ADR-0005)
    attempts: int


@dataclass(frozen=True)
class LLMUnavailable:
    reason: Reason
    detail: str
    attempts: int
    elapsed_ns: int


class _Failed(Exception):
    def __init__(self, reason: Reason, detail: str, *, got_token: bool) -> None:
        super().__init__(detail)
        self.reason: Reason = reason
        self.detail = detail
        self.got_token = got_token


def make_client(
    base_url: str = f"{vllm_server.BASE_URL}/v1", http_client: httpx2.AsyncClient | None = None
) -> AsyncOpenAI:
    """An AsyncOpenAI client for the local server: no SDK retries, a 1 s connect limit."""
    timeout = httpx2.Timeout(TOTAL_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
    return AsyncOpenAI(
        base_url=base_url,
        api_key="local-vllm-needs-no-key",
        max_retries=0,
        timeout=timeout,
        http_client=http_client,
    )


async def complete(
    client: AsyncOpenAI,
    messages: Iterable[ChatCompletionMessageParam],
    tools: Iterable[ChatCompletionToolUnionParam] | None = None,
) -> LLMReply | LLMUnavailable:
    """One greedy, streamed completion. Never raises for LLM trouble; does raise CancelledError."""
    start = time.perf_counter_ns()
    messages = list(messages)  # once: a retry must resend the same messages, even from a generator
    attempts = 0
    try:
        async with asyncio.timeout(TOTAL_TIMEOUT_S):
            while True:
                attempts += 1
                try:
                    return await _attempt(client, messages, tools, attempts)
                except _Failed as failed:
                    if failed.got_token or attempts == ATTEMPTS:
                        elapsed = time.perf_counter_ns() - start
                        return LLMUnavailable(failed.reason, failed.detail, attempts, elapsed)
    except TimeoutError:
        detail = f"no complete reply within {TOTAL_TIMEOUT_S} s"
        return LLMUnavailable("total", detail, attempts, time.perf_counter_ns() - start)


async def _attempt(
    client: AsyncOpenAI,
    messages: list[ChatCompletionMessageParam],
    tools: Iterable[ChatCompletionToolUnionParam] | None,
    attempt: int,
) -> LLMReply:
    first_deadline = asyncio.get_running_loop().time() + FIRST_TOKEN_TIMEOUT_S
    try:
        async with asyncio.timeout_at(first_deadline):
            stream = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=omit if tools is None else tools,
                stream=True,
                temperature=0,
                max_tokens=MAX_TOKENS,
            )
    except TimeoutError:
        raise _Failed("first_token", "no response headers in time", got_token=False) from None
    except APIStatusError as exc:
        raise _Failed("http_error", f"HTTP {exc.status_code}", got_token=False) from exc
    except APIConnectionError as exc:
        raise _Failed("connect", str(exc), got_token=False) from exc

    state = ChatCompletionStreamState()
    first_chunk_ns: int | None = None
    first_token_ns: int | None = None
    finish_reason: str | None = None
    async with stream:
        chunks = aiter(stream)
        while True:
            got_token = first_token_ns is not None
            if got_token:
                limit = asyncio.timeout(INTER_CHUNK_TIMEOUT_S)
            else:
                limit = asyncio.timeout_at(first_deadline)  # a role-only chunk doesn't reset it
            try:
                async with limit:
                    chunk = await anext(chunks)
            except StopAsyncIteration:
                break
            except TimeoutError:
                reason: Reason = "stalled" if got_token else "first_token"
                raise _Failed(reason, "no chunk in time", got_token=got_token) from None
            except APIConnectionError as exc:
                raise _Failed("dropped", str(exc), got_token=got_token) from exc
            except APIError as exc:  # vLLM's mid-stream SSE error event (after the subclass above)
                raise _Failed("http_error", str(exc), got_token=got_token) from exc
            now = time.perf_counter_ns()
            if first_chunk_ns is None:
                first_chunk_ns = now
            state.handle_chunk(chunk)
            for choice in chunk.choices:
                if first_token_ns is None and (choice.delta.content or choice.delta.tool_calls):
                    first_token_ns = now
                finish_reason = choice.finish_reason or finish_reason

    if finish_reason is None:
        got_token = first_token_ns is not None
        raise _Failed("dropped", "stream ended without a finish reason", got_token=got_token)
    message = state.get_final_completion().choices[0].message
    calls = tuple(
        ToolCall(call.id, call.function.name, call.function.arguments)
        for call in message.tool_calls or []
    )
    return LLMReply(
        message.content or "", calls, finish_reason, first_chunk_ns, first_token_ns, attempt
    )
