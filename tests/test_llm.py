"""Unit tests for agent/llm.py: one streamed chat completion with hard time limits (PLAN M3.1).

The server is faked with httpx2.MockTransport (the transport layer of the openai SDK), serving
canned SSE the way vLLM does: a role-only first chunk, then text or tool-call deltas, then a chunk
with finish_reason, then [DONE]. Timeouts are shrunk so the timing tests run in well under a second.
Cyril's rules: first token = first text or tool-call delta; at most one retry, only if nothing was
received; a barge-in cancellation is never turned into an LLMUnavailable.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import httpx2
import pytest

from agent import llm

Part = str | float | Exception  # SSE text, a pause in seconds, or an error raised mid-stream


def sse(delta: dict[str, object], finish: str | None = None) -> str:
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    body = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "llm"}
    return "data: " + json.dumps({**body, "choices": [choice]}) + "\n\n"


ROLE = sse({"role": "assistant", "content": ""})
DONE = "data: [DONE]\n\n"
TEXT = [ROLE, sse({"content": "नमस्ते"}), sse({"content": " जी"}), sse({}, "stop"), DONE]
DROP = httpx2.RemoteProtocolError("peer closed connection")


class FakeServer:
    """Serves one canned response per request, in order; records requests and HTTP responses."""

    def __init__(self, *responses: Sequence[Part] | Exception | int) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []
        self.sent: list[httpx2.Response] = []

    async def _body(self, parts: Sequence[Part]) -> AsyncIterator[bytes]:
        for part in parts:
            if isinstance(part, str):
                yield part.encode()
            elif isinstance(part, Exception):
                raise part
            else:
                await asyncio.sleep(part)

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, int):
            return httpx2.Response(response, json={"error": "busy"})
        headers = {"content-type": "text/event-stream"}
        return httpx2.Response(200, headers=headers, content=self._body(response))

    def client(self) -> object:
        async def record(response: httpx2.Response) -> None:
            self.sent.append(response)

        transport = httpx2.MockTransport(self.handle)
        http = httpx2.AsyncClient(transport=transport, event_hooks={"response": [record]})
        return llm.make_client("http://vllm.test/v1", http)


@pytest.fixture(autouse=True)
def short_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "FIRST_TOKEN_TIMEOUT_S", 0.2)
    monkeypatch.setattr(llm, "INTER_CHUNK_TIMEOUT_S", 0.2)
    monkeypatch.setattr(llm, "TOTAL_TIMEOUT_S", 1.0)


def ask(server: FakeServer, tools: list[dict[str, object]] | None = None) -> object:
    messages = [{"role": "user", "content": "kal shaam Sharma ji"}]
    return asyncio.run(llm.complete(server.client(), messages, tools))  # type: ignore[arg-type]


def test_text_reply() -> None:
    server = FakeServer(TEXT)
    reply = ask(server)
    assert isinstance(reply, llm.LLMReply)
    assert (reply.text, reply.tool_calls, reply.finish_reason, reply.attempts) == (
        "नमस्ते जी",
        (),
        "stop",
        1,
    )
    assert reply.first_chunk_ns is not None and reply.first_token_ns is not None


def test_split_tool_call_deltas_are_joined() -> None:
    call = {"index": 0, "id": "call_1", "type": "function"}
    tool_parts = [
        ROLE,
        sse({"tool_calls": [{**call, "function": {"name": "check_slots", "arguments": ""}}]}),
        sse({"tool_calls": [{"index": 0, "function": {"arguments": '{"doctor": '}}]}),
        sse({"tool_calls": [{"index": 0, "function": {"arguments": '"Sharma"}'}}]}),
        sse({}, "tool_calls"),
        DONE,
    ]
    reply = ask(FakeServer(tool_parts), tools=[{"type": "function", "function": {"name": "x"}}])
    assert isinstance(reply, llm.LLMReply)
    assert reply.tool_calls == (llm.ToolCall("call_1", "check_slots", '{"doctor": "Sharma"}'),)
    assert reply.finish_reason == "tool_calls"


def test_request_is_greedy_streamed_and_capped() -> None:
    server = FakeServer(TEXT)
    tools = [{"type": "function", "function": {"name": "check_slots"}}]
    ask(server, tools)
    sent = server.requests[0]
    assert (sent["model"], sent["stream"], sent["temperature"]) == ("llm", True, 0)
    assert sent["max_tokens"] == llm.MAX_TOKENS and sent["tools"] == tools


def test_role_only_chunk_is_first_chunk_not_first_token() -> None:
    reply = ask(FakeServer([ROLE, 0.1, *TEXT[1:]]))
    assert isinstance(reply, llm.LLMReply)
    assert reply.first_chunk_ns is not None and reply.first_token_ns is not None
    assert reply.first_token_ns - reply.first_chunk_ns >= 90_000_000  # the 0.1 s pause


def test_role_only_chunk_does_not_reset_the_first_token_window() -> None:
    # The role chunk arrives at once, the text after 0.15 + 0.15 s: past the 0.2 s window.
    slow = [ROLE, 0.15, sse({}), 0.15, *TEXT[1:]]
    reply = ask(FakeServer(slow, slow))
    assert isinstance(reply, llm.LLMUnavailable) and reply.reason == "first_token"


def test_slow_first_token_is_retried_once_then_given_up() -> None:
    server = FakeServer([0.3, *TEXT], [0.3, *TEXT])
    reply = ask(server)
    assert isinstance(reply, llm.LLMUnavailable)
    assert (reply.reason, reply.attempts, len(server.requests)) == ("first_token", 2, 2)


def test_retry_succeeds_when_the_first_attempt_sent_nothing() -> None:
    reply = ask(FakeServer([0.3, *TEXT], TEXT))
    assert isinstance(reply, llm.LLMReply) and (reply.text, reply.attempts) == ("नमस्ते जी", 2)


def test_retry_resends_the_same_messages_when_given_a_generator() -> None:
    server = FakeServer([0.3, *TEXT], TEXT)
    messages = ({"role": "user", "content": text} for text in ["kal", "shaam"])
    asyncio.run(llm.complete(server.client(), messages))  # type: ignore[arg-type]
    assert server.requests[0]["messages"] == server.requests[1]["messages"]
    assert len(server.requests[1]["messages"]) == 2  # type: ignore[arg-type]


def test_stall_after_a_token_is_not_retried() -> None:
    server = FakeServer([*TEXT[:2], 0.3, *TEXT[2:]], TEXT)
    reply = ask(server)
    assert isinstance(reply, llm.LLMUnavailable)
    assert (reply.reason, reply.attempts, len(server.requests)) == ("stalled", 1, 1)


def test_drop_after_a_token_is_not_retried() -> None:
    server = FakeServer([*TEXT[:2], DROP], TEXT)
    reply = ask(server)
    assert isinstance(reply, llm.LLMUnavailable)
    assert (reply.reason, reply.attempts, len(server.requests)) == ("dropped", 1, 1)


def test_drop_before_any_token_is_retried() -> None:
    server = FakeServer([ROLE, DROP], [ROLE, DROP])
    reply = ask(server)
    assert isinstance(reply, llm.LLMUnavailable)
    assert (reply.reason, reply.attempts, len(server.requests)) == ("dropped", 2, 2)


def test_stream_that_ends_without_finishing_is_dropped() -> None:
    reply = ask(FakeServer(TEXT[:2]))  # a token, then EOF with no finish_reason and no [DONE]
    assert isinstance(reply, llm.LLMUnavailable) and reply.reason == "dropped"


@pytest.mark.parametrize(
    ("failure", "reason"),
    [(httpx2.ConnectError("refused"), "connect"), (503, "http_error")],
)
def test_server_unreachable_or_erroring(failure: Exception | int, reason: str) -> None:
    server = FakeServer(failure, failure)
    reply = ask(server)
    assert isinstance(reply, llm.LLMUnavailable)
    assert (reply.reason, reply.attempts) == (reason, 2)


def test_total_budget_caps_a_reply_that_keeps_trickling() -> None:
    trickle: list[Part] = [ROLE]
    for _ in range(12):  # a token every 0.1 s: never stalls, but runs past the 1.0 s total
        trickle += [0.1, sse({"content": "."})]
    server = FakeServer([*trickle, sse({}, "stop"), DONE])
    reply = ask(server)
    assert isinstance(reply, llm.LLMUnavailable) and (reply.reason, reply.attempts) == ("total", 1)


def test_total_budget_cuts_the_retry_short(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "TOTAL_TIMEOUT_S", 0.3)
    server = FakeServer([0.3, *TEXT], [0.3, *TEXT])  # attempt 1 fails at 0.2 s; 0.1 s left
    reply = ask(server)
    assert isinstance(reply, llm.LLMUnavailable) and (reply.reason, reply.attempts) == ("total", 2)


def test_barge_in_cancellation_propagates_and_closes_the_stream() -> None:
    # CLAUDE.md rule 7: a cancelled turn must not look like an ordinary LLM failure.
    server = FakeServer([*TEXT[:2], 0.5, *TEXT[2:]])

    async def barge_in() -> None:
        task = asyncio.create_task(llm.complete(server.client(), [], None))  # type: ignore[arg-type]
        await asyncio.sleep(0.05)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(barge_in())
    # The HTTP response is released, not left holding a pooled connection. (The SDK also closes
    # it on its own here, so removing `async with stream` is an equivalent mutant.)
    assert len(server.sent) == 1 and server.sent[0].is_closed


def test_client_never_retries_on_its_own() -> None:
    client = llm.make_client("http://vllm.test/v1")
    assert client.max_retries == 0
    assert client.timeout.connect == llm.CONNECT_TIMEOUT_S  # type: ignore[union-attr]


def test_timing_constants_are_marked_initial() -> None:
    source = Path(llm.__file__).read_text(encoding="utf-8")
    assert "initial; revisit after M4.5" in source
