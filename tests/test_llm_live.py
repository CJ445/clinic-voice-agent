"""Live check on the RTX 4060: vLLM's hermes parser turns a Qwen tool call into tool_calls (M3.1).

Reading vLLM 0.19.1 and the chat templates says `--tool-call-parser hermes` fits the Qwen
<tool_call> JSON format; this test proves it end to end, through agent/serve.py's utilization
lookup, agent/vllm_server.py's tools=True command line and agent/llm.py's streaming client.
It checks the parser, not the model's judgement: only that a check_slots call arrives with
arguments that validate. Marked gpu, so CI skips it.
Run: uv run pytest -m gpu tests/test_llm_live.py
"""

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent import llm, serve, vllm_server
from clinic.tools import CheckSlotsIn, tool_schemas

pytestmark = pytest.mark.gpu

REPO = "cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit"
REVISION = "b37d1ccb77e27a70fe4222d995fda27940eead5a"


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    chosen = serve.choose_util(REPO, REVISION, override=None, records=serve.RECORDS)
    log = Path(tmp_path_factory.mktemp("vllm")) / "vllm.log"
    proc = vllm_server.start(REPO, REVISION, chosen.util, log, tools=True)
    try:
        vllm_server.wait_ready(proc)
        yield
    finally:
        vllm_server.stop(proc)


def test_hermes_parses_a_check_slots_call(server: None) -> None:
    tools = [{"type": "function", "function": schema} for schema in tool_schemas().values()]
    messages = [
        {"role": "system", "content": "You book clinic appointments. Today is 2026-11-25 (IST)."},
        {"role": "user", "content": "Dr. Sharma ke paas 30 November 2026 ko koi slot hai?"},
    ]
    reply = asyncio.run(llm.complete(llm.make_client(), messages, tools))  # type: ignore[arg-type]
    assert isinstance(reply, llm.LLMReply), reply
    assert reply.finish_reason == "tool_calls", reply
    assert [call.name for call in reply.tool_calls] == ["check_slots"]
    CheckSlotsIn.model_validate_json(reply.tool_calls[0].arguments)  # raises if it isn't valid
    assert reply.first_token_ns is not None
