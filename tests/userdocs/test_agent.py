import asyncio

from userdocs.agent import SYSTEM_PROMPT, run
from userdocs.tool_registry import Tool, ToolRegistry


class FakeSession:
    def __init__(self):
        self._messages = []

    def messages(self):
        return list(self._messages)

    def append_user_message(self, text):
        self._messages.append({"role": "user", "content": text})

    def append_assistant_message(self, text):
        self._messages.append({"role": "assistant", "content": text})

    def append_tool_result(self, call, result):
        self._messages.append({"role": "tool", "content": result})


class ScriptedLLM:
    """Returns a pre-scripted sequence of assistant messages, one per chat()."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def chat(self, messages, tools):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self._responses.pop(0)


def _echo_registry(result_text="the tool result"):
    async def handler(query: str) -> str:
        return result_text

    return ToolRegistry([
        Tool(
            name="search_documents",
            description="d",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            handler=handler,
        )
    ])


def test_run_returns_content_directly_when_no_tool_calls():
    llm = ScriptedLLM([{"content": "Hello!", "tool_calls": []}])
    session = FakeSession()

    answer = asyncio.run(run(llm, _echo_registry(), session, "hi"))

    assert answer == "Hello!"
    assert len(llm.calls) == 1


def test_run_invokes_tool_then_returns_the_followup_answer():
    llm = ScriptedLLM([
        {
            "content": "",
            "tool_calls": [
                {"function": {"name": "search_documents", "arguments": {"query": "vacation"}}}
            ],
        },
        {"content": "You get 25 days. Source: policy.pdf", "tool_calls": []},
    ])
    session = FakeSession()

    answer = asyncio.run(run(llm, _echo_registry("25 days"), session, "how many vacation days?"))

    assert answer == "You get 25 days. Source: policy.pdf"
    assert len(llm.calls) == 2
    # the tool's result was fed back into the conversation before the 2nd call
    assert any(m["role"] == "tool" for m in llm.calls[1]["messages"])


def test_run_prepends_the_system_prompt_on_the_first_call():
    llm = ScriptedLLM([{"content": "ok", "tool_calls": []}])
    session = FakeSession()

    asyncio.run(run(llm, _echo_registry(), session, "hi"))

    first_message = llm.calls[0]["messages"][0]
    assert first_message["role"] == "system"
    assert first_message["content"] == SYSTEM_PROMPT


def test_run_stops_at_max_steps_and_returns_a_fallback_message():
    # The LLM never stops asking for tools; the step budget must break the loop.
    endless_tool_call = {
        "content": "",
        "tool_calls": [
            {"function": {"name": "search_documents", "arguments": {"query": "x"}}}
        ],
    }
    llm = ScriptedLLM([endless_tool_call] * 3)
    session = FakeSession()

    answer = asyncio.run(run(llm, _echo_registry(), session, "hi", max_steps=3))

    assert len(llm.calls) == 3
    assert "rephrasing" in answer.lower()


def test_system_prompt_instructs_against_answering_from_general_knowledge():
    lowered = SYSTEM_PROMPT.lower()
    assert "general knowledge" in lowered
    assert "search_documents" in lowered


def test_system_prompt_lists_the_bot_commands():
    lowered = SYSTEM_PROMPT.lower()
    assert "/documents" in lowered
    assert "/delete" in lowered


def test_run_reports_llm_usage_via_callback_once_per_llm_call():
    llm = ScriptedLLM([
        {
            "content": "",
            "tool_calls": [
                {"function": {"name": "search_documents", "arguments": {"query": "vacation"}}}
            ],
            "prompt_tokens": 100,
            "completion_tokens": 20,
        },
        {
            "content": "You get 25 days. Source: policy.pdf",
            "tool_calls": [],
            "prompt_tokens": 150,
            "completion_tokens": 30,
        },
    ])
    session = FakeSession()
    usage_calls = []

    asyncio.run(
        run(
            llm,
            _echo_registry("25 days"),
            session,
            "how many vacation days?",
            on_llm_usage=lambda prompt_tokens, completion_tokens: usage_calls.append(
                (prompt_tokens, completion_tokens)
            ),
        )
    )

    assert usage_calls == [(100, 20), (150, 30)]
