"""Acceptance test for conversation-aware RAG (bonus +2).

Reproduces the assignment's own example:
    User:  Сколько дней отпуска предусмотрено?
    Agent: 25 дней.
    User:  А можно перенести их на следующий год?

The second question is only answerable if the prior turn is visible to the LLM.
The LLM is faked — what's under test is what the agent loop SHOWS it, not
whether a particular model resolves the pronoun correctly.
"""

import pytest

from telegram_bot.session import SessionStore
from userdocs.agent import run as agent_run
from userdocs.tool_registry import Tool, ToolRegistry


class RecordingLLM:
    """Records the exact `messages` list it was handed on each chat() call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.seen_messages = []

    async def chat(self, messages, tools):
        self.seen_messages.append([dict(m) for m in messages])
        return self._responses.pop(0)


def _registry(tool_result="25 дней"):
    async def handler(query: str) -> str:
        return tool_result

    return ToolRegistry([
        Tool(
            name="search_documents",
            description="d",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=handler,
        )
    ])


@pytest.mark.asyncio
async def test_followup_question_sees_the_previous_turn(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    session = store.get_or_create(user_id=1)

    # --- Turn 1 --------------------------------------------------------------
    llm_turn1 = RecordingLLM([
        {"content": "", "tool_calls": [
            {"function": {"name": "search_documents", "arguments": {"query": "дни отпуска"}}}
        ]},
        {"content": "25 дней. Источник: vacation_policy.pdf", "tool_calls": []},
    ])
    first_question = "Сколько дней отпуска предусмотрено?"
    session.append_user_message(first_question)
    answer1 = await agent_run(llm_turn1, _registry(), session, first_question)
    session.append_assistant_message(answer1)

    # --- Turn 2 --------------------------------------------------------------
    llm_turn2 = RecordingLLM([
        {"content": "Да, до 5 дней. Источник: vacation_policy.pdf", "tool_calls": []},
        {"content": "Да, до 5 дней. Источник: vacation_policy.pdf", "tool_calls": []},
    ])
    followup = "А можно перенести их на следующий год?"
    session.append_user_message(followup)
    await agent_run(llm_turn2, _registry(), session, followup)

    # The follow-up prompt must contain BOTH the earlier question and the
    # earlier answer — that's what lets "их" resolve to "отпускные дни".
    followup_prompt = llm_turn2.seen_messages[0]
    contents = [m.get("content", "") for m in followup_prompt]

    assert any(first_question in c for c in contents), "prior question missing from follow-up prompt"
    assert any("25 дней" in c for c in contents), "prior answer missing from follow-up prompt"
    assert any(followup in c for c in contents), "the follow-up question itself is missing"
    store.close()


@pytest.mark.asyncio
async def test_followup_prompt_does_not_carry_the_previous_turns_tool_output(tmp_path):
    """Prior turns contribute their conversational content, not the raw
    retrieved chunks — otherwise every follow-up prompt grows by a full Top-K
    of chunk text per earlier turn."""
    store = SessionStore(str(tmp_path / "test.db"))
    session = store.get_or_create(user_id=1)

    llm_turn1 = RecordingLLM([
        {"content": "", "tool_calls": [
            {"function": {"name": "search_documents", "arguments": {"query": "q"}}}
        ]},
        {"content": "25 дней.", "tool_calls": []},
    ])
    session.append_user_message("Сколько дней отпуска?")
    answer1 = await agent_run(
        llm_turn1, _registry("VERBATIM_CHUNK_TEXT_FROM_TURN_ONE"), session, "Сколько дней отпуска?"
    )
    session.append_assistant_message(answer1)

    llm_turn2 = RecordingLLM([{"content": "Да.", "tool_calls": []}, {"content": "Да.", "tool_calls": []}])
    session.append_user_message("А перенести их можно?")
    await agent_run(llm_turn2, _registry(), session, "А перенести их можно?")

    contents = [m.get("content", "") for m in llm_turn2.seen_messages[0]]
    assert not any("VERBATIM_CHUNK_TEXT_FROM_TURN_ONE" in c for c in contents)
    store.close()


@pytest.mark.asyncio
async def test_one_users_turns_never_appear_in_anothers_prompt(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))

    session_a = store.get_or_create(user_id=1)
    session_a.append_user_message("What is my salary band?")
    session_a.append_assistant_message("Band 7. Источник: confidential_comp.pdf")

    session_b = store.get_or_create(user_id=2)
    llm = RecordingLLM([
        {"content": "The office opens at 9.", "tool_calls": []},
        {"content": "The office opens at 9.", "tool_calls": []},
    ])
    session_b.append_user_message("What are the office hours?")
    await agent_run(llm, _registry(), session_b, "What are the office hours?")

    contents = [m.get("content", "") for m in llm.seen_messages[0]]
    assert not any("Band 7" in c for c in contents)
    assert not any("confidential_comp.pdf" in c for c in contents)
    store.close()
