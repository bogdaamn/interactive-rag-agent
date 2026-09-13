import pytest

from telegram_bot.errors import ERROR_MESSAGES
from telegram_bot.handlers.chat import handle_text
from telegram_bot.llm_errors import LLMError, LLMTimeoutError
from telegram_bot.stats import UsageStats
from userdocs.errors import SQLiteStoreError


class FakeMessage:
    def __init__(self, text="how many vacation days?", user_id=1):
        self.text = text
        self.from_user = type("FakeUser", (), {"id": user_id})()
        self.sent = []

    async def answer(self, text):
        self.sent.append(text)


class FakeSession:
    def __init__(self, fail_on=None):
        self.appended = []
        self._fail_on = fail_on

    def messages(self):
        return []

    def append_user_message(self, text):
        if self._fail_on == "user":
            raise SQLiteStoreError("disk full")
        self.appended.append(("user", text))

    def append_assistant_message(self, text):
        if self._fail_on == "assistant":
            raise SQLiteStoreError("disk full")
        self.appended.append(("assistant", text))

    def append_tool_result(self, call, result):
        self.appended.append(("tool", result))


class FakeSessionStore:
    def __init__(self):
        self.sessions = {}

    def get_or_create(self, user_id):
        return self.sessions.setdefault(user_id, FakeSession())


@pytest.mark.asyncio
async def test_handle_text_replies_with_the_agents_answer(monkeypatch):
    import telegram_bot.handlers.chat as chat_module

    async def fake_run(llm, registry, session, user_text, max_steps=4, on_llm_usage=None):
        return "You get 25 days. Source: policy.pdf"

    monkeypatch.setattr(chat_module, "agent_run", fake_run)

    message = FakeMessage()
    sessions = FakeSessionStore()
    await handle_text(message, store=None, llm=None, sessions=sessions, stats=UsageStats())

    assert message.sent == ["You get 25 days. Source: policy.pdf"]


@pytest.mark.asyncio
async def test_handle_text_records_both_sides_of_the_turn_in_the_session(monkeypatch):
    import telegram_bot.handlers.chat as chat_module

    async def fake_run(llm, registry, session, user_text, max_steps=4, on_llm_usage=None):
        return "25 days."

    monkeypatch.setattr(chat_module, "agent_run", fake_run)

    message = FakeMessage(text="how many vacation days?", user_id=9)
    sessions = FakeSessionStore()
    await handle_text(message, store=None, llm=None, sessions=sessions, stats=UsageStats())

    session = sessions.sessions[9]
    assert ("user", "how many vacation days?") in session.appended
    assert ("assistant", "25 days.") in session.appended


@pytest.mark.asyncio
async def test_handle_text_builds_a_tool_registry_bound_to_the_sender(monkeypatch):
    import telegram_bot.handlers.chat as chat_module

    seen = {}

    def fake_build_tool(store, user_id):
        seen["user_id"] = user_id
        return type("T", (), {"name": "search_documents", "schema": lambda self: {}})()

    async def fake_run(llm, registry, session, user_text, max_steps=4, on_llm_usage=None):
        return "ok"

    monkeypatch.setattr(chat_module, "build_search_documents_tool", fake_build_tool)
    monkeypatch.setattr(chat_module, "agent_run", fake_run)

    message = FakeMessage(user_id=4242)
    await handle_text(message, store=None, llm=None, sessions=FakeSessionStore(), stats=UsageStats())

    assert seen["user_id"] == 4242


@pytest.mark.parametrize("error_type", [LLMError, LLMTimeoutError])
@pytest.mark.asyncio
async def test_handle_text_replies_with_the_mapped_message_on_llm_failure(monkeypatch, error_type):
    import telegram_bot.handlers.chat as chat_module

    async def failing_run(llm, registry, session, user_text, max_steps=4, on_llm_usage=None):
        raise error_type("connection refused to ollama at localhost:11434")

    monkeypatch.setattr(chat_module, "agent_run", failing_run)

    message = FakeMessage()
    await handle_text(message, store=None, llm=None, sessions=FakeSessionStore(), stats=UsageStats())

    assert message.sent == [ERROR_MESSAGES[error_type]]
    assert not any("localhost:11434" in text for text in message.sent)


@pytest.mark.asyncio
async def test_handle_text_does_not_record_an_assistant_message_when_the_llm_fails(monkeypatch):
    """A failed turn must not pollute the conversation history with a
    non-answer, or the next follow-up question inherits it as context."""
    import telegram_bot.handlers.chat as chat_module

    async def failing_run(llm, registry, session, user_text, max_steps=4, on_llm_usage=None):
        raise LLMError("boom")

    monkeypatch.setattr(chat_module, "agent_run", failing_run)

    message = FakeMessage(user_id=3)
    sessions = FakeSessionStore()
    await handle_text(message, store=None, llm=None, sessions=sessions, stats=UsageStats())

    roles = [role for role, _text in sessions.sessions[3].appended]
    assert "assistant" not in roles


@pytest.mark.asyncio
async def test_handle_text_records_turn_usage_in_stats(monkeypatch):
    import telegram_bot.handlers.chat as chat_module

    async def fake_run(llm, registry, session, user_text, max_steps=4, on_llm_usage=None):
        on_llm_usage(100, 20)
        on_llm_usage(50, 10)
        return "25 days."

    monkeypatch.setattr(chat_module, "agent_run", fake_run)

    message = FakeMessage(user_id=9)
    stats = UsageStats()
    await handle_text(message, store=None, llm=None, sessions=FakeSessionStore(), stats=stats)

    snapshot = stats.snapshot(9)
    assert snapshot.prompt_tokens == 150
    assert snapshot.completion_tokens == 30
    assert snapshot.turns == 1


@pytest.mark.asyncio
async def test_handle_text_records_an_error_in_stats_on_llm_failure(monkeypatch):
    import telegram_bot.handlers.chat as chat_module

    async def failing_run(llm, registry, session, user_text, max_steps=4, on_llm_usage=None):
        raise LLMError("boom")

    monkeypatch.setattr(chat_module, "agent_run", failing_run)

    message = FakeMessage(user_id=3)
    stats = UsageStats()
    await handle_text(message, store=None, llm=None, sessions=FakeSessionStore(), stats=stats)

    assert stats.snapshot(3).errors == {"LLMError": 1}


@pytest.mark.asyncio
async def test_handle_text_replies_with_the_mapped_message_on_sqlite_failure(monkeypatch):
    """A DB failure while persisting the user's turn must not leave the user
    with total silence — it needs the same mapped-message treatment as an
    LLM failure."""
    import telegram_bot.handlers.chat as chat_module

    async def fake_run(llm, registry, session, user_text, max_steps=4, on_llm_usage=None):
        return "25 days."

    monkeypatch.setattr(chat_module, "agent_run", fake_run)

    message = FakeMessage(user_id=3)
    sessions = FakeSessionStore()
    sessions.sessions[3] = FakeSession(fail_on="user")
    stats = UsageStats()

    await handle_text(message, store=None, llm=None, sessions=sessions, stats=stats)

    assert message.sent == [ERROR_MESSAGES[SQLiteStoreError]]
    assert stats.snapshot(3).errors == {"SQLiteStoreError": 1}
    assert stats.snapshot(3).turns == 0


@pytest.mark.asyncio
async def test_handle_text_replies_with_the_mapped_message_when_saving_the_answer_fails(monkeypatch):
    """Same failure mode, but the DB error happens after a successful agent
    run, while persisting the assistant's reply."""
    import telegram_bot.handlers.chat as chat_module

    async def fake_run(llm, registry, session, user_text, max_steps=4, on_llm_usage=None):
        return "25 days."

    monkeypatch.setattr(chat_module, "agent_run", fake_run)

    message = FakeMessage(user_id=3)
    sessions = FakeSessionStore()
    sessions.sessions[3] = FakeSession(fail_on="assistant")
    stats = UsageStats()

    await handle_text(message, store=None, llm=None, sessions=sessions, stats=stats)

    assert message.sent == [ERROR_MESSAGES[SQLiteStoreError]]
    assert stats.snapshot(3).errors == {"SQLiteStoreError": 1}
    assert stats.snapshot(3).turns == 0
