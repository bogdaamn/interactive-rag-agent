# Conversation-Aware RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the per-user conversation history that lets a follow-up question be interpreted against the previous turn — the assignment's "Сколько дней отпуска?" → "25 дней." → "А можно перенести **их** на следующий год?" example — as SQLite-backed, `user_id`-keyed, bounded-window state (bonus +2).

**Architecture:** `src/telegram_bot/session.py`. A `SessionStore` owns one SQLite connection and hands out a `Session` per `user_id`. Persisted history is only the conversational turns (user + assistant); the tool-call/tool-result messages of the turn currently in flight are held in memory and dropped when the next user message arrives. `messages()` returns `persisted_window + pending`, which is exactly what the agent loop needs: the LLM sees prior turns for pronoun resolution *and* this turn's tool output, but the database never accumulates per-call scaffolding.

**Tech Stack:** Python 3.10+, stdlib `sqlite3`. No new dependencies.

**Spec:** `spec/v3/SPEC.md` §12 (conversation-aware RAG), §11.2 (the agent loop this feeds)

**Depends on:** `docs/superpowers/plans/2026-09-10-rag-retrieval-tool.md` Task 10 — `userdocs.agent.run` duck-types against the `Session` interface built here (`messages`, `append_user_message`, `append_assistant_message`, `append_tool_result`). This plan can be implemented before or after the Telegram plan, but `telegram_bot/handlers/chat.py` (Telegram plan Task 8) and `telegram_bot/main.py` (Task 10) both import `SessionStore`, so **do this plan's Task 1 before the Telegram plan's Task 8** if running them out of order.

## Global Constraints

- Keyed by **`user_id`, not `chat_id`** — this is the one place `../telegram-bot/session.py` is deliberately *not* reused (spec §2's reuse table). That store is `chat_id`-keyed with "no multi-user isolation" as a stated non-goal, which is incompatible with this assignment's §10 requirement.
- `CONVERSATION_HISTORY_TURNS = 3` from `userdocs/config.py` → a window of the most recent 6 persisted messages (spec §12). Never unbounded: every prompt would grow forever otherwise.
- One user's history must never appear in another user's prompt — same hard isolation requirement as documents, with its own test.
- Tool-call scaffolding is **not persisted**. Only `role in ("user", "assistant")` messages go to the database.
- `messages()` must return a message sequence Ollama's `/api/chat` accepts: every `role: "tool"` message is immediately preceded by the `role: "assistant"` message carrying the `tool_calls` that produced it.
- Storage lives in the same database file as documents (`userdocs.db`) — one file to back up, and it means a user's conversation and their documents are deleted from the same place. `SessionStore` opens its own connection rather than sharing `UserDocsStore`'s, since the two are used from different call paths and sharing one `sqlite3.Connection` across them buys nothing.
- TDD, strictly: failing test → confirm the failure → minimal implementation → confirm green → commit.

---

### Task 1: `SessionStore` schema and message persistence

**Files:**
- Create: `src/telegram_bot/session.py`
- Create: `tests/telegram_bot/test_session.py`

**Interfaces:**
- Consumes: `userdocs.errors.SQLiteStoreError` (ingestion plan Task 1).
- Produces: `telegram_bot.session.SessionStore(db_path: str)` with `.get_or_create(user_id: int) -> Session` and `.close() -> None`; `telegram_bot.session.Session` with `.append_user_message(text: str) -> None`, `.append_assistant_message(text: str) -> None`, `.messages() -> list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/telegram_bot/test_session.py
from telegram_bot.session import SessionStore


def test_append_and_read_back_a_single_turn(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    session = store.get_or_create(user_id=1)

    session.append_user_message("How many vacation days?")
    session.append_assistant_message("25 days.")

    assert session.messages() == [
        {"role": "user", "content": "How many vacation days?"},
        {"role": "assistant", "content": "25 days."},
    ]
    store.close()


def test_messages_persist_across_store_reopen(tmp_path):
    db_path = str(tmp_path / "test.db")

    store = SessionStore(db_path)
    session = store.get_or_create(user_id=1)
    session.append_user_message("How many vacation days?")
    session.append_assistant_message("25 days.")
    store.close()

    store2 = SessionStore(db_path)
    session2 = store2.get_or_create(user_id=1)
    assert session2.messages() == [
        {"role": "user", "content": "How many vacation days?"},
        {"role": "assistant", "content": "25 days."},
    ]
    store2.close()


def test_get_or_create_returns_an_empty_session_for_a_new_user(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    assert store.get_or_create(user_id=99).messages() == []
    store.close()


def test_get_or_create_returns_the_same_session_object_for_the_same_user(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    assert store.get_or_create(user_id=1) is store.get_or_create(user_id=1)
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/telegram_bot/test_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telegram_bot.session'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/telegram_bot/session.py
"""Per-user conversation history (bonus: conversation-aware RAG).

See spec/v3/SPEC.md §12. Keyed by user_id — deliberately NOT reusing
../telegram-bot/session.py, which is chat_id-keyed with flat JSONL files and
lists "no multi-user session isolation" as an explicit non-goal.

Lives in the same database file as documents so a user's conversation and their
documents are one thing to inspect, back up, or remove.
"""

import datetime
import sqlite3

from userdocs.errors import SQLiteStoreError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_user_id
    ON conversation_messages(user_id);
"""


class Session:
    def __init__(self, conn: sqlite3.Connection, user_id: int):
        self._conn = conn
        self._user_id = user_id

    def _append(self, role: str, content: str) -> None:
        try:
            created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._conn.execute(
                "INSERT INTO conversation_messages (user_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (self._user_id, role, content, created_at),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise SQLiteStoreError(f"Failed to append {role} message: {exc}") from exc

    def append_user_message(self, text: str) -> None:
        self._append("user", text)

    def append_assistant_message(self, text: str) -> None:
        self._append("assistant", text)

    def messages(self) -> list:
        try:
            rows = self._conn.execute(
                "SELECT role, content FROM conversation_messages "
                "WHERE user_id = ? ORDER BY id ASC",
                (self._user_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise SQLiteStoreError(f"Failed to read conversation history: {exc}") from exc
        return [{"role": role, "content": content} for role, content in rows]


class SessionStore:
    def __init__(self, db_path: str):
        try:
            self._conn = sqlite3.connect(db_path)
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise SQLiteStoreError(f"Failed to initialize session storage: {exc}") from exc
        self._sessions: dict = {}

    def get_or_create(self, user_id: int) -> Session:
        if user_id not in self._sessions:
            self._sessions[user_id] = Session(self._conn, user_id)
        return self._sessions[user_id]

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/telegram_bot/test_session.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/telegram_bot/session.py tests/telegram_bot/test_session.py
git commit -m "feat(telegram_bot): add SQLite-backed per-user conversation history"
```

---

### Task 2: Per-user isolation of conversation history

**Files:**
- Modify: `tests/telegram_bot/test_session.py`

**Interfaces:**
- No production change expected — this task is a **verification task**: it proves the `WHERE user_id = ?` filter written in Task 1 actually holds, and locks it against regression. If the test fails, the fix goes in `session.py`.

> This is a legitimate standalone task rather than an extra assertion folded into Task 1: it's the specific behavior assignment §10 makes a hard requirement, and it deserves its own reviewable gate. Isolation bugs are exactly the kind that get introduced later by a well-meaning "optimize this query" change, so it needs a test whose failure message says *isolation broke*, not *history looks wrong*.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/telegram_bot/test_session.py
def test_one_users_history_never_appears_in_anothers(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))

    session_a = store.get_or_create(user_id=1)
    session_a.append_user_message("What is my salary?")
    session_a.append_assistant_message("Your salary is confidential information X.")

    session_b = store.get_or_create(user_id=2)
    session_b.append_user_message("What are the office hours?")

    b_contents = [m["content"] for m in session_b.messages()]
    assert b_contents == ["What are the office hours?"]
    assert not any("confidential information X" in c for c in b_contents)
    store.close()


def test_a_new_users_session_is_empty_even_when_others_have_history(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))

    session_a = store.get_or_create(user_id=1)
    session_a.append_user_message("first user's question")
    session_a.append_assistant_message("first user's answer")

    assert store.get_or_create(user_id=2).messages() == []
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/telegram_bot/test_session.py -v`
Expected: These two tests should **PASS immediately** against Task 1's implementation — the `WHERE user_id = ?` filter is already there. That is the expected and correct outcome for a verification task; do not weaken the implementation to manufacture a red phase. If either test *fails*, that's a real isolation bug in Task 1 — fix `session.py` and re-run before committing.

- [ ] **Step 3: Confirm no implementation change is needed**

Read `src/telegram_bot/session.py::Session.messages` and confirm the `WHERE user_id = ?` clause is present and that `user_id` comes from `self._user_id` (set at construction by `get_or_create`), never from a caller-supplied argument on `messages()`. No code change if so.

- [ ] **Step 4: Run the full session test file to verify everything passes**

Run: `pytest tests/telegram_bot/test_session.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/telegram_bot/test_session.py
git commit -m "test(telegram_bot): assert conversation history is isolated per user"
```

---

### Task 3: Bounded history window

**Files:**
- Modify: `src/telegram_bot/session.py`
- Modify: `tests/telegram_bot/test_session.py`

**Interfaces:**
- Consumes: `userdocs.config.CONVERSATION_HISTORY_TURNS` (ingestion plan Task 1).
- Produces: `Session.messages()` now returns at most `CONVERSATION_HISTORY_TURNS * 2` persisted messages, the most recent ones, still in chronological order.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/telegram_bot/test_session.py
from userdocs.config import CONVERSATION_HISTORY_TURNS


def test_messages_returns_at_most_the_configured_window(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    session = store.get_or_create(user_id=1)

    for turn in range(CONVERSATION_HISTORY_TURNS + 4):
        session.append_user_message(f"question {turn}")
        session.append_assistant_message(f"answer {turn}")

    messages = session.messages()
    assert len(messages) == CONVERSATION_HISTORY_TURNS * 2
    store.close()


def test_messages_window_keeps_the_most_recent_turns_in_order(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    session = store.get_or_create(user_id=1)

    for turn in range(CONVERSATION_HISTORY_TURNS + 4):
        session.append_user_message(f"question {turn}")
        session.append_assistant_message(f"answer {turn}")

    messages = session.messages()
    last_turn = CONVERSATION_HISTORY_TURNS + 3

    # oldest turns are dropped, newest are kept, chronological order preserved
    assert messages[-1] == {"role": "assistant", "content": f"answer {last_turn}"}
    assert messages[-2] == {"role": "user", "content": f"question {last_turn}"}
    assert not any(m["content"] == "question 0" for m in messages)
    store.close()


def test_messages_window_does_not_truncate_a_short_history(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    session = store.get_or_create(user_id=1)
    session.append_user_message("only question")
    session.append_assistant_message("only answer")

    assert len(session.messages()) == 2
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/telegram_bot/test_session.py -v`
Expected: FAIL — `test_messages_returns_at_most_the_configured_window` asserts `len == 6` but gets `14` (all messages, unwindowed)

- [ ] **Step 3: Write minimal implementation**

```python
# src/telegram_bot/session.py — add to imports
from userdocs.config import CONVERSATION_HISTORY_TURNS
```

```python
# src/telegram_bot/session.py — replace Session.messages with:

    def messages(self) -> list:
        """The most recent CONVERSATION_HISTORY_TURNS turns, oldest first.

        Bounded so a long conversation doesn't grow every prompt without limit
        (spec §12). Fetched newest-first with a LIMIT — cheaper than reading the
        whole history and slicing — then reversed back into chronological order,
        which is what the LLM expects.
        """
        window_size = CONVERSATION_HISTORY_TURNS * 2
        try:
            rows = self._conn.execute(
                "SELECT role, content FROM conversation_messages "
                "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (self._user_id, window_size),
            ).fetchall()
        except sqlite3.Error as exc:
            raise SQLiteStoreError(f"Failed to read conversation history: {exc}") from exc
        return [{"role": role, "content": content} for role, content in reversed(rows)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/telegram_bot/test_session.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add src/telegram_bot/session.py tests/telegram_bot/test_session.py
git commit -m "feat(telegram_bot): bound conversation history to a configured turn window"
```

---

### Task 4: Ephemeral tool messages for the in-flight turn

**Files:**
- Modify: `src/telegram_bot/session.py`
- Modify: `tests/telegram_bot/test_session.py`

**Interfaces:**
- Produces: `Session.append_tool_result(call: dict, result: str) -> None` — appends an in-memory `{"role": "assistant", "content": "", "tool_calls": [call]}` / `{"role": "tool", "content": result}` pair. `Session.messages()` now returns `persisted_window + pending`. `append_user_message` clears `pending` first.

> Why the assistant/tool **pair** rather than just the tool message: Ollama's `/api/chat` (and the OpenAI-shaped format it follows) requires a `role: "tool"` message to answer a preceding `role: "assistant"` message that carried the matching `tool_calls`. Emitting a bare tool message produces an invalid sequence some models reject outright and others silently mishandle. Pairing one assistant message per call keeps the sequence valid for any number of calls, and `userdocs/agent.py`'s existing `session.append_tool_result(call, result)` signature (retrieval plan Task 10) needs no change.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/telegram_bot/test_session.py
_CALL = {"function": {"name": "search_documents", "arguments": {"query": "vacation"}}}


def test_append_tool_result_makes_the_result_visible_to_the_next_llm_call(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    session = store.get_or_create(user_id=1)

    session.append_user_message("How many vacation days?")
    session.append_tool_result(_CALL, "[Source: policy.pdf, page 3]\n25 days")

    contents = [m["content"] for m in session.messages()]
    assert any("25 days" in c for c in contents)
    store.close()


def test_append_tool_result_pairs_each_tool_message_with_an_assistant_tool_call(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    session = store.get_or_create(user_id=1)

    session.append_user_message("How many vacation days?")
    session.append_tool_result(_CALL, "25 days")

    messages = session.messages()
    tool_index = next(i for i, m in enumerate(messages) if m["role"] == "tool")
    preceding = messages[tool_index - 1]
    assert preceding["role"] == "assistant"
    assert preceding["tool_calls"] == [_CALL]
    store.close()


def test_tool_messages_are_not_persisted(tmp_path):
    db_path = str(tmp_path / "test.db")

    store = SessionStore(db_path)
    session = store.get_or_create(user_id=1)
    session.append_user_message("How many vacation days?")
    session.append_tool_result(_CALL, "25 days")
    session.append_assistant_message("You get 25 days. Source: policy.pdf")
    store.close()

    store2 = SessionStore(db_path)
    roles = [m["role"] for m in store2.get_or_create(user_id=1).messages()]
    assert roles == ["user", "assistant"]
    assert "tool" not in roles
    store2.close()


def test_append_user_message_clears_the_previous_turns_tool_messages(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    session = store.get_or_create(user_id=1)

    session.append_user_message("How many vacation days?")
    session.append_tool_result(_CALL, "stale tool output from turn one")
    session.append_assistant_message("25 days.")

    session.append_user_message("Can I carry them over?")

    contents = [m["content"] for m in session.messages()]
    assert not any("stale tool output" in c for c in contents)
    # ...but the conversational turn itself is still there, which is the whole
    # point of conversation-aware RAG
    assert any("25 days." in c for c in contents)
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/telegram_bot/test_session.py -v`
Expected: FAIL with `AttributeError: 'Session' object has no attribute 'append_tool_result'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/telegram_bot/session.py — replace class Session with:

class Session:
    """One user's conversation.

    Persisted history holds only the conversational turns (user/assistant), so
    the database doesn't accumulate per-call scaffolding and a follow-up
    question sees clean prior context. The tool-call messages of the turn
    currently in flight are held in `_pending` and discarded when the next user
    message arrives — the agent loop needs them within a turn, but they're
    noise across turns.
    """

    def __init__(self, conn: sqlite3.Connection, user_id: int):
        self._conn = conn
        self._user_id = user_id
        self._pending: list = []

    def _append(self, role: str, content: str) -> None:
        try:
            created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._conn.execute(
                "INSERT INTO conversation_messages (user_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (self._user_id, role, content, created_at),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise SQLiteStoreError(f"Failed to append {role} message: {exc}") from exc

    def append_user_message(self, text: str) -> None:
        self._pending = []   # a new question starts a new turn
        self._append("user", text)

    def append_assistant_message(self, text: str) -> None:
        self._append("assistant", text)

    def append_tool_result(self, call: dict, result: str) -> None:
        """In-memory only. The assistant/tool pairing is required by Ollama's
        chat format: a tool message must answer a preceding assistant message
        that carried the matching tool_calls."""
        self._pending.append({"role": "assistant", "content": "", "tool_calls": [call]})
        self._pending.append({"role": "tool", "content": result})

    def messages(self) -> list:
        """The most recent CONVERSATION_HISTORY_TURNS persisted turns (oldest
        first), followed by this turn's in-flight tool messages.

        Bounded so a long conversation doesn't grow every prompt without limit
        (spec §12). Fetched newest-first with a LIMIT — cheaper than reading the
        whole history and slicing — then reversed back into chronological order,
        which is what the LLM expects.
        """
        window_size = CONVERSATION_HISTORY_TURNS * 2
        try:
            rows = self._conn.execute(
                "SELECT role, content FROM conversation_messages "
                "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (self._user_id, window_size),
            ).fetchall()
        except sqlite3.Error as exc:
            raise SQLiteStoreError(f"Failed to read conversation history: {exc}") from exc
        persisted = [{"role": role, "content": content} for role, content in reversed(rows)]
        return persisted + list(self._pending)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/telegram_bot/test_session.py -v`
Expected: PASS (13 passed)

- [ ] **Step 5: Commit**

```bash
git add src/telegram_bot/session.py tests/telegram_bot/test_session.py
git commit -m "feat(telegram_bot): keep in-flight tool messages ephemeral and Ollama-valid"
```

---

### Task 5: Follow-up question resolution through the agent loop

**Files:**
- Create: `tests/telegram_bot/test_conversation_aware_rag.py`

**Interfaces:**
- Consumes: `telegram_bot.session.SessionStore` (Tasks 1–4), `userdocs.agent.run` (retrieval plan Task 10), `userdocs.tool_registry.{Tool, ToolRegistry}` (retrieval plan Task 8).
- Produces: no new production code — this is the integration test that proves the bonus feature actually works end-to-end at the boundary that matters (what the LLM is shown on the follow-up turn).

> This is the acceptance test for **bonus +2**. Tasks 1–4 build the mechanism; without this test nothing verifies that the mechanism is actually reaching the LLM on the second question, which is the only thing the bonus is scored on.

- [ ] **Step 1: Write the failing test**

```python
# tests/telegram_bot/test_conversation_aware_rag.py
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

    llm_turn2 = RecordingLLM([{"content": "Да.", "tool_calls": []}])
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
    llm = RecordingLLM([{"content": "The office opens at 9.", "tool_calls": []}])
    session_b.append_user_message("What are the office hours?")
    await agent_run(llm, _registry(), session_b, "What are the office hours?")

    contents = [m.get("content", "") for m in llm.seen_messages[0]]
    assert not any("Band 7" in c for c in contents)
    assert not any("confidential_comp.pdf" in c for c in contents)
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/telegram_bot/test_conversation_aware_rag.py -v`
Expected: All three should **PASS** if Tasks 1–4 and the retrieval plan's Task 10 are correct — this test verifies the composition of already-built pieces rather than driving new code. If any fails, that's a real integration bug: fix `session.py` or `userdocs/agent.py` before committing, and note which. Most likely failure mode to watch for: `agent.run` prepending `SYSTEM_PROMPT` but a caller also appending the user message twice, or `_pending` not being cleared between turns.

- [ ] **Step 3: Fix any integration bug the test exposes**

If all three pass, no change. If one fails, the fix belongs in `src/telegram_bot/session.py` or `src/userdocs/agent.py` — do not weaken the test to match the behavior.

- [ ] **Step 4: Run the full suite for both packages**

Run: `pytest tests/userdocs tests/telegram_bot -v`
Expected: PASS (everything from this plan plus the ingestion, retrieval, and Telegram plans)

- [ ] **Step 5: Commit**

```bash
git add tests/telegram_bot/test_conversation_aware_rag.py
git commit -m "test(telegram_bot): verify follow-up questions resolve against prior turns"
```

---

## Summary

**Bonus +2 (conversation-aware RAG)** is complete and traceable:

| Piece | Where |
|---|---|
| `user_id`-keyed SQLite conversation store | `src/telegram_bot/session.py` (Task 1) |
| Isolation between users' histories | Task 2's tests + `WHERE user_id = ?` |
| Bounded window (3 turns / 6 messages) | Task 3 |
| Ollama-valid ephemeral tool messages | Task 4 |
| The assignment's own follow-up example, verified | `tests/telegram_bot/test_conversation_aware_rag.py` (Task 5) |

Two of the five tasks here are verification tasks with no production change (Tasks 2 and 5). That's intentional — they cover the assignment's hard isolation requirement and the bonus's actual scored behavior, and both are the kind of property a later refactor breaks silently without a test whose name says exactly what went wrong.

Remaining: `docs/superpowers/plans/2026-09-10-testing-and-evaluation.md` (end-to-end pipeline test, the 5-question RAG evaluation dataset and runner, and the README sections assignment §17 requires).
