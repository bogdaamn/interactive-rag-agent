"""Per-user conversation history (bonus: conversation-aware RAG).

See spec/v3/SPEC.md §12. Keyed by user_id — deliberately NOT reusing
../telegram-bot/session.py, which is chat_id-keyed with flat JSONL files and
lists "no multi-user session isolation" as an explicit non-goal.

Lives in the same database file as documents so a user's conversation and their
documents are one thing to inspect, back up, or remove.
"""

import datetime
import sqlite3

from userdocs.config import CONVERSATION_HISTORY_TURNS
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
        self._pending = []  # a new question starts a new turn
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
