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
