"""SQLite + sqlite-vec storage layer. See spec/v3/SPEC.md §7.

This is the ONLY module that runs SQL against userdocs.db. Every method that
reads or writes documents/chunks/chunk_vectors takes an explicit user_id and
narrows to it before touching chunks/chunk_vectors — see search_vectors and
delete_document below for the two-step "documents WHERE user_id, then chunks
WHERE document_id IN (...)" pattern this hard-requires (spec §10).
"""

import datetime
import sqlite3

import sqlite_vec

from userdocs.config import EMBEDDING_DIM
from userdocs.errors import SQLiteStoreError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    filename    TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    text          TEXT NOT NULL,
    page          INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
"""


class UserDocsStore:
    def __init__(self, db_path: str):
        try:
            self._conn = sqlite3.connect(db_path)
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING "
                f"vec0(embedding FLOAT[{EMBEDDING_DIM}])"
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise SQLiteStoreError(f"Failed to initialize userdocs.db: {exc}") from exc

    def insert_document(self, user_id: int, filename: str, file_type: str) -> int:
        try:
            created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            cursor = self._conn.execute(
                "INSERT INTO documents (user_id, filename, file_type, created_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, filename, file_type, created_at),
            )
            self._conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as exc:
            raise SQLiteStoreError(f"Failed to insert document: {exc}") from exc

    def insert_chunks_with_vectors(self, document_id: int, chunks: list, vectors) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunks length ({len(chunks)}) != vectors row count ({vectors.shape[0]})"
            )
        try:
            for chunk, vector in zip(chunks, vectors):
                cursor = self._conn.execute(
                    "INSERT INTO chunks (document_id, chunk_index, text, page) "
                    "VALUES (?, ?, ?, ?)",
                    (document_id, chunk["chunk_index"], chunk["text"], chunk["page"]),
                )
                chunk_id = cursor.lastrowid
                self._conn.execute(
                    "INSERT INTO chunk_vectors (rowid, embedding) VALUES (?, ?)",
                    (chunk_id, sqlite_vec.serialize_float32(vector.tolist())),
                )
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to insert chunks/vectors: {exc}") from exc

    def close(self) -> None:
        self._conn.close()
