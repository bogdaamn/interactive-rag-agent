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

    def search_vectors(self, user_id: int, query_vector, k: int) -> list:
        try:
            document_ids = [
                row[0]
                for row in self._conn.execute(
                    "SELECT id FROM documents WHERE user_id = ?", (user_id,)
                ).fetchall()
            ]
            if not document_ids:
                return []

            placeholders = ",".join("?" for _ in document_ids)
            chunk_ids = {
                row[0]
                for row in self._conn.execute(
                    f"SELECT id FROM chunks WHERE document_id IN ({placeholders})",
                    document_ids,
                ).fetchall()
            }
            if not chunk_ids:
                return []

            over_fetch_k = min(max(k, len(chunk_ids)), 500)
            rows = self._conn.execute(
                "SELECT rowid, distance FROM chunk_vectors "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (sqlite_vec.serialize_float32(list(query_vector)), over_fetch_k),
            ).fetchall()

            results = []
            for rowid, distance in rows:
                if rowid in chunk_ids:
                    cosine_similarity = 1.0 - (distance ** 2) / 2.0
                    results.append((rowid, cosine_similarity))
                if len(results) >= k:
                    break
            return results
        except sqlite3.Error as exc:
            raise SQLiteStoreError(f"Vector search failed: {exc}") from exc

    def get_chunk(self, chunk_id: int) -> dict:
        try:
            row = self._conn.execute(
                "SELECT c.text, c.chunk_index, c.page, d.filename "
                "FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE c.id = ?",
                (chunk_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise SQLiteStoreError(f"Failed to fetch chunk {chunk_id}: {exc}") from exc
        if row is None:
            raise SQLiteStoreError(f"No chunk with id {chunk_id}")
        text, chunk_index, page, filename = row
        return {"text": text, "chunk_index": chunk_index, "page": page, "filename": filename}

    def list_documents(self, user_id: int) -> list:
        try:
            rows = self._conn.execute(
                "SELECT id, filename, file_type, created_at FROM documents "
                "WHERE user_id = ? ORDER BY created_at ASC",
                (user_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise SQLiteStoreError(f"Failed to list documents for user {user_id}: {exc}") from exc
        return [
            {"id": r[0], "filename": r[1], "file_type": r[2], "created_at": r[3]}
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
