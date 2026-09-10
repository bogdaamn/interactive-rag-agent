"""FTS5 keyword search over user document chunks (bonus: hybrid search).

See spec/v3/SPEC.md §8.2. Mirrors src/rag/fts_index.py's query-sanitization
approach (split on whitespace, double-quote each token, OR them together) so a
user's raw question can never be interpreted as FTS5 operator syntax, but adds
the per-user chunk-id narrowing that spec §10 requires — FTS5 has no per-row
access control of its own.
"""

import sqlite3


def _safe_match_expression(query: str) -> str:
    """Turn arbitrary user text into a valid FTS5 MATCH expression.

    Each whitespace-separated token is double-quoted (which makes FTS5 treat
    it as a literal string, not an operator), embedded double quotes are
    doubled per FTS5's own escaping rules, and the tokens are OR-ed so any
    match counts. Returns "" if nothing usable remains.
    """
    tokens = []
    for raw_token in query.split():
        cleaned = raw_token.replace('"', '""').strip()
        if cleaned:
            tokens.append(f'"{cleaned}"')
    return " OR ".join(tokens)


def _user_chunk_ids(store, user_id: int) -> set:
    document_ids = [
        row[0]
        for row in store._conn.execute(
            "SELECT id FROM documents WHERE user_id = ?", (user_id,)
        ).fetchall()
    ]
    if not document_ids:
        return set()
    placeholders = ",".join("?" for _ in document_ids)
    return {
        row[0]
        for row in store._conn.execute(
            f"SELECT id FROM chunks WHERE document_id IN ({placeholders})", document_ids
        ).fetchall()
    }


def search_fts(store, user_id: int, query: str, k: int) -> list:
    chunk_ids = _user_chunk_ids(store, user_id)
    if not chunk_ids:
        return []

    match_expression = _safe_match_expression(query)
    if not match_expression:
        return []

    try:
        rows = store._conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?",
            (match_expression, max(k, len(chunk_ids))),
        ).fetchall()
    except sqlite3.OperationalError:
        # A malformed MATCH expression must degrade to "no keyword hits",
        # never crash the whole retrieval path.
        return []

    results = []
    for (rowid,) in rows:
        if rowid in chunk_ids:
            results.append(rowid)
        if len(results) >= k:
            break
    return results
