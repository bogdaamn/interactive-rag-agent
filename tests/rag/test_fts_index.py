import sqlite3

from rag.fts_index import build_fts_index, search_fts


def test_build_fts_index_creates_queryable_table(tmp_path):
    fts_db_path = tmp_path / "fts.db"
    chunks = [
        {"id": 1, "text": "invoice processing steps"},
        {"id": 2, "text": "vacation policy details"},
    ]

    build_fts_index(chunks, fts_db_path)

    conn = sqlite3.connect(str(fts_db_path))
    rows = conn.execute("SELECT id, text FROM chunks_fts ORDER BY id").fetchall()
    conn.close()
    assert rows == [(1, "invoice processing steps"), (2, "vacation policy details")]


def test_build_fts_index_overwrites_existing_file(tmp_path):
    fts_db_path = tmp_path / "fts.db"
    build_fts_index([{"id": 1, "text": "old content"}], fts_db_path)
    build_fts_index([{"id": 2, "text": "new content"}], fts_db_path)

    conn = sqlite3.connect(str(fts_db_path))
    rows = conn.execute("SELECT id, text FROM chunks_fts").fetchall()
    conn.close()
    assert rows == [(2, "new content")]


def test_search_fts_ranks_by_bm25_best_match_first(tmp_path):
    fts_db_path = tmp_path / "fts.db"
    chunks = [
        {"id": 1, "text": "invoice invoice invoice processing system with OAuth2 OAuth2 tokens OAuth2."},
        {"id": 2, "text": "Our company kitchen has a coffee machine and nothing else relevant here."},
        {"id": 3, "text": "A brief note mentioning invoice once."},
    ]
    build_fts_index(chunks, fts_db_path)

    result = search_fts(fts_db_path, "invoice OAuth2", top_k=10)

    assert result == [1, 3]  # id 2 has no matching terms, excluded entirely


def test_search_fts_respects_top_k_limit(tmp_path):
    fts_db_path = tmp_path / "fts.db"
    chunks = [{"id": i, "text": "invoice"} for i in range(5)]
    build_fts_index(chunks, fts_db_path)

    result = search_fts(fts_db_path, "invoice", top_k=2)

    assert len(result) == 2


def test_search_fts_returns_empty_list_for_blank_query(tmp_path):
    fts_db_path = tmp_path / "fts.db"
    build_fts_index([{"id": 1, "text": "invoice"}], fts_db_path)

    assert search_fts(fts_db_path, "", top_k=10) == []
    assert search_fts(fts_db_path, "   ", top_k=10) == []


def test_search_fts_returns_empty_list_on_malformed_query_instead_of_raising(tmp_path):
    fts_db_path = tmp_path / "fts.db"
    build_fts_index([{"id": 1, "text": "hello world"}], fts_db_path)

    assert search_fts(fts_db_path, '"', top_k=10) == []


def test_search_fts_warns_and_returns_empty_list_when_index_missing(tmp_path, capsys):
    fts_db_path = tmp_path / "fts.db"  # never built - no chunks_fts table

    result = search_fts(fts_db_path, "invoice", top_k=10)

    assert result == []
    captured = capsys.readouterr()
    assert str(fts_db_path) in captured.out
    assert "not found" in captured.out.lower()
