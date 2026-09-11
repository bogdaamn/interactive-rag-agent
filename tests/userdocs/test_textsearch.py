import numpy as np

from userdocs.store import UserDocsStore
from userdocs.textsearch import search_fts


def _ingest(store, user_id, filename, texts):
    doc_id = store.insert_document(user_id, filename, ".txt")
    chunks = [{"text": t, "chunk_index": i, "page": None} for i, t in enumerate(texts)]
    store.insert_chunks_with_vectors(
        doc_id, chunks, np.zeros((len(texts), 384), dtype=np.float32)
    )
    return doc_id


def test_search_fts_finds_matching_chunk(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    _ingest(store, 1, "policy.txt", ["employees receive vacation days", "office hours are 9 to 5"])

    results = search_fts(store, user_id=1, query="vacation", k=5)

    assert len(results) == 1
    chunk_text = store.get_chunk(results[0])["text"]
    assert "vacation" in chunk_text
    store.close()


def test_search_fts_never_returns_another_users_chunks(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    _ingest(store, 1, "secret.txt", ["confidential vacation policy for executives"])
    _ingest(store, 2, "public.txt", ["general office information"])

    # user 2 searches for a term that only exists in user 1's document
    results = search_fts(store, user_id=2, query="vacation", k=10)

    assert results == []
    store.close()


def test_search_fts_returns_empty_for_user_with_no_documents(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    assert search_fts(store, user_id=999, query="anything", k=5) == []
    store.close()


def test_search_fts_returns_empty_on_malformed_query_instead_of_raising(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    _ingest(store, 1, "policy.txt", ["employees receive vacation days"])

    # unbalanced quote / FTS5 operator soup would be a syntax error if passed raw
    assert search_fts(store, user_id=1, query='"', k=5) == []
    assert search_fts(store, user_id=1, query="AND OR NEAR", k=5) == []
    store.close()
