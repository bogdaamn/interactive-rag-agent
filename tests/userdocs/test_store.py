import numpy as np

from userdocs.store import UserDocsStore


def test_insert_document_returns_new_id(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    doc_id = store.insert_document(user_id=1, filename="policy.pdf", file_type=".pdf")
    assert isinstance(doc_id, int)
    assert doc_id > 0
    store.close()


def test_insert_document_persists_across_reopen(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = UserDocsStore(db_path)
    doc_id = store.insert_document(user_id=1, filename="policy.pdf", file_type=".pdf")
    store.close()

    store2 = UserDocsStore(db_path)
    row = store2._conn.execute(
        "SELECT id, user_id, filename, file_type FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    assert row == (doc_id, 1, "policy.pdf", ".pdf")
    store2.close()


def test_insert_chunks_with_vectors_stores_chunks_and_vectors(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    doc_id = store.insert_document(user_id=1, filename="policy.pdf", file_type=".pdf")

    chunks = [
        {"text": "chunk one", "chunk_index": 0, "page": 1},
        {"text": "chunk two", "chunk_index": 1, "page": 1},
    ]
    vectors = np.random.default_rng(0).random((2, 384)).astype(np.float32)

    store.insert_chunks_with_vectors(doc_id, chunks, vectors)

    rows = store._conn.execute(
        "SELECT chunk_index, text, page FROM chunks WHERE document_id = ? ORDER BY chunk_index",
        (doc_id,),
    ).fetchall()
    assert rows == [(0, "chunk one", 1), (1, "chunk two", 1)]

    vector_count = store._conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
    assert vector_count == 2
    store.close()


def test_insert_chunks_with_vectors_raises_on_length_mismatch(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    doc_id = store.insert_document(user_id=1, filename="policy.pdf", file_type=".pdf")
    chunks = [{"text": "only one chunk", "chunk_index": 0, "page": None}]
    vectors = np.zeros((2, 384), dtype=np.float32)  # mismatched length
    try:
        store.insert_chunks_with_vectors(doc_id, chunks, vectors)
        assert False, "expected ValueError"
    except ValueError:
        pass
    store.close()
