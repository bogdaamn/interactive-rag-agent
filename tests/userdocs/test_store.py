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


def _ingest_one_doc(store, user_id, filename, chunk_texts, rng):
    doc_id = store.insert_document(user_id, filename, ".txt")
    chunks = [{"text": t, "chunk_index": i, "page": None} for i, t in enumerate(chunk_texts)]
    vectors = rng.random((len(chunk_texts), 384)).astype(np.float32)
    # normalize rows so cosine-similarity math is meaningful in the test too
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    store.insert_chunks_with_vectors(doc_id, chunks, vectors)
    return doc_id, vectors


def test_search_vectors_returns_best_match_first(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    rng = np.random.default_rng(42)
    doc_id, vectors = _ingest_one_doc(store, 1, "doc.txt", ["a", "b", "c"], rng)

    # query with the exact vector for chunk index 1 -> should be the top hit
    query_vector = vectors[1]
    results = store.search_vectors(user_id=1, query_vector=query_vector, k=2)

    assert len(results) <= 2
    assert results[0][1] >= results[-1][1]  # descending similarity
    # the closest chunk_id should correspond to chunk_index 1 for this document
    top_chunk_id = results[0][0]
    row = store._conn.execute(
        "SELECT chunk_index FROM chunks WHERE id = ?", (top_chunk_id,)
    ).fetchone()
    assert row[0] == 1
    store.close()


def test_search_vectors_returns_empty_list_for_user_with_no_documents(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    query_vector = np.zeros(384, dtype=np.float32)
    results = store.search_vectors(user_id=999, query_vector=query_vector, k=5)
    assert results == []
    store.close()


def test_search_vectors_never_returns_another_users_chunks(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    rng = np.random.default_rng(7)
    _, vectors_a = _ingest_one_doc(store, user_id=1, filename="secret.pdf",
                                    chunk_texts=["s1", "s2"], rng=rng)
    _ingest_one_doc(store, user_id=2, filename="public.pdf",
                     chunk_texts=["p1", "p2"], rng=rng)

    # Search as user 2, using a query vector identical to one of user 1's chunks.
    results = store.search_vectors(user_id=2, query_vector=vectors_a[0], k=10)

    result_chunk_ids = {chunk_id for chunk_id, _ in results}
    user1_chunk_ids = {
        row[0] for row in store._conn.execute(
            "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.user_id = 1"
        ).fetchall()
    }
    assert result_chunk_ids.isdisjoint(user1_chunk_ids)
    store.close()


def test_get_chunk_returns_text_and_source_filename(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    doc_id = store.insert_document(1, "policy.pdf", ".pdf")
    store.insert_chunks_with_vectors(
        doc_id,
        [{"text": "vacation info", "chunk_index": 0, "page": 3}],
        np.zeros((1, 384), dtype=np.float32),
    )
    chunk_id = store._conn.execute("SELECT id FROM chunks").fetchone()[0]

    chunk = store.get_chunk(chunk_id)
    assert chunk == {
        "text": "vacation info",
        "chunk_index": 0,
        "page": 3,
        "filename": "policy.pdf",
    }
    store.close()


def test_list_documents_returns_only_this_users_documents_ordered(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    store.insert_document(1, "a.txt", ".txt")
    store.insert_document(1, "b.txt", ".txt")
    store.insert_document(2, "other_user.txt", ".txt")

    docs = store.list_documents(user_id=1)
    assert [d["filename"] for d in docs] == ["a.txt", "b.txt"]
    store.close()


def test_delete_document_removes_document_chunks_and_vectors(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    doc_id = store.insert_document(1, "old.txt", ".txt")
    store.insert_chunks_with_vectors(
        doc_id, [{"text": "x", "chunk_index": 0, "page": None}],
        np.zeros((1, 384), dtype=np.float32),
    )

    deleted = store.delete_document(user_id=1, filename="old.txt")

    assert deleted is True
    assert store._conn.execute("SELECT COUNT(*) FROM documents WHERE id = ?", (doc_id,)).fetchone()[0] == 0
    assert store._conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,)).fetchone()[0] == 0
    assert store._conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0] == 0
    store.close()


def test_delete_document_returns_false_when_not_found(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    deleted = store.delete_document(user_id=1, filename="does_not_exist.txt")
    assert deleted is False
    store.close()


def test_delete_document_cannot_delete_another_users_file(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    doc_id = store.insert_document(user_id=1, filename="shared_name.txt", file_type=".txt")
    store.insert_chunks_with_vectors(
        doc_id, [{"text": "user1 data", "chunk_index": 0, "page": None}],
        np.zeros((1, 384), dtype=np.float32),
    )

    # user 2 tries to delete a file with the same name they never uploaded
    deleted = store.delete_document(user_id=2, filename="shared_name.txt")

    assert deleted is False
    assert store._conn.execute("SELECT COUNT(*) FROM documents WHERE id = ?", (doc_id,)).fetchone()[0] == 1
    store.close()
