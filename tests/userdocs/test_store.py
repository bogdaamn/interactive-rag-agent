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
