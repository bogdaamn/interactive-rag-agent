import asyncio

from userdocs.errors import DocumentTooLargeError
from userdocs.pipeline import ingest_document, ingest_document_async
from userdocs.store import UserDocsStore


def test_ingest_document_stores_chunks_and_returns_result(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    raw = "Employees receive 25 paid vacation days per year.".encode("utf-8")

    result = ingest_document(store, user_id=1, filename="policy.txt", raw_bytes=raw)

    assert result.filename == "policy.txt"
    assert result.chunk_count >= 1
    stored_docs = store.list_documents(user_id=1)
    assert len(stored_docs) == 1
    assert stored_docs[0]["id"] == result.document_id
    store.close()


def test_ingest_document_rejects_oversized_upload(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    raw = b"x" * (20 * 1024 * 1024 + 1)  # one byte over MAX_DOCUMENT_BYTES
    try:
        ingest_document(store, user_id=1, filename="huge.txt", raw_bytes=raw)
        assert False, "expected DocumentTooLargeError"
    except DocumentTooLargeError:
        pass
    assert store.list_documents(user_id=1) == []
    store.close()


def test_ingest_document_async_awaits_on_progress_for_each_stage(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    raw = "Some short document text.".encode("utf-8")
    seen = []

    async def on_progress(message: str) -> None:
        seen.append(message)

    result = asyncio.run(
        ingest_document_async(store, 1, "doc.txt", raw, on_progress)
    )

    assert result.chunk_count >= 1
    assert any("Extracting" in m for m in seen)
    assert any("chunks created" in m for m in seen)
    assert any("Embeddings generated" in m for m in seen)
    store.close()


def test_ingest_document_works_with_on_progress_omitted(tmp_path):
    """Every on_progress call must be skipped entirely when it's None, so the
    pipeline has no Telegram dependency of its own."""
    store = UserDocsStore(str(tmp_path / "test.db"))
    result = ingest_document(store, 1, "doc.txt", b"Some text.", on_progress=None)
    assert result.chunk_count >= 1
    store.close()
