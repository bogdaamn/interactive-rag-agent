from rag.chunk import chunk_documents


def test_chunk_documents_assigns_globally_unique_sequential_ids(monkeypatch):
    monkeypatch.setattr("rag.chunk.CHUNK_SIZE", 5)
    monkeypatch.setattr("rag.chunk.CHUNK_OVERLAP", 1)

    documents = [
        {"path": "doc1.txt", "text": "word " * 20},
        {"path": "doc2.txt", "text": "term " * 20},
    ]
    chunks = chunk_documents(documents)

    ids = [c["id"] for c in chunks]
    assert ids == list(range(len(chunks)))
    assert len(set(ids)) == len(chunks)

    doc1_chunk_ids = [c["chunk_id"] for c in chunks if c["source"] == "doc1.txt"]
    assert doc1_chunk_ids[0] == 0
    assert len(doc1_chunk_ids) > 1  # CHUNK_SIZE=5 must split "word "*20 into >1 chunk
