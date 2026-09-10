import numpy as np

from userdocs.retrieve import RetrievedChunk, retrieve
from userdocs.store import UserDocsStore


class FakeStore:
    """Stands in for UserDocsStore so this test needs no embedding model,
    no sqlite-vec, and no real database — retrieve()'s orchestration logic is
    what's under test here, not storage."""

    def __init__(self, vector_hits, chunks):
        self._vector_hits = vector_hits          # list[(chunk_id, cosine_similarity)]
        self._chunks = chunks                     # dict[chunk_id, dict]

    def search_vectors(self, user_id, query_vector, k):
        return self._vector_hits[:k]

    def get_chunk(self, chunk_id):
        return self._chunks[chunk_id]


def test_retrieve_returns_chunks_above_threshold_best_first(monkeypatch):
    import userdocs.retrieve as retrieve_module

    monkeypatch.setattr(
        retrieve_module, "embed_chunks", lambda texts: np.zeros((1, 384), dtype=np.float32)
    )
    monkeypatch.setattr(retrieve_module, "HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(retrieve_module, "RERANK_ENABLED", False)

    store = FakeStore(
        vector_hits=[(1, 0.91), (2, 0.55)],
        chunks={
            1: {"text": "25 vacation days", "chunk_index": 0, "page": 3, "filename": "policy.pdf"},
            2: {"text": "office hours", "chunk_index": 1, "page": 4, "filename": "policy.pdf"},
        },
    )

    results = retrieve(store, user_id=1, query="how many vacation days?")

    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert [r.text for r in results] == ["25 vacation days", "office hours"]
    assert results[0].filename == "policy.pdf"
    assert results[0].page == 3
    assert results[0].score == 0.91


def test_retrieve_returns_empty_when_all_scores_below_threshold(monkeypatch):
    import userdocs.retrieve as retrieve_module

    monkeypatch.setattr(
        retrieve_module, "embed_chunks", lambda texts: np.zeros((1, 384), dtype=np.float32)
    )
    monkeypatch.setattr(retrieve_module, "HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(retrieve_module, "RERANK_ENABLED", False)

    # Both chunks are technically the nearest neighbours, but neither is
    # actually relevant — this is the case the no-hallucination rule exists for.
    store = FakeStore(
        vector_hits=[(1, 0.12), (2, 0.05)],
        chunks={
            1: {"text": "unrelated text", "chunk_index": 0, "page": None, "filename": "other.txt"},
            2: {"text": "also unrelated", "chunk_index": 1, "page": None, "filename": "other.txt"},
        },
    )

    assert retrieve(store, user_id=1, query="parental leave policy") == []


def test_retrieve_returns_empty_when_user_has_no_documents(monkeypatch):
    import userdocs.retrieve as retrieve_module

    monkeypatch.setattr(
        retrieve_module, "embed_chunks", lambda texts: np.zeros((1, 384), dtype=np.float32)
    )
    monkeypatch.setattr(retrieve_module, "HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(retrieve_module, "RERANK_ENABLED", False)

    store = FakeStore(vector_hits=[], chunks={})
    assert retrieve(store, user_id=1, query="anything at all") == []
