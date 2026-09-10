import numpy as np

from userdocs.errors import RerankError
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


def test_retrieve_includes_keyword_only_hits_when_hybrid_enabled(monkeypatch):
    import userdocs.retrieve as retrieve_module

    monkeypatch.setattr(
        retrieve_module, "embed_chunks", lambda texts: np.zeros((1, 384), dtype=np.float32)
    )
    monkeypatch.setattr(retrieve_module, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(retrieve_module, "RERANK_ENABLED", False)
    # chunk 3 is found ONLY by keyword search, not by vector search
    monkeypatch.setattr(retrieve_module, "search_fts", lambda store, uid, q, k: [3])

    store = FakeStore(
        vector_hits=[(1, 0.80), (3, 0.61)],
        chunks={
            1: {"text": "vector hit", "chunk_index": 0, "page": None, "filename": "a.txt"},
            3: {"text": "keyword hit", "chunk_index": 2, "page": None, "filename": "a.txt"},
        },
    )

    results = retrieve(store, user_id=1, query="some query")

    texts = [r.text for r in results]
    assert "keyword hit" in texts
    assert "vector hit" in texts


def test_retrieve_drops_fused_hits_that_have_no_similarity_score(monkeypatch):
    """A chunk surfaced only by keyword search whose cosine similarity is below
    the threshold must still be dropped — the threshold is the single gate for
    every branch, not just the vector one."""
    import userdocs.retrieve as retrieve_module

    monkeypatch.setattr(
        retrieve_module, "embed_chunks", lambda texts: np.zeros((1, 384), dtype=np.float32)
    )
    monkeypatch.setattr(retrieve_module, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(retrieve_module, "RERANK_ENABLED", False)
    monkeypatch.setattr(retrieve_module, "search_fts", lambda store, uid, q, k: [9])

    store = FakeStore(
        vector_hits=[(1, 0.80)],   # chunk 9 has no vector hit at all
        chunks={
            1: {"text": "relevant", "chunk_index": 0, "page": None, "filename": "a.txt"},
            9: {"text": "keyword only, irrelevant", "chunk_index": 8, "page": None, "filename": "a.txt"},
        },
    )

    results = retrieve(store, user_id=1, query="some query")

    assert [r.text for r in results] == ["relevant"]


def test_retrieve_applies_reranking_when_enabled(monkeypatch):
    import userdocs.retrieve as retrieve_module

    monkeypatch.setattr(
        retrieve_module, "embed_chunks", lambda texts: np.zeros((1, 384), dtype=np.float32)
    )
    monkeypatch.setattr(retrieve_module, "HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(retrieve_module, "RERANK_ENABLED", True)
    # rerank reverses the order, so a difference is observable
    monkeypatch.setattr(retrieve_module, "rerank", lambda query, candidates: list(reversed(candidates)))

    store = FakeStore(
        vector_hits=[(1, 0.90), (2, 0.80)],
        chunks={
            1: {"text": "first by vector", "chunk_index": 0, "page": None, "filename": "a.txt"},
            2: {"text": "second by vector", "chunk_index": 1, "page": None, "filename": "a.txt"},
        },
    )

    results = retrieve(store, user_id=1, query="q")

    assert [r.text for r in results] == ["second by vector", "first by vector"]


def test_retrieve_falls_back_to_unreranked_order_when_rerank_fails(monkeypatch):
    import userdocs.retrieve as retrieve_module

    monkeypatch.setattr(
        retrieve_module, "embed_chunks", lambda texts: np.zeros((1, 384), dtype=np.float32)
    )
    monkeypatch.setattr(retrieve_module, "HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(retrieve_module, "RERANK_ENABLED", True)

    def boom(query, candidates):
        raise RerankError("model unavailable")

    monkeypatch.setattr(retrieve_module, "rerank", boom)

    store = FakeStore(
        vector_hits=[(1, 0.90), (2, 0.80)],
        chunks={
            1: {"text": "first by vector", "chunk_index": 0, "page": None, "filename": "a.txt"},
            2: {"text": "second by vector", "chunk_index": 1, "page": None, "filename": "a.txt"},
        },
    )

    # A reranking failure must degrade quality, not availability.
    results = retrieve(store, user_id=1, query="q")
    assert [r.text for r in results] == ["first by vector", "second by vector"]
