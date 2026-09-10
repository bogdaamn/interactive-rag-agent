import numpy as np
import faiss

import rag.query as query


def test_vector_search_returns_ids_from_index_search(monkeypatch):
    class FakeModel:
        def encode(self, texts):
            return np.array([[0.1, 0.2]], dtype="float32")

    class FakeIndex:
        def search(self, q_emb, top_k):
            return np.array([[0.9, 0.5]]), np.array([[7, 3]])

    monkeypatch.setattr(query, "model", FakeModel())
    monkeypatch.setattr(query, "index", FakeIndex())
    monkeypatch.setattr(query, "chunks", [{"id": 7}, {"id": 3}])
    monkeypatch.setattr(faiss, "normalize_L2", lambda x: None)

    result = query._vector_search("cats", top_k=2)

    assert result == [7, 3]


def test_vector_search_returns_empty_list_when_index_not_loaded(monkeypatch):
    monkeypatch.setattr(query, "index", None)
    monkeypatch.setattr(query, "chunks", [])

    assert query._vector_search("cats", top_k=5) == []


def test_hybrid_search_task_fuses_per_variant_rankings():
    calls = []

    def fake_search_fn(variant, top_k):
        calls.append((variant, top_k))
        return {"cats": [1, 2, 3], "kittens": [2, 1, 4]}[variant]

    result = query._hybrid_search_task(fake_search_fn, ["cats", "kittens"], top_k=10, rrf_k=60)

    assert calls == [("cats", 10), ("kittens", 10)]
    assert result == [1, 2, 3, 4]


def test_retrieve_fuses_vector_and_fts_branches_and_falls_back_without_keywords(monkeypatch):
    fake_chunks = [
        {"id": 1, "text": "vector best", "source": "a.txt", "chunk_id": 0},
        {"id": 2, "text": "shared", "source": "b.txt", "chunk_id": 0},
        {"id": 3, "text": "fts best", "source": "c.txt", "chunk_id": 0},
    ]
    monkeypatch.setattr(query, "index", object())
    monkeypatch.setattr(query, "chunks", fake_chunks)
    monkeypatch.setattr(query, "chunks_by_id", {c["id"]: c for c in fake_chunks})
    monkeypatch.setattr(query, "generate_keywords", lambda q: [])

    monkeypatch.setattr(query, "_vector_search", lambda variant, top_k: [1, 2])
    monkeypatch.setattr(query, "search_fts", lambda fts_db_path, variant, top_k: [3, 2])

    result = query.retrieve("anything")

    # id 2 appears in both branches -> boosted above single-branch-only ids
    assert [c["id"] for c in result] == [2, 1, 3]


def test_retrieve_uses_generated_keywords_as_additional_query_variants(monkeypatch):
    fake_chunks = [
        {"id": 1, "text": "a", "source": "a.txt", "chunk_id": 0},
        {"id": 2, "text": "b", "source": "b.txt", "chunk_id": 0},
    ]
    monkeypatch.setattr(query, "index", object())
    monkeypatch.setattr(query, "chunks", fake_chunks)
    monkeypatch.setattr(query, "chunks_by_id", {c["id"]: c for c in fake_chunks})
    monkeypatch.setattr(query, "generate_keywords", lambda q: ["synonym"])

    vector_calls = []
    fts_calls = []

    def fake_vector_search(variant, top_k):
        vector_calls.append(variant)
        return [1, 2]

    def fake_search_fts(fts_db_path, variant, top_k):
        fts_calls.append(variant)
        return [2, 1]

    monkeypatch.setattr(query, "_vector_search", fake_vector_search)
    monkeypatch.setattr(query, "search_fts", fake_search_fts)

    query.retrieve("original question")

    assert vector_calls == ["original question", "synonym"]
    assert fts_calls == ["original question", "synonym"]


def test_retrieve_returns_empty_list_when_index_not_loaded(monkeypatch):
    monkeypatch.setattr(query, "index", None)
    monkeypatch.setattr(query, "chunks", [])
    monkeypatch.setattr(query, "_ensure_index_exists", lambda: False)

    assert query.retrieve("anything") == []
