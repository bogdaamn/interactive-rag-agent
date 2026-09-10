import benchmark


def test_vector_only_retrieve_uses_single_vector_search_no_fusion(monkeypatch):
    fake_chunks_by_id = {1: {"id": 1, "text": "t", "source": "s.txt", "chunk_id": 0}}
    monkeypatch.setattr(benchmark.rag_query, "chunks_by_id", fake_chunks_by_id)
    monkeypatch.setattr(benchmark.rag_query, "_vector_search", lambda q, top_k: [1])

    result = benchmark.vector_only_retrieve("anything")

    assert result == [fake_chunks_by_id[1]]


def test_run_benchmark_with_no_sample_queries_does_not_raise(monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "SAMPLE_QUERIES", [])

    benchmark.run_benchmark()

    assert "No SAMPLE_QUERIES configured" in capsys.readouterr().out
