from userdocs.errors import RerankError
from userdocs.rerank import rerank, rerank_with_scores
from userdocs.retrieve import RetrievedChunk


def _chunk(text, score):
    return RetrievedChunk(text=text, filename="a.txt", page=None, chunk_index=0, score=score)


def test_rerank_reorders_by_cross_encoder_score(monkeypatch):
    import userdocs.rerank as rerank_module

    class FakeCrossEncoder:
        def predict(self, pairs):
            # score by position of the word "answer" — the second candidate wins
            return [1.0 if "answer" in text else 0.1 for _query, text in pairs]

    monkeypatch.setattr(rerank_module, "_get_model", lambda: FakeCrossEncoder())

    candidates = [_chunk("unrelated filler", 0.9), _chunk("the actual answer", 0.5)]
    result = rerank("what is it?", candidates)

    assert [c.text for c in result] == ["the actual answer", "unrelated filler"]


def test_rerank_preserves_original_similarity_score(monkeypatch):
    """The cross-encoder only reorders. RetrievedChunk.score stays the cosine
    similarity so the relevance threshold (spec §14) keeps working against a
    calibrated value — a cross-encoder's raw output isn't comparable to it."""
    import userdocs.rerank as rerank_module

    class FakeCrossEncoder:
        def predict(self, pairs):
            return [0.99 for _ in pairs]

    monkeypatch.setattr(rerank_module, "_get_model", lambda: FakeCrossEncoder())

    candidates = [_chunk("first", 0.42)]
    result = rerank("q", candidates)

    assert result[0].score == 0.42


def test_rerank_raises_rerank_error_on_model_failure(monkeypatch):
    import userdocs.rerank as rerank_module

    class BoomCrossEncoder:
        def predict(self, pairs):
            raise RuntimeError("model exploded")

    monkeypatch.setattr(rerank_module, "_get_model", lambda: BoomCrossEncoder())

    try:
        rerank("q", [_chunk("x", 0.5)])
        assert False, "expected RerankError"
    except RerankError:
        pass


def test_rerank_empty_candidates_returns_empty_without_loading_model():
    # No monkeypatch: if this touched the model it would try a real download.
    assert rerank("q", []) == []


def test_rerank_with_scores_exposes_the_raw_cross_encoder_score(monkeypatch):
    import userdocs.rerank as rerank_module

    class FakeCrossEncoder:
        def predict(self, pairs):
            return [7.16, -5.53]

    monkeypatch.setattr(rerank_module, "_get_model", lambda: FakeCrossEncoder())

    a, b = _chunk("relevant", 0.65), _chunk("irrelevant", 0.44)
    result = rerank_with_scores("q", [a, b])

    assert result == [(a, 7.16), (b, -5.53)]
