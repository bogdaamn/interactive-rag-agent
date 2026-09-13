"""Cross-encoder reranking of retrieval candidates (bonus).

See spec/v3/SPEC.md §8.3. The cross-encoder reads (query, chunk) as a pair
rather than comparing two independently-computed embeddings, which makes it
more accurate for ranking — but its output is an uncalibrated relevance logit,
not a cosine similarity, so RetrievedChunk.score is deliberately left as the
original cosine value and `rerank()` itself never drops a candidate: it only
ever reorders, it never re-admits a chunk that failed the vector-cosine
relevance threshold.

That uncalibrated-against-0.30 property doesn't mean the score is useless as
its own, separate signal, though. Verified directly against this repo's
fixture corpus: for "what is the notice period for the employees?", vector
cosine alone lets three single-chunk documents clear the 0.30 threshold
(employee_handbook.docx 0.65, vacation_policy.pdf 0.44, office_info.txt
0.33) because short, same-topic-area (HR policy) documents embed close
together — but the cross-encoder cleanly separates them on its own scale
(+7.16 vs -5.53 and -7.68). `rerank_with_scores` exposes that score so
retrieve.py can apply it as a second, independent filter (see
RERANK_THRESHOLD in config.py) on top of the vector-cosine gate, catching
same-topic-but-not-actually-relevant chunks the coarser cosine cutoff can't.

Model is lazy-loaded so importing this module (or calling rerank with no
candidates) never triggers a model download.
"""

from sentence_transformers import CrossEncoder

from userdocs.config import RERANK_MODEL
from userdocs.errors import RerankError

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank_with_scores(query: str, candidates: list) -> list:
    """Like rerank(), but also returns each candidate's raw cross-encoder
    score (best first) instead of discarding it — the score is on its own
    scale, not comparable to the cosine relevance threshold."""
    if not candidates:
        return []

    try:
        model = _get_model()
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = model.predict(pairs)
    except Exception as exc:
        raise RerankError(f"Reranking failed: {exc}") from exc

    return sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)


def rerank(query: str, candidates: list) -> list:
    return [candidate for candidate, _score in rerank_with_scores(query, candidates)]
