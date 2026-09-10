"""Cross-encoder reranking of retrieval candidates (bonus).

See spec/v3/SPEC.md §8.3. The cross-encoder reads (query, chunk) as a pair
rather than comparing two independently-computed embeddings, which makes it
more accurate for ranking — but its output is an uncalibrated relevance logit,
not a cosine similarity, so RetrievedChunk.score is deliberately left as the
original cosine value. Reranking changes the ORDER, never which candidates
cleared the relevance threshold.

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


def rerank(query: str, candidates: list) -> list:
    if not candidates:
        return []

    try:
        model = _get_model()
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = model.predict(pairs)
    except Exception as exc:
        raise RerankError(f"Reranking failed: {exc}") from exc

    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [candidate for candidate, _score in ranked]
