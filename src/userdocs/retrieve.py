# src/userdocs/retrieve.py
"""Retrieval orchestration. See spec/v3/SPEC.md §8.1.

The relevance threshold here is the no-hallucination rule's enforcement point
(spec §14): vector search always returns its nearest neighbours, however
irrelevant, so an empty-result check alone would never fire for a question the
documents don't answer. Filtering on cosine similarity is what makes
"I didn't find that in your documents" a real, testable behavior.

Hybrid search (bonus) fuses the vector branch with an FTS5 keyword branch via
RRF. A chunk that only the keyword branch found has no cosine similarity from
the vector search, so it can't be threshold-checked — those are dropped rather
than admitted unscored, keeping the threshold the single gate for every branch.
"""

from dataclasses import dataclass

from userdocs.config import (
    CANDIDATE_K,
    HYBRID_SEARCH_ENABLED,
    RELEVANCE_THRESHOLD,
    RERANK_ENABLED,
    TOP_K,
)
from userdocs.embed import embed_chunks
from userdocs.errors import RerankError
from userdocs.fusion import rrf_fuse
from userdocs.rerank import rerank
from userdocs.textsearch import search_fts


@dataclass
class RetrievedChunk:
    text: str
    filename: str
    page: int | None
    chunk_index: int
    score: float


def retrieve(store, user_id: int, query: str) -> list:
    query_vector = embed_chunks([query])[0]
    vector_hits = store.search_vectors(user_id, query_vector, CANDIDATE_K)
    similarity_by_chunk_id = {chunk_id: score for chunk_id, score in vector_hits}

    if HYBRID_SEARCH_ENABLED:
        keyword_hits = search_fts(store, user_id, query, CANDIDATE_K)
        ordered_ids = rrf_fuse([[cid for cid, _ in vector_hits], keyword_hits])
    else:
        ordered_ids = [cid for cid, _ in vector_hits]

    candidates = []
    for chunk_id in ordered_ids[:TOP_K]:
        score = similarity_by_chunk_id.get(chunk_id)
        if score is None:
            # Keyword-only hit: no cosine similarity available, so it can't
            # clear the relevance threshold. Dropped, not admitted unscored.
            continue
        chunk = store.get_chunk(chunk_id)
        candidates.append(
            RetrievedChunk(
                text=chunk["text"],
                filename=chunk["filename"],
                page=chunk["page"],
                chunk_index=chunk["chunk_index"],
                score=score,
            )
        )

    survivors = [c for c in candidates if c.score >= RELEVANCE_THRESHOLD]

    if RERANK_ENABLED and survivors:
        try:
            survivors = rerank(query, survivors)
        except RerankError:
            # Reranking is a quality improvement, not a correctness
            # requirement — keep the pre-rerank order rather than losing the
            # answer entirely.
            pass

    return survivors
