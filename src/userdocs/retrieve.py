"""Retrieval orchestration. See spec/v3/SPEC.md §8.1.

The relevance threshold here is the no-hallucination rule's enforcement point
(spec §14): vector search always returns its nearest neighbours, however
irrelevant, so an empty-result check alone would never fire for a question the
documents don't answer. Filtering on cosine similarity is what makes
"I didn't find that in your documents" a real, testable behavior.
"""

from dataclasses import dataclass

from userdocs.config import CANDIDATE_K, RELEVANCE_THRESHOLD, TOP_K
from userdocs.embed import embed_chunks

HYBRID_SEARCH_ENABLED = False   # flipped on in Task 5
RERANK_ENABLED = False           # flipped on in Task 7


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

    candidates = []
    for chunk_id, score in vector_hits[:TOP_K]:
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

    return [c for c in candidates if c.score >= RELEVANCE_THRESHOLD]
