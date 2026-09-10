"""Reciprocal Rank Fusion for combining vector and text search results.

See spec/v3/SPEC.md §8.1 step 2. score(id) = sum over lists L of
1 / (k + rank_L(id)), where rank is 1-based. Higher score = better.
"""

from userdocs.config import RRF_K


def rrf_fuse(ranked_lists: list, k: int = RRF_K) -> list:
    scores: dict = {}
    best_rank: dict = {}

    for ranked_list in ranked_lists:
        seen_in_this_list = set()
        rank = 0
        for item_id in ranked_list:
            if item_id in seen_in_this_list:
                continue  # first occurrence wins within one list
            seen_in_this_list.add(item_id)
            rank += 1
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in best_rank or rank < best_rank[item_id]:
                best_rank[item_id] = rank

    # descending score, then ascending best rank, then ascending id (fully
    # deterministic — no dependence on dict insertion order)
    return sorted(
        scores.keys(),
        key=lambda item_id: (-scores[item_id], best_rank[item_id], item_id),
    )
