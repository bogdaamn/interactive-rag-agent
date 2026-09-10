def rrf_fuse(ranked_lists, k=60):
    """Merge N ranked lists of ids into one ranking via Reciprocal Rank Fusion.

    score(id) = sum, over every list L containing id, of 1 / (k + rank_L(id))
    where rank_L(id) is the 1-based position of id in L. Ties are broken by
    (1) the id's best rank in any single input list, then (2) the id's
    numeric value, for a fully deterministic order.
    """
    scores = {}
    best_rank = {}

    for ranked_list in ranked_lists:
        seen_in_this_list = set()
        for rank, doc_id in enumerate(ranked_list, start=1):
            if doc_id in seen_in_this_list:
                continue  # only the first (best-ranked) occurrence counts
            seen_in_this_list.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in best_rank or rank < best_rank[doc_id]:
                best_rank[doc_id] = rank

    return sorted(
        scores.keys(),
        key=lambda doc_id: (-scores[doc_id], best_rank[doc_id], doc_id),
    )
