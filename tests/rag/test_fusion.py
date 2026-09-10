from rag.fusion import rrf_fuse


def test_single_list_preserves_order():
    assert rrf_fuse([[1, 2, 3]]) == [1, 2, 3]


def test_disjoint_lists_interleave_by_rank():
    # same rank position in two disjoint lists -> tie, broken by numeric id
    assert rrf_fuse([[1, 2, 3], [4, 5, 6]]) == [1, 4, 2, 5, 3, 6]


def test_overlapping_ids_ranked_above_single_method_ids():
    # ids 1 and 2 appear in both lists (boosted); 3 and 4 appear in only one
    assert rrf_fuse([[1, 2, 3], [1, 2, 4]]) == [1, 2, 3, 4]


def test_duplicate_id_within_one_list_counts_once_at_best_rank():
    assert rrf_fuse([[1, 1, 2]]) == [1, 2]


def test_empty_list_is_ignored_not_an_error():
    assert rrf_fuse([[1, 2, 3], []]) == [1, 2, 3]
