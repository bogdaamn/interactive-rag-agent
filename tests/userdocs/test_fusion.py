from userdocs.fusion import rrf_fuse


def test_rrf_fuse_single_list_preserves_order():
    assert rrf_fuse([[10, 20, 30]]) == [10, 20, 30]


def test_rrf_fuse_ranks_item_appearing_in_both_lists_highest():
    # id 20 is rank 2 in list A and rank 1 in list B -> best combined score
    fused = rrf_fuse([[10, 20, 30], [20, 40]])
    assert fused[0] == 20


def test_rrf_fuse_handles_empty_lists():
    assert rrf_fuse([[], []]) == []
    assert rrf_fuse([[], [5, 6]]) == [5, 6]


def test_rrf_fuse_breaks_ties_by_best_rank_then_id():
    # id 1 and id 2 both appear once at rank 1 in separate lists -> equal
    # scores; tie broken by numeric id ascending for determinism
    assert rrf_fuse([[2], [1]]) == [1, 2]


def test_rrf_fuse_deduplicates_within_a_single_list():
    fused = rrf_fuse([[7, 7, 8]])
    assert fused == [7, 8]
