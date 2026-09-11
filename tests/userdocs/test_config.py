from userdocs import config


def test_chunk_overlap_smaller_than_chunk_size():
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE


def test_embedding_dim_matches_known_model():
    assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert config.EMBEDDING_DIM == 384


def test_relevance_threshold_is_a_valid_cosine_bound():
    assert 0.0 <= config.RELEVANCE_THRESHOLD <= 1.0


def test_max_document_bytes_is_20mb():
    assert config.MAX_DOCUMENT_BYTES == 20 * 1024 * 1024
