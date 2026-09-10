# tests/test_config.py
import config


def test_hybrid_search_config_constants():
    assert config.FTS_DB_PATH == "fts.db"
    assert config.RRF_K == 60
    assert config.NUM_QUERY_EXPANSIONS == 3
    assert config.CANDIDATE_K == config.TOP_K * 3
