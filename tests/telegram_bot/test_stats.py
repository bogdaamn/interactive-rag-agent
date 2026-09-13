from telegram_bot.stats import UsageStats, UserStats


def test_snapshot_of_an_unknown_user_is_all_zero():
    stats = UsageStats()

    snapshot = stats.snapshot(999)

    assert snapshot == UserStats()


def test_record_turn_accumulates_tokens_and_counts_turns():
    stats = UsageStats()

    stats.record_turn(1, prompt_tokens=100, completion_tokens=20)
    stats.record_turn(1, prompt_tokens=50, completion_tokens=10)

    snapshot = stats.snapshot(1)
    assert snapshot.prompt_tokens == 150
    assert snapshot.completion_tokens == 30
    assert snapshot.turns == 2


def test_record_ingestion_accumulates_documents_and_chunks():
    stats = UsageStats()

    stats.record_ingestion(1, chunk_count=5)
    stats.record_ingestion(1, chunk_count=3)

    snapshot = stats.snapshot(1)
    assert snapshot.documents_indexed == 2
    assert snapshot.chunks_indexed == 8


def test_record_error_counts_by_category():
    stats = UsageStats()

    stats.record_error(1, "LLMTimeoutError")
    stats.record_error(1, "LLMTimeoutError")
    stats.record_error(1, "SQLiteStoreError")

    snapshot = stats.snapshot(1)
    assert snapshot.errors == {"LLMTimeoutError": 2, "SQLiteStoreError": 1}


def test_stats_are_isolated_per_user():
    stats = UsageStats()

    stats.record_turn(1, prompt_tokens=100, completion_tokens=20)
    stats.record_ingestion(1, chunk_count=5)
    stats.record_error(1, "LLMError")

    other = stats.snapshot(2)
    assert other == UserStats()


def test_snapshot_returns_an_independent_copy():
    stats = UsageStats()
    stats.record_error(1, "LLMError")

    snapshot = stats.snapshot(1)
    snapshot.errors["LLMError"] = 999

    assert stats.snapshot(1).errors == {"LLMError": 1}
