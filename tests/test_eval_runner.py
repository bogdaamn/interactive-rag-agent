from scripts.run_userdocs_eval import EvalResult, evaluate, format_report


class FakeStore:
    pass


def test_evaluate_marks_a_question_as_hit_when_the_expected_source_is_retrieved(monkeypatch):
    import scripts.run_userdocs_eval as runner

    from userdocs.retrieve import RetrievedChunk

    monkeypatch.setattr(
        runner,
        "retrieve",
        lambda store, user_id, query: [
            RetrievedChunk(text="t", filename="vacation_policy.pdf", page=1, chunk_index=0, score=0.9)
        ],
    )

    results = evaluate(
        FakeStore(),
        user_id=1,
        dataset=[{"question": "q", "expected_source": "vacation_policy.pdf"}],
    )

    assert len(results) == 1
    assert results[0].hit is True
    assert results[0].retrieved_sources == ["vacation_policy.pdf"]


def test_evaluate_marks_a_question_as_miss_when_the_expected_source_is_absent(monkeypatch):
    import scripts.run_userdocs_eval as runner

    from userdocs.retrieve import RetrievedChunk

    monkeypatch.setattr(
        runner,
        "retrieve",
        lambda store, user_id, query: [
            RetrievedChunk(text="t", filename="benefits.md", page=None, chunk_index=0, score=0.9)
        ],
    )

    results = evaluate(
        FakeStore(),
        user_id=1,
        dataset=[{"question": "q", "expected_source": "vacation_policy.pdf"}],
    )

    assert results[0].hit is False


def test_evaluate_marks_an_empty_retrieval_as_miss(monkeypatch):
    import scripts.run_userdocs_eval as runner

    monkeypatch.setattr(runner, "retrieve", lambda store, user_id, query: [])

    results = evaluate(
        FakeStore(), user_id=1, dataset=[{"question": "q", "expected_source": "a.pdf"}]
    )

    assert results[0].hit is False
    assert results[0].retrieved_sources == []


def test_evaluate_deduplicates_retrieved_sources(monkeypatch):
    """Top-K often returns several chunks from the same file; the report should
    name each source once."""
    import scripts.run_userdocs_eval as runner

    from userdocs.retrieve import RetrievedChunk

    monkeypatch.setattr(
        runner,
        "retrieve",
        lambda store, user_id, query: [
            RetrievedChunk(text="a", filename="policy.pdf", page=1, chunk_index=0, score=0.9),
            RetrievedChunk(text="b", filename="policy.pdf", page=2, chunk_index=1, score=0.8),
        ],
    )

    results = evaluate(
        FakeStore(), user_id=1, dataset=[{"question": "q", "expected_source": "policy.pdf"}]
    )

    assert results[0].retrieved_sources == ["policy.pdf"]


def test_format_report_shows_a_marker_per_question_and_a_total():
    results = [
        EvalResult("q1", "a.pdf", ["a.pdf"], True),
        EvalResult("q2", "b.pdf", ["a.pdf"], False),
    ]

    report = format_report(results)

    assert "q1" in report
    assert "q2" in report
    assert "1/2" in report
    # a miss must show what WAS retrieved, so the failure is diagnosable
    assert "a.pdf" in report


def test_format_report_handles_an_empty_result_set():
    assert format_report([]).strip()
