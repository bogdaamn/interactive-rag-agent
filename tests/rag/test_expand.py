import requests

from rag.expand import generate_keywords


def test_returns_parsed_keywords_on_success():
    fake_ask_llm = lambda prompt, **kwargs: "invoice, payment terms, late fee"
    result = generate_keywords("What happens if I pay late?", ask_llm_fn=fake_ask_llm)
    assert result == ["invoice", "payment terms", "late fee"]


def test_truncates_to_num_query_expansions():
    fake_ask_llm = lambda prompt, **kwargs: "a, b, c, d, e"
    result = generate_keywords("q", ask_llm_fn=fake_ask_llm)
    assert result == ["a", "b", "c"]


def test_drops_empty_and_whitespace_only_pieces():
    fake_ask_llm = lambda prompt, **kwargs: "invoice, , payment terms,   "
    result = generate_keywords("q", ask_llm_fn=fake_ask_llm)
    assert result == ["invoice", "payment terms"]


def test_drops_pieces_that_echo_the_original_query():
    query = "What happens if I pay late?"
    fake_ask_llm = lambda prompt, **kwargs: f"{query}, late fee"
    result = generate_keywords(query, ask_llm_fn=fake_ask_llm)
    assert result == ["late fee"]


def test_returns_empty_list_when_response_has_no_usable_phrases():
    fake_ask_llm = lambda prompt, **kwargs: ""
    assert generate_keywords("q", ask_llm_fn=fake_ask_llm) == []


def test_returns_empty_list_on_timeout_instead_of_raising():
    def raising_ask_llm(prompt, **kwargs):
        raise requests.exceptions.Timeout("simulated timeout")

    assert generate_keywords("q", ask_llm_fn=raising_ask_llm) == []


def test_returns_empty_list_on_any_other_exception_instead_of_raising():
    def raising_ask_llm(prompt, **kwargs):
        raise ValueError("simulated garbage response")

    assert generate_keywords("q", ask_llm_fn=raising_ask_llm) == []


def test_calls_ask_llm_fn_with_query_in_prompt_and_low_temperature_and_timeout():
    captured = {}

    def fake_ask_llm(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return "keyword"

    generate_keywords("my search question", ask_llm_fn=fake_ask_llm)

    assert "my search question" in captured["prompt"]
    assert captured["kwargs"] == {"temperature": 0.1, "timeout": 8.0}
