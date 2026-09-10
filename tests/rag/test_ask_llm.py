import rag.query as query


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_ask_llm_without_temperature_or_timeout_preserves_existing_behavior(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse({"response": "answer"})

    monkeypatch.setattr(query.requests, "post", fake_post)

    result = query.ask_llm("hello")

    assert result == "answer"
    assert "options" not in captured["json"]
    assert captured["timeout"] is None


def test_ask_llm_with_temperature_adds_options_payload(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return _FakeResponse({"response": "answer"})

    monkeypatch.setattr(query.requests, "post", fake_post)

    query.ask_llm("hello", temperature=0.1)

    assert captured["json"]["options"] == {"temperature": 0.1}


def test_ask_llm_with_timeout_passes_it_through(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["timeout"] = timeout
        return _FakeResponse({"response": "answer"})

    monkeypatch.setattr(query.requests, "post", fake_post)

    query.ask_llm("hello", timeout=8.0)

    assert captured["timeout"] == 8.0
