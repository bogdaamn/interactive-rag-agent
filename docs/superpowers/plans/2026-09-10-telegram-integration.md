# Telegram Integration & Error Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the aiogram Telegram bot surface — document upload with live progress messages, `/documents` and `/delete` commands, plain-text questions routed through the agent loop — and map all 10 error categories to friendly user-facing messages so a stack trace never reaches a user.

**Architecture:** A new `src/telegram_bot/` package. `llm_client.py` wraps Ollama's native `/api/chat` tool-calling over `httpx` and collapses six network failure modes into `LLMError`/`LLMTimeoutError`. `errors.py` holds the one mapping table from exception type → user message. Each handler in `handlers/` is a thin async function that calls into `userdocs/` and translates any typed exception through that table — the handlers are the *only* place errors are caught, so every layer below them stays free of defensive `try/except`.

**Tech Stack:** Python 3.10+, `aiogram` 3.x, `httpx`, `python-dotenv` (all new dependencies), `pytest-asyncio` (new test dependency).

**Spec:** `spec/v3/SPEC.md` §9 (error handling, all 10 categories), §10 (Telegram handlers, commands, progress messages), §11.2 (LLM client / native tool-calling)

**Depends on:**
- `docs/superpowers/plans/2026-09-10-document-ingestion-pipeline.md` — `UserDocsStore`, `ingest_document_async`, `userdocs.errors`.
- `docs/superpowers/plans/2026-09-10-rag-retrieval-tool.md` — `build_search_documents_tool`, `ToolRegistry`, `agent.run`.
- `docs/superpowers/plans/2026-09-10-conversation-aware-rag.md` — `SessionStore`. **Task 8 of this plan (the chat handler) consumes it**, so either complete that plan first or implement its Task 1 before this plan's Task 8.

## Global Constraints

- **No stack trace ever reaches a user** (assignment §12). Every handler catches the typed exceptions it can produce and replies with the fixed string from `telegram_bot/errors.py::ERROR_MESSAGES`.
- Exactly one catch site per error category — the handler that called into the pipeline/agent. No defensive `try/except` inside `userdocs/*` beyond the typed re-raises the ingestion plan already built.
- The two scripted upload messages must match the assignment's §1 text exactly:
  `"📄 Документ получен.\n\nНачинаю обработку..."` and
  `"✅ Документ готов.\n\nТеперь вы можете задавать вопросы по документу."`
- Progress messages **edit one message in place** rather than sending six separate messages per upload (spec §10.1) — otherwise a single upload floods the chat.
- Handlers take their dependencies (`store`, `llm`, `sessions`, and a per-message tool registry) as explicit parameters, never module-level globals — this is what makes them testable without a running bot, and it's how `user_id` binding stays in trusted code.
- The `search_documents` tool is constructed **per message** from `message.from_user.id` (spec §11.1) — never reused across users.
- `python-dotenv` config: fail loudly with `ConfigError` on a missing/invalid value, never silently default a bot token or clamp a range (mirrors `../telegram-bot/config.py`).
- TDD, strictly: failing test → confirm the failure → minimal implementation → confirm green → commit.

---

### Task 1: Bot config from environment

**Files:**
- Create: `src/telegram_bot/__init__.py`
- Create: `src/telegram_bot/config.py`
- Modify: `requirements.txt`
- Test: `tests/telegram_bot/__init__.py`
- Test: `tests/telegram_bot/test_config.py`

**Interfaces:**
- Produces: `telegram_bot.config.ConfigError` (an `Exception` subclass); `telegram_bot.config.Config` (frozen dataclass: `bot_token: str`, `allowed_user_ids: frozenset[int]`, `ollama_base_url: str`, `ollama_model: str`, `llm_timeout_seconds: float`, `userdocs_db_path: str`); `telegram_bot.config.load_config(env: dict | None = None) -> Config`.

- [ ] **Step 1: Write the failing test**

```python
# tests/telegram_bot/__init__.py
```

```python
# tests/telegram_bot/test_config.py
import pytest

from telegram_bot.config import ConfigError, load_config

_MINIMAL_ENV = {"TELEGRAM_BOT_TOKEN": "123:abc"}


def test_load_config_reads_bot_token():
    config = load_config(_MINIMAL_ENV)
    assert config.bot_token == "123:abc"


def test_load_config_applies_documented_defaults():
    config = load_config(_MINIMAL_ENV)
    assert config.ollama_base_url == "http://localhost:11434"
    assert config.ollama_model == "qwen2.5:7b"
    assert config.llm_timeout_seconds == 120.0
    assert config.userdocs_db_path == "userdocs.db"
    assert config.allowed_user_ids == frozenset()


def test_load_config_parses_allowed_user_ids():
    config = load_config({**_MINIMAL_ENV, "ALLOWED_USER_IDS": "111, 222,333"})
    assert config.allowed_user_ids == frozenset({111, 222, 333})


def test_load_config_raises_when_bot_token_missing():
    with pytest.raises(ConfigError, match="TELEGRAM_BOT_TOKEN"):
        load_config({})


def test_load_config_raises_on_non_numeric_allowed_user_id():
    with pytest.raises(ConfigError, match="ALLOWED_USER_IDS"):
        load_config({**_MINIMAL_ENV, "ALLOWED_USER_IDS": "111,not-a-number"})


def test_load_config_raises_on_non_positive_timeout():
    with pytest.raises(ConfigError, match="LLM_TIMEOUT_SECONDS"):
        load_config({**_MINIMAL_ENV, "LLM_TIMEOUT_SECONDS": "0"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/telegram_bot/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telegram_bot'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/telegram_bot/__init__.py
```

```python
# src/telegram_bot/config.py
"""Environment-driven bot config. See spec/v3/SPEC.md §11.2's model decision.

Follows ../telegram-bot/config.py's approach: a frozen dataclass built once at
startup, with a hard ConfigError on anything invalid rather than a silent
default or clamp — a bot that starts with a subtly wrong config is worse than
one that refuses to start.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    bot_token: str
    allowed_user_ids: frozenset
    ollama_base_url: str
    ollama_model: str
    llm_timeout_seconds: float
    userdocs_db_path: str


def _parse_allowed_user_ids(raw: str) -> frozenset:
    if not raw.strip():
        return frozenset()
    ids = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            ids.add(int(token))
        except ValueError as exc:
            raise ConfigError(
                f"ALLOWED_USER_IDS contains a non-numeric entry: {token!r}"
            ) from exc
    return frozenset(ids)


def load_config(env: dict | None = None) -> Config:
    if env is None:
        load_dotenv()
        env = dict(os.environ)

    bot_token = env.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not bot_token:
        raise ConfigError("TELEGRAM_BOT_TOKEN is required but was not set")

    raw_timeout = env.get("LLM_TIMEOUT_SECONDS", "120")
    try:
        llm_timeout_seconds = float(raw_timeout)
    except ValueError as exc:
        raise ConfigError(f"LLM_TIMEOUT_SECONDS is not a number: {raw_timeout!r}") from exc
    if llm_timeout_seconds <= 0:
        raise ConfigError(f"LLM_TIMEOUT_SECONDS must be positive, got {llm_timeout_seconds}")

    return Config(
        bot_token=bot_token,
        allowed_user_ids=_parse_allowed_user_ids(env.get("ALLOWED_USER_IDS", "")),
        ollama_base_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=env.get("OLLAMA_MODEL", "qwen2.5:7b"),
        llm_timeout_seconds=llm_timeout_seconds,
        userdocs_db_path=env.get("USERDOCS_DB_PATH", "userdocs.db"),
    )
```

Append to `requirements.txt`:

```text
aiogram
httpx
python-dotenv
pytest-asyncio
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/telegram_bot/test_config.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/telegram_bot/__init__.py src/telegram_bot/config.py \
        tests/telegram_bot/__init__.py tests/telegram_bot/test_config.py requirements.txt
git commit -m "feat(telegram_bot): add env-driven config with hard validation"
```

---

### Task 2: `OllamaClient` — native tool-calling with granular error mapping

**Files:**
- Create: `src/telegram_bot/llm_errors.py`
- Create: `src/telegram_bot/llm_client.py`
- Create: `tests/telegram_bot/test_llm_client.py`

**Interfaces:**
- Produces: `telegram_bot.llm_errors.LLMError`, `telegram_bot.llm_errors.LLMTimeoutError(LLMError)`; `telegram_bot.llm_client.OllamaClient(base_url: str, model: str, timeout_seconds: float)` with `async chat(messages: list[dict], tools: list[dict]) -> dict` returning `{"content": str, "tool_calls": list[dict]}`.

> Note: `LLMError`/`LLMTimeoutError` live in their own module rather than in `telegram_bot/errors.py` because `errors.py` (Task 3) imports both these *and* `userdocs.errors` to build the mapping table — keeping the exception definitions separate from the mapping avoids a circular import.

- [ ] **Step 1: Write the failing test**

```python
# tests/telegram_bot/test_llm_client.py
import httpx
import pytest

from telegram_bot.llm_client import OllamaClient
from telegram_bot.llm_errors import LLMError, LLMTimeoutError


def _client(transport):
    client = OllamaClient(base_url="http://ollama.test", model="test-model", timeout_seconds=5.0)
    client._http = httpx.AsyncClient(transport=transport, timeout=5.0)
    return client


@pytest.mark.asyncio
async def test_chat_returns_content_and_empty_tool_calls():
    def handler(request):
        return httpx.Response(200, json={"message": {"content": "Hello!"}})

    client = _client(httpx.MockTransport(handler))
    result = await client.chat([{"role": "user", "content": "hi"}], tools=[])

    assert result == {"content": "Hello!", "tool_calls": []}


@pytest.mark.asyncio
async def test_chat_sends_model_messages_and_tools_and_disables_streaming():
    captured = {}

    def handler(request):
        import json as _json

        captured.update(_json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    client = _client(httpx.MockTransport(handler))
    tools = [{"type": "function", "function": {"name": "t"}}]
    await client.chat([{"role": "user", "content": "hi"}], tools=tools)

    assert captured["model"] == "test-model"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["tools"] == tools
    assert captured["stream"] is False


@pytest.mark.asyncio
async def test_chat_normalizes_tool_call_arguments_given_as_a_json_string():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "search_documents", "arguments": '{"query": "vacation"}'}}
                    ],
                }
            },
        )

    client = _client(httpx.MockTransport(handler))
    result = await client.chat([], tools=[])

    assert result["tool_calls"][0]["function"]["arguments"] == {"query": "vacation"}


@pytest.mark.asyncio
async def test_chat_raises_llm_timeout_error_on_read_timeout():
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(LLMTimeoutError):
        await client.chat([], tools=[])


@pytest.mark.asyncio
async def test_chat_raises_llm_error_on_connect_failure():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(LLMError):
        await client.chat([], tools=[])


@pytest.mark.asyncio
async def test_chat_raises_llm_error_on_http_500():
    def handler(request):
        return httpx.Response(500, text="internal error")

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(LLMError):
        await client.chat([], tools=[])


@pytest.mark.asyncio
async def test_chat_raises_llm_error_on_unparseable_body():
    def handler(request):
        return httpx.Response(200, text="not json at all")

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(LLMError):
        await client.chat([], tools=[])


@pytest.mark.asyncio
async def test_llm_timeout_error_is_an_llm_error():
    assert issubclass(LLMTimeoutError, LLMError)
```

Also add `pytest.ini` (or `pyproject.toml`) config so `pytest-asyncio` picks these up:

```ini
# pytest.ini
[pytest]
asyncio_mode = auto
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/telegram_bot/test_llm_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telegram_bot.llm_client'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/telegram_bot/llm_errors.py
"""LLM transport exceptions — error categories #8 and #9 in spec/v3/SPEC.md §9."""


class LLMError(Exception):
    """Category #8: the LLM call failed (connection, HTTP status, bad body)."""


class LLMTimeoutError(LLMError):
    """Category #9: the LLM call timed out. A subclass of LLMError so a caller
    that only cares about "the LLM failed" can catch one type, while the
    handler can still give a timeout-specific message."""
```

```python
# src/telegram_bot/llm_client.py
"""Ollama chat client with native tool-calling. See spec/v3/SPEC.md §11.2.

Adapted from ../telegram-bot/llm_client.py. Uses /api/chat with a `tools` array
(the model's own tool-calling format) rather than /api/generate with a
hand-written JSON convention — assignment §8 wants the *model* deciding when to
search, and this is the interface that lets it.

Every httpx failure mode is collapsed into LLMError (or LLMTimeoutError for the
two timeout variants) so callers never need to know httpx exists.
"""

import json

import httpx

from telegram_bot.llm_errors import LLMError, LLMTimeoutError


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: float):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._http = httpx.AsyncClient(timeout=timeout_seconds)

    @staticmethod
    def _normalize_tool_calls(tool_calls: list) -> list:
        """Ollama returns tool-call arguments as either a decoded object or a
        JSON string depending on the model. Normalize to a dict so downstream
        code has one shape to handle."""
        normalized = []
        for call in tool_calls:
            function = dict(call.get("function", {}))
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    function["arguments"] = {}
            normalized.append({**call, "function": function})
        return normalized

    async def chat(self, messages: list, tools: list) -> dict:
        payload = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "stream": False,
        }
        try:
            response = await self._http.post(f"{self._base_url}/api/chat", json=payload)
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            raise LLMTimeoutError(f"LLM request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise LLMError(f"Could not connect to the LLM: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        if response.status_code >= 400:
            raise LLMError(f"LLM returned HTTP {response.status_code}")

        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(f"LLM returned an unparseable body: {exc}") from exc

        message = body.get("message")
        if not isinstance(message, dict):
            raise LLMError("LLM response contained no message object")

        return {
            "content": message.get("content", ""),
            "tool_calls": self._normalize_tool_calls(message.get("tool_calls") or []),
        }

    async def close(self) -> None:
        await self._http.aclose()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/telegram_bot/test_llm_client.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add src/telegram_bot/llm_errors.py src/telegram_bot/llm_client.py \
        tests/telegram_bot/test_llm_client.py pytest.ini
git commit -m "feat(telegram_bot): add Ollama client with native tool-calling and typed errors"
```

---

### Task 3: The error → user-message mapping table (all 10 categories)

**Files:**
- Create: `src/telegram_bot/errors.py`
- Create: `tests/telegram_bot/test_errors.py`

**Interfaces:**
- Consumes: `userdocs.errors.*` (ingestion plan Task 1), `telegram_bot.llm_errors.*` (Task 2).
- Produces: `telegram_bot.errors.ERROR_MESSAGES: dict[type, str]`; `telegram_bot.errors.INGEST_ERRORS: tuple[type, ...]` (the exception types the document handler catches); `telegram_bot.errors.message_for(exc: Exception) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/telegram_bot/test_errors.py
from telegram_bot.errors import ERROR_MESSAGES, INGEST_ERRORS, message_for
from telegram_bot.llm_errors import LLMError, LLMTimeoutError
from userdocs.errors import (
    CorruptDocumentError,
    DocumentTooLargeError,
    EmbeddingError,
    EmptyDocumentError,
    SQLiteStoreError,
    UnsupportedFormatError,
)

_ALL_MAPPED_TYPES = [
    UnsupportedFormatError,
    CorruptDocumentError,
    EmptyDocumentError,
    DocumentTooLargeError,
    EmbeddingError,
    SQLiteStoreError,
    LLMError,
    LLMTimeoutError,
]


def test_every_error_category_has_a_message():
    for error_type in _ALL_MAPPED_TYPES:
        assert error_type in ERROR_MESSAGES
        assert ERROR_MESSAGES[error_type].strip()


def test_no_message_leaks_a_stack_trace_or_exception_class_name():
    for error_type, text in ERROR_MESSAGES.items():
        assert "Traceback" not in text
        assert error_type.__name__ not in text


def test_message_for_uses_the_exact_type_when_mapped():
    assert message_for(UnsupportedFormatError("x")) == ERROR_MESSAGES[UnsupportedFormatError]


def test_message_for_prefers_the_most_specific_type():
    """LLMTimeoutError subclasses LLMError — it must get its own timeout
    message, not the generic LLM one."""
    timeout_message = message_for(LLMTimeoutError("slow"))
    assert timeout_message == ERROR_MESSAGES[LLMTimeoutError]
    assert timeout_message != ERROR_MESSAGES[LLMError]


def test_message_for_falls_back_to_a_generic_message_for_unmapped_errors():
    text = message_for(RuntimeError("something nobody anticipated"))
    assert text.strip()
    assert "something nobody anticipated" not in text  # never echo raw exception text


def test_ingest_errors_covers_every_ingestion_side_category():
    assert set(INGEST_ERRORS) == {
        UnsupportedFormatError,
        CorruptDocumentError,
        EmptyDocumentError,
        DocumentTooLargeError,
        EmbeddingError,
        SQLiteStoreError,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/telegram_bot/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telegram_bot.errors'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/telegram_bot/errors.py
"""Exception → user-facing message mapping. See spec/v3/SPEC.md §9's table.

This is the single place a raised exception becomes text a user reads.
Assignment §12's requirement is that a user never sees a stack trace, so no
message here interpolates str(exc) — the exception detail goes to the log, the
fixed friendly string goes to the chat.

Category #10 (Telegram API error) has no entry: if the Telegram call itself
failed there is no channel left to send a message on. It's logged by the
handler's outermost except and nothing is sent.
"""

from telegram_bot.llm_errors import LLMError, LLMTimeoutError
from userdocs.errors import (
    CorruptDocumentError,
    DocumentTooLargeError,
    EmbeddingError,
    EmptyDocumentError,
    SQLiteStoreError,
    UnsupportedFormatError,
)

GENERIC_MESSAGE = "❌ Something went wrong. Please try again."

ERROR_MESSAGES = {
    # 1
    UnsupportedFormatError: (
        "❌ Unsupported file format.\n\nPlease send a .txt, .md, .docx, or .pdf file."
    ),
    # 2 and 3 (corrupted PDF / corrupted DOCX share one type and one message)
    CorruptDocumentError: (
        "❌ Не удалось обработать документ.\n\n"
        "Пожалуйста, убедитесь, что файл не повреждён."
    ),
    # 4
    EmptyDocumentError: (
        "❌ This document appears to be empty — there's nothing to index."
    ),
    # 5
    DocumentTooLargeError: (
        "❌ This file is too large (max 20 MB).\n\nPlease split it or send a smaller file."
    ),
    # 6
    EmbeddingError: (
        "❌ Something went wrong while indexing your document. Please try again."
    ),
    # 7
    SQLiteStoreError: (
        "❌ Something went wrong while saving your document. Please try again."
    ),
    # 8
    LLMError: "❌ The assistant is temporarily unavailable. Please try again shortly.",
    # 9
    LLMTimeoutError: "⏳ That took too long — please try again.",
}

INGEST_ERRORS = (
    UnsupportedFormatError,
    CorruptDocumentError,
    EmptyDocumentError,
    DocumentTooLargeError,
    EmbeddingError,
    SQLiteStoreError,
)


def message_for(exc: Exception) -> str:
    """Most-specific-type-wins lookup: walk the exception's MRO so a subclass
    (LLMTimeoutError) gets its own message before falling back to its base
    (LLMError), and an entirely unmapped exception gets a generic message
    rather than leaking its text."""
    for candidate in type(exc).__mro__:
        if candidate in ERROR_MESSAGES:
            return ERROR_MESSAGES[candidate]
    return GENERIC_MESSAGE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/telegram_bot/test_errors.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/telegram_bot/errors.py tests/telegram_bot/test_errors.py
git commit -m "feat(telegram_bot): map all 10 error categories to friendly user messages"
```

---

### Task 4: Document upload handler — happy path with the scripted messages

**Files:**
- Create: `src/telegram_bot/handlers/__init__.py`
- Create: `src/telegram_bot/handlers/documents.py`
- Test: `tests/telegram_bot/handlers/__init__.py`
- Test: `tests/telegram_bot/handlers/test_documents.py`

**Interfaces:**
- Consumes: `userdocs.pipeline.ingest_document_async` (ingestion plan Task 14).
- Produces: `telegram_bot.handlers.documents.RECEIVED_MESSAGE`, `READY_MESSAGE` (the two scripted strings); `async telegram_bot.handlers.documents.handle_document(message, store) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/telegram_bot/handlers/__init__.py
```

```python
# tests/telegram_bot/handlers/test_documents.py
import pytest

from telegram_bot.handlers.documents import (
    READY_MESSAGE,
    RECEIVED_MESSAGE,
    handle_document,
)


class FakeSentMessage:
    """Stands in for the Message object message.answer() returns, so
    edit_text() calls can be observed."""

    def __init__(self, text, log):
        self.text = text
        self._log = log

    async def edit_text(self, text):
        self.text = text
        self._log.append(("edit", text))


class FakeBot:
    def __init__(self, file_bytes=b"document contents"):
        self._file_bytes = file_bytes

    async def get_file(self, file_id):
        return type("FakeFile", (), {"file_path": f"remote/{file_id}"})()

    async def download_file(self, file_path):
        import io

        return io.BytesIO(self._file_bytes)


class FakeDocument:
    def __init__(self, file_name="policy.txt", file_id="fid-1"):
        self.file_name = file_name
        self.file_id = file_id


class FakeMessage:
    def __init__(self, user_id=1, file_name="policy.txt", file_bytes=b"document contents"):
        self.from_user = type("FakeUser", (), {"id": user_id})()
        self.document = FakeDocument(file_name=file_name)
        self.bot = FakeBot(file_bytes=file_bytes)
        self.sent = []          # list of ("answer", text) / ("edit", text)

    async def answer(self, text):
        self.sent.append(("answer", text))
        return FakeSentMessage(text, self.sent)


@pytest.mark.asyncio
async def test_handle_document_sends_scripted_received_and_ready_messages(monkeypatch, tmp_path):
    import telegram_bot.handlers.documents as documents_module

    async def fake_ingest(store, user_id, filename, raw_bytes, on_progress=None):
        if on_progress is not None:
            await on_progress("⏳ Extracting text...")
        return type("R", (), {"document_id": 1, "filename": filename, "chunk_count": 3})()

    monkeypatch.setattr(documents_module, "ingest_document_async", fake_ingest)

    message = FakeMessage()
    await handle_document(message, store=None)

    answers = [text for kind, text in message.sent if kind == "answer"]
    assert answers[0] == RECEIVED_MESSAGE
    assert READY_MESSAGE in answers


@pytest.mark.asyncio
async def test_handle_document_passes_the_senders_user_id_to_the_pipeline(monkeypatch):
    import telegram_bot.handlers.documents as documents_module

    seen = {}

    async def fake_ingest(store, user_id, filename, raw_bytes, on_progress=None):
        seen["user_id"] = user_id
        seen["filename"] = filename
        seen["raw_bytes"] = raw_bytes
        return type("R", (), {"document_id": 1, "filename": filename, "chunk_count": 1})()

    monkeypatch.setattr(documents_module, "ingest_document_async", fake_ingest)

    message = FakeMessage(user_id=555, file_name="handbook.pdf", file_bytes=b"pdf bytes")
    await handle_document(message, store=None)

    assert seen["user_id"] == 555
    assert seen["filename"] == "handbook.pdf"
    assert seen["raw_bytes"] == b"pdf bytes"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/telegram_bot/handlers/test_documents.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telegram_bot.handlers'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/telegram_bot/handlers/__init__.py
```

```python
# src/telegram_bot/handlers/documents.py
"""Document upload handler. See spec/v3/SPEC.md §10.1.

The two scripted messages below are verbatim from the assignment's §1 user
scenario and must not be reworded.
"""

import logging

from userdocs.pipeline import ingest_document_async

logger = logging.getLogger(__name__)

RECEIVED_MESSAGE = "📄 Документ получен.\n\nНачинаю обработку..."
READY_MESSAGE = "✅ Документ готов.\n\nТеперь вы можете задавать вопросы по документу."


async def handle_document(message, store) -> None:
    await message.answer(RECEIVED_MESSAGE)

    file = await message.bot.get_file(message.document.file_id)
    downloaded = await message.bot.download_file(file.file_path)
    raw_bytes = downloaded.read()

    result = await ingest_document_async(
        store,
        message.from_user.id,
        message.document.file_name,
        raw_bytes,
    )
    logger.info(
        "Indexed %s for user %s (%s chunks)",
        result.filename,
        message.from_user.id,
        result.chunk_count,
    )
    await message.answer(READY_MESSAGE)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/telegram_bot/handlers/test_documents.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/telegram_bot/handlers/__init__.py src/telegram_bot/handlers/documents.py \
        tests/telegram_bot/handlers/__init__.py tests/telegram_bot/handlers/test_documents.py
git commit -m "feat(telegram_bot): handle document uploads through the ingest pipeline"
```

---

### Task 5: Progress messages during upload (bonus +1)

**Files:**
- Modify: `src/telegram_bot/handlers/documents.py`
- Modify: `tests/telegram_bot/handlers/test_documents.py`

**Interfaces:**
- Produces: `handle_document` now sends one progress message and edits it in place for each pipeline stage, via the `on_progress` callback `ingest_document_async` already accepts.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/telegram_bot/handlers/test_documents.py
@pytest.mark.asyncio
async def test_handle_document_edits_one_progress_message_per_stage(monkeypatch):
    import telegram_bot.handlers.documents as documents_module

    async def fake_ingest(store, user_id, filename, raw_bytes, on_progress=None):
        for stage in [
            "⏳ Extracting text...",
            "✅ Text extracted",
            "⏳ Creating chunks...",
            "✅ 127 chunks created",
            "⏳ Generating embeddings...",
            "✅ Embeddings generated",
        ]:
            await on_progress(stage)
        return type("R", (), {"document_id": 1, "filename": filename, "chunk_count": 127})()

    monkeypatch.setattr(documents_module, "ingest_document_async", fake_ingest)

    message = FakeMessage()
    await handle_document(message, store=None)

    edits = [text for kind, text in message.sent if kind == "edit"]
    assert "✅ 127 chunks created" in edits
    assert "✅ Embeddings generated" in edits

    # exactly one message is *sent* for progress (then edited repeatedly) —
    # six separate sends would flood the chat
    answers = [text for kind, text in message.sent if kind == "answer"]
    progress_answers = [t for t in answers if t.startswith("⏳") or t.startswith("✅ Text")]
    assert len(progress_answers) <= 1


@pytest.mark.asyncio
async def test_handle_document_survives_a_failed_progress_edit(monkeypatch):
    """Telegram rejects an edit whose text is unchanged, and rate-limits rapid
    edits. A failed progress update must not abort the upload (error category
    #10) — the document still gets indexed and the user still gets READY."""
    import telegram_bot.handlers.documents as documents_module

    class FlakyMessage(FakeMessage):
        async def answer(self, text):
            self.sent.append(("answer", text))
            outer = self

            class FlakySent:
                async def edit_text(self, text):
                    outer.sent.append(("edit-failed", text))
                    raise RuntimeError("Bad Request: message is not modified")

            return FlakySent()

    async def fake_ingest(store, user_id, filename, raw_bytes, on_progress=None):
        await on_progress("⏳ Extracting text...")
        await on_progress("✅ Text extracted")
        return type("R", (), {"document_id": 1, "filename": filename, "chunk_count": 2})()

    monkeypatch.setattr(documents_module, "ingest_document_async", fake_ingest)

    message = FlakyMessage()
    await handle_document(message, store=None)

    answers = [text for kind, text in message.sent if kind == "answer"]
    assert READY_MESSAGE in answers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/telegram_bot/handlers/test_documents.py -v`
Expected: FAIL — `TypeError: fake_ingest() ... on_progress` is `None`, so `await on_progress(stage)` raises `TypeError: 'NoneType' object is not callable`

- [ ] **Step 3: Write minimal implementation**

```python
# src/telegram_bot/handlers/documents.py — replace handle_document with:

async def handle_document(message, store) -> None:
    await message.answer(RECEIVED_MESSAGE)

    file = await message.bot.get_file(message.document.file_id)
    downloaded = await message.bot.download_file(file.file_path)
    raw_bytes = downloaded.read()

    progress_message = await message.answer("⏳ Starting...")

    async def on_progress(text: str) -> None:
        try:
            await progress_message.edit_text(text)
        except Exception as exc:
            # Telegram rejects an unchanged-text edit and rate-limits rapid
            # edits (error category #10). A progress update is cosmetic — never
            # let it abort the actual indexing work.
            logger.warning("Could not update progress message: %s", exc)

    result = await ingest_document_async(
        store,
        message.from_user.id,
        message.document.file_name,
        raw_bytes,
        on_progress,
    )
    logger.info(
        "Indexed %s for user %s (%s chunks)",
        result.filename,
        message.from_user.id,
        result.chunk_count,
    )
    await message.answer(READY_MESSAGE)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/telegram_bot/handlers/test_documents.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/telegram_bot/handlers/documents.py tests/telegram_bot/handlers/test_documents.py
git commit -m "feat(telegram_bot): show in-place upload progress messages"
```

---

### Task 6: Document upload error handling (categories #1–7)

**Files:**
- Modify: `src/telegram_bot/handlers/documents.py`
- Modify: `tests/telegram_bot/handlers/test_documents.py`

**Interfaces:**
- Consumes: `telegram_bot.errors.{INGEST_ERRORS, message_for}` (Task 3).
- Produces: `handle_document` now replies with the mapped message and returns early on any `INGEST_ERRORS` exception, never sending `READY_MESSAGE`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/telegram_bot/handlers/test_documents.py
from telegram_bot.errors import ERROR_MESSAGES
from userdocs.errors import (
    CorruptDocumentError,
    DocumentTooLargeError,
    EmbeddingError,
    EmptyDocumentError,
    SQLiteStoreError,
    UnsupportedFormatError,
)


@pytest.mark.parametrize(
    "error_type",
    [
        UnsupportedFormatError,
        CorruptDocumentError,
        EmptyDocumentError,
        DocumentTooLargeError,
        EmbeddingError,
        SQLiteStoreError,
    ],
)
@pytest.mark.asyncio
async def test_handle_document_replies_with_the_mapped_message_on_each_error(monkeypatch, error_type):
    import telegram_bot.handlers.documents as documents_module

    async def failing_ingest(store, user_id, filename, raw_bytes, on_progress=None):
        raise error_type("internal detail that must not reach the user")

    monkeypatch.setattr(documents_module, "ingest_document_async", failing_ingest)

    message = FakeMessage()
    await handle_document(message, store=None)

    answers = [text for kind, text in message.sent if kind == "answer"]
    assert ERROR_MESSAGES[error_type] in answers
    assert READY_MESSAGE not in answers
    assert not any("internal detail" in text for text in answers)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/telegram_bot/handlers/test_documents.py -v`
Expected: FAIL — all 6 parametrized cases fail; the exception propagates out of `handle_document` instead of being turned into a reply

- [ ] **Step 3: Write minimal implementation**

```python
# src/telegram_bot/handlers/documents.py — add to imports
from telegram_bot.errors import INGEST_ERRORS, message_for
```

```python
# src/telegram_bot/handlers/documents.py — wrap the ingest call:

    try:
        result = await ingest_document_async(
            store,
            message.from_user.id,
            message.document.file_name,
            raw_bytes,
            on_progress,
        )
    except INGEST_ERRORS as exc:
        logger.warning(
            "Ingestion failed for user %s, file %s: %s",
            message.from_user.id,
            message.document.file_name,
            exc,
        )
        await message.answer(message_for(exc))
        return

    logger.info(
        "Indexed %s for user %s (%s chunks)",
        result.filename,
        message.from_user.id,
        result.chunk_count,
    )
    await message.answer(READY_MESSAGE)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/telegram_bot/handlers/test_documents.py -v`
Expected: PASS (10 passed — 4 previous + 6 parametrized)

- [ ] **Step 5: Commit**

```bash
git add src/telegram_bot/handlers/documents.py tests/telegram_bot/handlers/test_documents.py
git commit -m "feat(telegram_bot): reply with friendly messages for all ingestion errors"
```

---

### Task 7: `/documents` and `/delete` commands

**Files:**
- Create: `src/telegram_bot/handlers/commands.py`
- Create: `tests/telegram_bot/handlers/test_commands.py`

**Interfaces:**
- Consumes: `UserDocsStore.{list_documents, delete_document}` (ingestion plan Tasks 12–13), `telegram_bot.errors.message_for`.
- Produces: `async telegram_bot.handlers.commands.handle_documents_command(message, store) -> None`; `async telegram_bot.handlers.commands.handle_delete_command(message, store) -> None`; `telegram_bot.handlers.commands.NO_DOCUMENTS_MESSAGE`, `DELETE_USAGE_MESSAGE`.

- [ ] **Step 1: Write the failing test**

```python
# tests/telegram_bot/handlers/test_commands.py
import pytest

from telegram_bot.handlers.commands import (
    DELETE_USAGE_MESSAGE,
    NO_DOCUMENTS_MESSAGE,
    handle_delete_command,
    handle_documents_command,
)
from userdocs.errors import SQLiteStoreError


class FakeMessage:
    def __init__(self, text="", user_id=1):
        self.text = text
        self.from_user = type("FakeUser", (), {"id": user_id})()
        self.sent = []

    async def answer(self, text):
        self.sent.append(text)


class FakeStore:
    def __init__(self, documents=None, delete_result=True, delete_error=None):
        self._documents = documents or {}
        self._delete_result = delete_result
        self._delete_error = delete_error
        self.delete_calls = []

    def list_documents(self, user_id):
        return self._documents.get(user_id, [])

    def delete_document(self, user_id, filename):
        self.delete_calls.append((user_id, filename))
        if self._delete_error is not None:
            raise self._delete_error
        return self._delete_result


@pytest.mark.asyncio
async def test_documents_command_lists_numbered_filenames():
    store = FakeStore(documents={1: [
        {"id": 1, "filename": "employee_handbook.pdf", "file_type": ".pdf", "created_at": "t1"},
        {"id": 2, "filename": "vacation_policy.pdf", "file_type": ".pdf", "created_at": "t2"},
        {"id": 3, "filename": "benefits.md", "file_type": ".md", "created_at": "t3"},
    ]})
    message = FakeMessage(user_id=1)

    await handle_documents_command(message, store)

    reply = message.sent[0]
    assert "1. employee_handbook.pdf" in reply
    assert "2. vacation_policy.pdf" in reply
    assert "3. benefits.md" in reply


@pytest.mark.asyncio
async def test_documents_command_reports_empty_list():
    message = FakeMessage(user_id=1)
    await handle_documents_command(message, FakeStore())
    assert message.sent == [NO_DOCUMENTS_MESSAGE]


@pytest.mark.asyncio
async def test_documents_command_only_lists_the_senders_documents():
    store = FakeStore(documents={
        1: [{"id": 1, "filename": "mine.txt", "file_type": ".txt", "created_at": "t"}],
        2: [{"id": 2, "filename": "theirs.txt", "file_type": ".txt", "created_at": "t"}],
    })
    message = FakeMessage(user_id=1)

    await handle_documents_command(message, store)

    assert "mine.txt" in message.sent[0]
    assert "theirs.txt" not in message.sent[0]


@pytest.mark.asyncio
async def test_delete_command_confirms_a_successful_delete():
    store = FakeStore(delete_result=True)
    message = FakeMessage(text="/delete vacation_policy.pdf", user_id=7)

    await handle_delete_command(message, store)

    assert store.delete_calls == [(7, "vacation_policy.pdf")]
    assert "vacation_policy.pdf" in message.sent[0]
    assert message.sent[0].startswith("✅")


@pytest.mark.asyncio
async def test_delete_command_reports_a_missing_document():
    store = FakeStore(delete_result=False)
    message = FakeMessage(text="/delete nope.pdf", user_id=7)

    await handle_delete_command(message, store)

    assert message.sent[0].startswith("❌")
    assert "nope.pdf" in message.sent[0]


@pytest.mark.asyncio
async def test_delete_command_without_a_filename_shows_usage():
    store = FakeStore()
    message = FakeMessage(text="/delete", user_id=7)

    await handle_delete_command(message, store)

    assert message.sent == [DELETE_USAGE_MESSAGE]
    assert store.delete_calls == []


@pytest.mark.asyncio
async def test_delete_command_handles_a_storage_error_gracefully():
    store = FakeStore(delete_error=SQLiteStoreError("db is locked"))
    message = FakeMessage(text="/delete policy.pdf", user_id=7)

    await handle_delete_command(message, store)

    assert not any("db is locked" in text for text in message.sent)
    assert message.sent[0].startswith("❌")


@pytest.mark.asyncio
async def test_documents_command_handles_a_storage_error_gracefully():
    class BoomStore:
        def list_documents(self, user_id):
            raise SQLiteStoreError("db is locked")

    message = FakeMessage(user_id=1)
    await handle_documents_command(message, BoomStore())

    assert not any("db is locked" in text for text in message.sent)
    assert message.sent[0].startswith("❌")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/telegram_bot/handlers/test_commands.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telegram_bot.handlers.commands'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/telegram_bot/handlers/commands.py
"""/documents and /delete command handlers. See spec/v3/SPEC.md §10.2.

Both handlers scope every store call to message.from_user.id, so a user can
neither list nor delete another user's documents even if they know the exact
filename (assignment §10 — the store enforces this too, but keeping the
handler explicit means the isolation isn't reliant on one layer alone).
"""

import logging

from telegram_bot.errors import message_for
from userdocs.errors import SQLiteStoreError

logger = logging.getLogger(__name__)

NO_DOCUMENTS_MESSAGE = "📚 You haven't uploaded any documents yet."
DELETE_USAGE_MESSAGE = "Usage: /delete <filename>\n\nExample: /delete vacation_policy.pdf"


async def handle_documents_command(message, store) -> None:
    try:
        documents = store.list_documents(message.from_user.id)
    except SQLiteStoreError as exc:
        logger.warning("Could not list documents for user %s: %s", message.from_user.id, exc)
        await message.answer(message_for(exc))
        return

    if not documents:
        await message.answer(NO_DOCUMENTS_MESSAGE)
        return

    lines = ["📚 Your documents:", ""]
    for index, document in enumerate(documents, start=1):
        lines.append(f"{index}. {document['filename']}")
    await message.answer("\n".join(lines))


async def handle_delete_command(message, store) -> None:
    filename = (message.text or "").removeprefix("/delete").strip()
    if not filename:
        await message.answer(DELETE_USAGE_MESSAGE)
        return

    try:
        deleted = store.delete_document(message.from_user.id, filename)
    except SQLiteStoreError as exc:
        logger.warning("Could not delete %s for user %s: %s", filename, message.from_user.id, exc)
        await message.answer(message_for(exc))
        return

    if deleted:
        await message.answer(f"✅ Deleted {filename}.")
    else:
        await message.answer(f"❌ No document named {filename} found.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/telegram_bot/handlers/test_commands.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add src/telegram_bot/handlers/commands.py tests/telegram_bot/handlers/test_commands.py
git commit -m "feat(telegram_bot): add /documents and /delete commands"
```

---

### Task 8: Text message handler → agent loop

**Files:**
- Create: `src/telegram_bot/handlers/chat.py`
- Create: `tests/telegram_bot/handlers/test_chat.py`

**Interfaces:**
- Consumes: `userdocs.agent.run` (retrieval plan Task 10), `userdocs.tools.build_search_documents_tool` (retrieval plan Task 9), `userdocs.tool_registry.ToolRegistry` (retrieval plan Task 8), `telegram_bot.session.SessionStore` (conversation-aware-RAG plan Task 1), `telegram_bot.errors.message_for`.
- Produces: `async telegram_bot.handlers.chat.handle_text(message, store, llm, sessions) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/telegram_bot/handlers/test_chat.py
import pytest

from telegram_bot.errors import ERROR_MESSAGES
from telegram_bot.handlers.chat import handle_text
from telegram_bot.llm_errors import LLMError, LLMTimeoutError


class FakeMessage:
    def __init__(self, text="how many vacation days?", user_id=1):
        self.text = text
        self.from_user = type("FakeUser", (), {"id": user_id})()
        self.sent = []

    async def answer(self, text):
        self.sent.append(text)


class FakeSession:
    def __init__(self):
        self.appended = []

    def messages(self):
        return []

    def append_user_message(self, text):
        self.appended.append(("user", text))

    def append_assistant_message(self, text):
        self.appended.append(("assistant", text))

    def append_tool_result(self, call, result):
        self.appended.append(("tool", result))


class FakeSessionStore:
    def __init__(self):
        self.sessions = {}

    def get_or_create(self, user_id):
        return self.sessions.setdefault(user_id, FakeSession())


@pytest.mark.asyncio
async def test_handle_text_replies_with_the_agents_answer(monkeypatch):
    import telegram_bot.handlers.chat as chat_module

    async def fake_run(llm, registry, session, user_text, max_steps=4):
        return "You get 25 days. Source: policy.pdf"

    monkeypatch.setattr(chat_module, "agent_run", fake_run)

    message = FakeMessage()
    sessions = FakeSessionStore()
    await handle_text(message, store=None, llm=None, sessions=sessions)

    assert message.sent == ["You get 25 days. Source: policy.pdf"]


@pytest.mark.asyncio
async def test_handle_text_records_both_sides_of_the_turn_in_the_session(monkeypatch):
    import telegram_bot.handlers.chat as chat_module

    async def fake_run(llm, registry, session, user_text, max_steps=4):
        return "25 days."

    monkeypatch.setattr(chat_module, "agent_run", fake_run)

    message = FakeMessage(text="how many vacation days?", user_id=9)
    sessions = FakeSessionStore()
    await handle_text(message, store=None, llm=None, sessions=sessions)

    session = sessions.sessions[9]
    assert ("user", "how many vacation days?") in session.appended
    assert ("assistant", "25 days.") in session.appended


@pytest.mark.asyncio
async def test_handle_text_builds_a_tool_registry_bound_to_the_sender(monkeypatch):
    import telegram_bot.handlers.chat as chat_module

    seen = {}

    def fake_build_tool(store, user_id):
        seen["user_id"] = user_id
        return type("T", (), {"name": "search_documents", "schema": lambda self: {}})()

    async def fake_run(llm, registry, session, user_text, max_steps=4):
        return "ok"

    monkeypatch.setattr(chat_module, "build_search_documents_tool", fake_build_tool)
    monkeypatch.setattr(chat_module, "agent_run", fake_run)

    message = FakeMessage(user_id=4242)
    await handle_text(message, store=None, llm=None, sessions=FakeSessionStore())

    assert seen["user_id"] == 4242


@pytest.mark.parametrize("error_type", [LLMError, LLMTimeoutError])
@pytest.mark.asyncio
async def test_handle_text_replies_with_the_mapped_message_on_llm_failure(monkeypatch, error_type):
    import telegram_bot.handlers.chat as chat_module

    async def failing_run(llm, registry, session, user_text, max_steps=4):
        raise error_type("connection refused to ollama at localhost:11434")

    monkeypatch.setattr(chat_module, "agent_run", failing_run)

    message = FakeMessage()
    await handle_text(message, store=None, llm=None, sessions=FakeSessionStore())

    assert message.sent == [ERROR_MESSAGES[error_type]]
    assert not any("localhost:11434" in text for text in message.sent)


@pytest.mark.asyncio
async def test_handle_text_does_not_record_an_assistant_message_when_the_llm_fails(monkeypatch):
    """A failed turn must not pollute the conversation history with a
    non-answer, or the next follow-up question inherits it as context."""
    import telegram_bot.handlers.chat as chat_module

    async def failing_run(llm, registry, session, user_text, max_steps=4):
        raise LLMError("boom")

    monkeypatch.setattr(chat_module, "agent_run", failing_run)

    message = FakeMessage(user_id=3)
    sessions = FakeSessionStore()
    await handle_text(message, store=None, llm=None, sessions=sessions)

    roles = [role for role, _text in sessions.sessions[3].appended]
    assert "assistant" not in roles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/telegram_bot/handlers/test_chat.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telegram_bot.handlers.chat'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/telegram_bot/handlers/chat.py
"""Plain-text message handler. See spec/v3/SPEC.md §10.3.

The tool registry is built fresh per message, with the sender's user_id baked
into the search_documents closure — the LLM never sees a user_id parameter it
could change (spec §11.1).
"""

import logging

from telegram_bot.errors import message_for
from telegram_bot.llm_errors import LLMError
from userdocs.agent import run as agent_run
from userdocs.config import AGENT_MAX_STEPS
from userdocs.tool_registry import ToolRegistry
from userdocs.tools import build_search_documents_tool

logger = logging.getLogger(__name__)


async def handle_text(message, store, llm, sessions) -> None:
    user_id = message.from_user.id
    session = sessions.get_or_create(user_id)
    session.append_user_message(message.text)

    registry = ToolRegistry([build_search_documents_tool(store, user_id)])

    try:
        answer = await agent_run(
            llm, registry, session, message.text, max_steps=AGENT_MAX_STEPS
        )
    except LLMError as exc:
        logger.warning("Agent run failed for user %s: %s", user_id, exc)
        await message.answer(message_for(exc))
        return

    session.append_assistant_message(answer)
    await message.answer(answer)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/telegram_bot/handlers/test_chat.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/telegram_bot/handlers/chat.py tests/telegram_bot/handlers/test_chat.py
git commit -m "feat(telegram_bot): route text questions through the agent tool-use loop"
```

---

### Task 9: Auth middleware

**Files:**
- Create: `src/telegram_bot/middleware.py`
- Create: `tests/telegram_bot/test_middleware.py`

**Interfaces:**
- Consumes: `telegram_bot.config.Config` (Task 1).
- Produces: `telegram_bot.middleware.AuthMiddleware(allowed_user_ids: frozenset)` — an aiogram-style `async __call__(handler, event, data)` that calls `handler` when allowed and returns `None` without calling it when not. An empty `allowed_user_ids` means "no allowlist configured, allow everyone."

- [ ] **Step 1: Write the failing test**

```python
# tests/telegram_bot/test_middleware.py
import pytest

from telegram_bot.middleware import AuthMiddleware


class FakeEvent:
    def __init__(self, user_id=1):
        self.from_user = type("FakeUser", (), {"id": user_id})()


@pytest.mark.asyncio
async def test_middleware_calls_handler_for_an_allowed_user():
    called = []

    async def handler(event, data):
        called.append(event)
        return "handled"

    middleware = AuthMiddleware(frozenset({1, 2}))
    result = await middleware(handler, FakeEvent(user_id=1), {})

    assert result == "handled"
    assert len(called) == 1


@pytest.mark.asyncio
async def test_middleware_blocks_a_disallowed_user():
    called = []

    async def handler(event, data):
        called.append(event)

    middleware = AuthMiddleware(frozenset({1, 2}))
    result = await middleware(handler, FakeEvent(user_id=99), {})

    assert result is None
    assert called == []


@pytest.mark.asyncio
async def test_middleware_allows_everyone_when_no_allowlist_configured():
    called = []

    async def handler(event, data):
        called.append(event)
        return "handled"

    middleware = AuthMiddleware(frozenset())
    result = await middleware(handler, FakeEvent(user_id=12345), {})

    assert result == "handled"
    assert len(called) == 1


@pytest.mark.asyncio
async def test_middleware_blocks_an_event_with_no_sender_when_an_allowlist_exists():
    called = []

    async def handler(event, data):
        called.append(event)

    class SenderlessEvent:
        from_user = None

    middleware = AuthMiddleware(frozenset({1}))
    result = await middleware(handler, SenderlessEvent(), {})

    assert result is None
    assert called == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/telegram_bot/test_middleware.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telegram_bot.middleware'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/telegram_bot/middleware.py
"""Allowlist auth middleware, adapted from ../telegram-bot/main.py.

An empty allowlist means no allowlist is configured, so everyone is allowed —
this bot's isolation guarantee is per-user data separation (spec §10), not
access control, so refusing every user by default would be wrong. When an
allowlist IS configured, an event with no identifiable sender is blocked:
there's no user_id to check it against.
"""

import logging

logger = logging.getLogger(__name__)


class AuthMiddleware:
    def __init__(self, allowed_user_ids: frozenset):
        self._allowed_user_ids = allowed_user_ids

    async def __call__(self, handler, event, data):
        if not self._allowed_user_ids:
            return await handler(event, data)

        from_user = getattr(event, "from_user", None)
        user_id = getattr(from_user, "id", None) if from_user is not None else None

        if user_id is None or user_id not in self._allowed_user_ids:
            logger.info("Blocked message from unauthorized user_id=%s", user_id)
            return None

        return await handler(event, data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/telegram_bot/test_middleware.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/telegram_bot/middleware.py tests/telegram_bot/test_middleware.py
git commit -m "feat(telegram_bot): add allowlist auth middleware"
```

---

### Task 10: Bot bootstrap and polling entrypoint

**Files:**
- Create: `src/telegram_bot/main.py`
- Create: `tests/telegram_bot/test_main.py`
- Modify: `.gitignore`
- Create: `.env.example`

**Interfaces:**
- Consumes: everything above, plus `UserDocsStore` and `SessionStore`.
- Produces: `telegram_bot.main.build_dispatcher(store, llm, sessions, allowed_user_ids) -> aiogram.Dispatcher`; `async telegram_bot.main.run_bot() -> None`; a `if __name__ == "__main__":` entrypoint.

> Note: `build_dispatcher` is factored out from `run_bot` specifically so it can be tested without a bot token or network — the test below asserts the routes are registered, which is the part that's easy to get wrong (e.g. registering the catch-all text handler before the command handlers, which would swallow `/documents`).

- [ ] **Step 1: Write the failing test**

```python
# tests/telegram_bot/test_main.py
from telegram_bot.main import build_dispatcher


def _handler_names(dispatcher):
    return [handler.callback.__name__ for handler in dispatcher.message.handlers]


def test_build_dispatcher_registers_all_four_message_routes():
    dispatcher = build_dispatcher(store=None, llm=None, sessions=None, allowed_user_ids=frozenset())
    names = _handler_names(dispatcher)

    assert "documents_route" in names
    assert "delete_route" in names
    assert "document_route" in names
    assert "text_route" in names


def test_command_routes_are_registered_before_the_catch_all_text_route():
    """aiogram checks handlers in registration order and dispatches to the
    first whose filters match. F.text matches "/documents" too, so if the
    catch-all text route were registered first it would swallow both commands
    and they would silently never run."""
    dispatcher = build_dispatcher(store=None, llm=None, sessions=None, allowed_user_ids=frozenset())
    names = _handler_names(dispatcher)

    assert names.index("documents_route") < names.index("text_route")
    assert names.index("delete_route") < names.index("text_route")


def test_build_dispatcher_registers_an_auth_middleware():
    from telegram_bot.middleware import AuthMiddleware

    dispatcher = build_dispatcher(store=None, llm=None, sessions=None, allowed_user_ids=frozenset({1}))
    registered = list(dispatcher.message.middleware)

    assert any(isinstance(middleware, AuthMiddleware) for middleware in registered)
```

> Note for the implementer on the third test: it relies on aiogram's
> `MiddlewareManager` being iterable. If `list(dispatcher.message.middleware)`
> raises `TypeError` on the installed aiogram version, assert against
> `dispatcher.message.middleware._middlewares` instead and leave a comment
> saying why — the behavior being pinned (that an `AuthMiddleware` is actually
> attached, not just constructed) is worth a private-attribute read, since a
> silently-unregistered auth middleware means the allowlist does nothing.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/telegram_bot/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telegram_bot.main'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/telegram_bot/main.py
"""Bot bootstrap and long-polling entrypoint. See spec/v3/SPEC.md §10.

Handler registration order matters: aiogram dispatches to the first matching
handler, so the command routes are registered before the catch-all text route.

Dependencies (store, llm, sessions) are closed over by the route functions
rather than pulled from module globals, which is what lets build_dispatcher be
tested with None placeholders and no bot token.
"""

import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command

from telegram_bot.config import load_config
from telegram_bot.handlers.chat import handle_text
from telegram_bot.handlers.commands import handle_delete_command, handle_documents_command
from telegram_bot.handlers.documents import handle_document
from telegram_bot.llm_client import OllamaClient
from telegram_bot.middleware import AuthMiddleware
from telegram_bot.session import SessionStore
from userdocs.store import UserDocsStore

logger = logging.getLogger(__name__)


def build_dispatcher(store, llm, sessions, allowed_user_ids) -> Dispatcher:
    dispatcher = Dispatcher()
    dispatcher.message.middleware(AuthMiddleware(allowed_user_ids))

    @dispatcher.message(Command("documents"))
    async def documents_route(message):
        await handle_documents_command(message, store)

    @dispatcher.message(Command("delete"))
    async def delete_route(message):
        await handle_delete_command(message, store)

    @dispatcher.message(F.document)
    async def document_route(message):
        await handle_document(message, store)

    @dispatcher.message(F.text)
    async def text_route(message):
        await handle_text(message, store, llm, sessions)

    return dispatcher


async def run_bot() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()

    store = UserDocsStore(config.userdocs_db_path)
    sessions = SessionStore(config.userdocs_db_path)
    llm = OllamaClient(
        base_url=config.ollama_base_url,
        model=config.ollama_model,
        timeout_seconds=config.llm_timeout_seconds,
    )
    bot = Bot(token=config.bot_token)
    dispatcher = build_dispatcher(store, llm, sessions, config.allowed_user_ids)

    try:
        await dispatcher.start_polling(bot)
    finally:
        await llm.close()
        store.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(run_bot())
```

Create `.env.example`:

```text
# Required
TELEGRAM_BOT_TOKEN=123456:replace-me

# Optional — comma-separated Telegram user ids. Empty means anyone may use the bot.
ALLOWED_USER_IDS=

# Optional — defaults shown
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
LLM_TIMEOUT_SECONDS=120
USERDOCS_DB_PATH=userdocs.db
```

Add to `.gitignore` (verify `*.db` is already covered — if it is, only `.env` needs adding):

```text
.env
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/telegram_bot/test_main.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/telegram_bot/main.py tests/telegram_bot/test_main.py .env.example .gitignore
git commit -m "feat(telegram_bot): add dispatcher wiring and polling entrypoint"
```

---

## Summary

At the end of this plan the bot runs end-to-end: upload a document (with live in-place progress — **bonus +1**), ask a question, get an answer with source attribution, list with `/documents`, remove with `/delete`. All 10 error categories from assignment §12 have a typed exception, exactly one catch site, a friendly fixed message, and a test proving the raw exception text never reaches the user:

| Category | Caught in | Test |
|---|---|---|
| #1 unsupported format | `handlers/documents.py` | `test_documents.py` (parametrized) |
| #2 corrupt PDF | `handlers/documents.py` | same |
| #3 corrupt DOCX | `handlers/documents.py` | same |
| #4 empty document | `handlers/documents.py` | same |
| #5 too large | `handlers/documents.py` | same |
| #6 embedding error | `handlers/documents.py` | same |
| #7 SQLite error | `handlers/documents.py`, `handlers/commands.py` | same + `test_commands.py` |
| #8 LLM error | `handlers/chat.py` | `test_chat.py` (parametrized) |
| #9 timeout | `handlers/chat.py` | same |
| #10 Telegram API error | `handlers/documents.py`'s `on_progress`; aiogram's own error middleware elsewhere | `test_handle_document_survives_a_failed_progress_edit` |

Remaining: `docs/superpowers/plans/2026-09-10-conversation-aware-rag.md` (the `SessionStore` Task 8 and Task 10 import) and `2026-09-10-testing-and-evaluation.md` (end-to-end test, eval dataset, README).
