# LLM/Tool Telemetry & Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument every LLM call and MCP tool call the assistant makes with telemetry (tokens, cost, latency), add a real normalized-match response cache, and expose a `dashboard`/`timeline` CLI for viewing the collected data.

**Architecture:** A new `src/telemetry/` package provides a single generic middleware function per call kind (`llm_middleware.record_llm_call`, `tool_middleware.record_tool_call`) that all 3 LLM call sites and the 1 tool call site route through. `contextvars` propagate `agent_id`/`task_id`/`turn_number` ambiently so nested calls (query expansion calling `ask_llm`) don't need it threaded through signatures. A SQLite DB (`src/telemetry.db`) holds `llm_calls`, `tool_calls`, and `llm_cache` tables. `main.py` gains `dashboard` and `timeline` subcommands rendered with `rich.table.Table`.

**Tech Stack:** Python 3.10+, sqlite3 (stdlib), `contextvars` (stdlib), `rich.table.Table` (existing dependency), pytest + monkeypatch/tmp_path (existing test stack).

**Spec:** `spec/v2/SPEC.md`

## Global Constraints

- Implementation follows TDD (test-first) per `superpowers:test-driven-development`.
- Tests mirror source structure 1:1: `src/telemetry/<module>.py` → `tests/telemetry/test_<module>.py`, matching this repo's existing `tests/rag/test_*.py` convention.
- No new LLM/tool call sites are being added — exactly 3 LLM call sites (`rag/query.py::ask_llm`, `assistant.py::_llm_decide_mcp_usage`'s `ollama.Client().chat()`, and `rag/expand.py::generate_keywords`'s indirect use of `ask_llm`) and 1 tool call site (`mcp/client.py::MCPClient.call_tool`) get wrapped.
- `*.db` is already gitignored repo-wide — `telemetry.db` needs no new gitignore entry.
- Scope is Part 1 (collection + viewing) only: no alerting, no retention/rotation, no multi-process concurrency safety beyond SQLite's own file locking.
- Modules resolving paths from `config.py` must re-import the constant with a *local* `from config import X` inside the function body (not a module-level import), so tests can `monkeypatch.setattr("config.X", ...)` and have it take effect — this repo's existing pattern in `rag/build_index.py` for `FTS_DB_PATH`.

---

## File Structure

```
src/config.py                    # + TELEMETRY_DB_PATH, AGENT_ID constants
src/telemetry/
  __init__.py                    # empty, marks the package
  context.py                     # contextvars: current_task_id/turn_number/agent_id
  storage.py                     # sqlite connection + schema + insert_llm_call/insert_tool_call
  pricing.py                     # PRICING table + estimate_cost()
  llm_cache.py                   # normalize (reused from rag.expand), cache key, get/put/clear
  llm_middleware.py              # record_llm_call() — the single LLM-call instrumentation point
  tool_middleware.py             # record_tool_call() — the single tool-call instrumentation point
  dashboard.py                   # aggregate + comparison queries and rich rendering
  timeline.py                    # per-task query and rich rendering
src/rag/query.py                 # ask_llm() gains call_site param, routes through record_llm_call
src/rag/expand.py                # generate_keywords() passes call_site="query_expansion"
src/rag/build_index.py           # build_index() clears llm_cache on every run
src/assistant.py                 # _llm_decide_mcp_usage wraps its chat() call; task/turn lifecycle
src/mcp/client.py                # call_tool() routes through record_tool_call
src/main.py                      # argparse-based dispatch incl. new dashboard/timeline subcommands
tests/conftest.py                 # autouse fixture redirecting telemetry writes to a tmp DB
tests/test_config.py              # + new constants
tests/telemetry/test_context.py
tests/telemetry/test_storage.py
tests/telemetry/test_pricing.py
tests/telemetry/test_llm_cache.py
tests/telemetry/test_llm_middleware.py
tests/telemetry/test_tool_middleware.py
tests/telemetry/test_dashboard.py
tests/telemetry/test_timeline.py
tests/rag/test_ask_llm.py         # + call_site / telemetry integration tests
tests/rag/test_expand.py          # + call_site pass-through test
tests/rag/test_build_index.py     # + cache-clear-on-build test
tests/test_assistant.py           # new
tests/mcp/test_client.py          # new
tests/test_main.py                # new
```

---

### Task 1: Config constants + test-isolation fixture

**Files:**
- Modify: `src/config.py`
- Create: `tests/conftest.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `config.TELEMETRY_DB_PATH` (str, `"telemetry.db"`), `config.AGENT_ID` (str, `"company-kb-assistant"`) — every later task reads these.
- Produces: an autouse pytest fixture that redirects `config.TELEMETRY_DB_PATH` to a per-test tmp file, so no test run ever writes to the real project `src/telemetry.db`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py — append to the existing file
def test_telemetry_config_constants():
    assert config.TELEMETRY_DB_PATH == "telemetry.db"
    assert config.AGENT_ID == "company-kb-assistant"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_telemetry_config_constants -v`
Expected: FAIL with `AttributeError: module 'config' has no attribute 'TELEMETRY_DB_PATH'`

- [ ] **Step 3: Add the constants**

```python
# src/config.py — append
# Telemetry configuration
TELEMETRY_DB_PATH = "telemetry.db"   # relative to src dir, mirrors FTS_DB_PATH
AGENT_ID = "company-kb-assistant"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_telemetry_config_constants -v`
Expected: PASS

- [ ] **Step 5: Add the test-isolation conftest fixture**

```python
# tests/conftest.py
import pytest


@pytest.fixture(autouse=True)
def isolated_telemetry_db(tmp_path, monkeypatch):
    """Redirect every test's telemetry writes to a per-test tmp file so test
    runs never touch (or get polluted by) the real project telemetry.db.
    storage.py re-imports config.TELEMETRY_DB_PATH locally on every call, so
    this monkeypatch takes effect even for code that doesn't accept a
    db_path override."""
    monkeypatch.setattr("config.TELEMETRY_DB_PATH", str(tmp_path / "telemetry.db"))
```

- [ ] **Step 6: Run the full test suite to confirm the fixture doesn't break anything yet**

Run: `pytest -q`
Expected: PASS (fixture is inert until later tasks add code that reads `config.TELEMETRY_DB_PATH`)

- [ ] **Step 7: Commit**

```bash
git add src/config.py tests/test_config.py tests/conftest.py
git commit -m "feat: add telemetry config constants and test-isolation fixture"
```

---

### Task 2: `telemetry.context` — ambient agent/task/turn context

**Files:**
- Create: `src/telemetry/__init__.py`
- Create: `src/telemetry/context.py`
- Test: `tests/telemetry/test_context.py`

**Interfaces:**
- Consumes: `config.AGENT_ID` (Task 1).
- Produces: `context.current_agent_id`, `context.current_task_id`, `context.current_turn_number` (all `contextvars.ContextVar`), `context.start_task(task_id=None) -> str`, `context.next_turn() -> int`. Every later task that records a row reads these three ContextVars via `.get()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/telemetry/test_context.py
import uuid

from telemetry import context


def test_current_agent_id_defaults_to_config_agent_id():
    assert context.current_agent_id.get() == "company-kb-assistant"


def test_start_task_generates_a_uuid4_and_resets_turn_number():
    context.current_turn_number.set(7)

    task_id = context.start_task()

    assert uuid.UUID(task_id).version == 4
    assert context.current_task_id.get() == task_id
    assert context.current_turn_number.get() == 0


def test_start_task_accepts_an_explicit_task_id():
    task_id = context.start_task("explicit-id-123")

    assert task_id == "explicit-id-123"
    assert context.current_task_id.get() == "explicit-id-123"


def test_next_turn_increments_from_the_current_value():
    context.start_task("t1")

    assert context.next_turn() == 1
    assert context.next_turn() == 2
    assert context.current_turn_number.get() == 2


def test_nested_calls_see_the_ambient_context():
    context.start_task("t2")
    context.next_turn()

    def inner():
        return context.current_task_id.get(), context.current_turn_number.get()

    assert inner() == ("t2", 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/telemetry/test_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telemetry'`

- [ ] **Step 3: Create the package and implement context.py**

```python
# src/telemetry/__init__.py
```

```python
# src/telemetry/context.py
import contextvars
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AGENT_ID

current_agent_id = contextvars.ContextVar("current_agent_id", default=AGENT_ID)
current_task_id = contextvars.ContextVar("current_task_id", default=None)
current_turn_number = contextvars.ContextVar("current_turn_number", default=0)


def start_task(task_id=None):
    """Begin a new task: generate (or accept) a task_id, reset the turn
    counter to 0, and set both as the ambient context. Returns the task_id."""
    task_id = task_id or str(uuid.uuid4())
    current_task_id.set(task_id)
    current_turn_number.set(0)
    return task_id


def next_turn():
    """Advance to the next turn within the current task. Returns the new
    turn number."""
    turn = current_turn_number.get() + 1
    current_turn_number.set(turn)
    return turn
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/telemetry/test_context.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/__init__.py src/telemetry/context.py tests/telemetry/test_context.py
git commit -m "feat: add telemetry.context ambient agent/task/turn tracking"
```

---

### Task 3: `telemetry.storage` — schema + row inserts

**Files:**
- Create: `src/telemetry/storage.py`
- Test: `tests/telemetry/test_storage.py`

**Interfaces:**
- Consumes: `config.TELEMETRY_DB_PATH` (Task 1, re-imported locally per Global Constraints).
- Produces: `storage.now_iso() -> str`, `storage.get_connection(db_path=None) -> sqlite3.Connection` (schema already ensured), `storage.insert_llm_call(agent_id, task_id, turn_number, model, input_tokens, output_tokens, cached_tokens, reasoning_tokens, latency_ms, estimated_cost, call_site, timestamp=None, db_path=None)`, `storage.insert_tool_call(agent_id, task_id, turn_number, tool_name, input_size, output_size, output_tokens, duration_ms, timestamp=None, db_path=None)`. Later tasks (`llm_cache`, `llm_middleware`, `tool_middleware`, `dashboard`, `timeline`) all call `storage.get_connection` and the two insert functions with exactly these keyword names.

- [ ] **Step 1: Write the failing tests**

```python
# tests/telemetry/test_storage.py
import sqlite3

from telemetry import storage


def test_get_connection_creates_all_three_tables(tmp_path):
    db_path = tmp_path / "telemetry.db"

    conn = storage.get_connection(db_path)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    conn.close()

    assert {"llm_calls", "tool_calls", "llm_cache"} <= tables


def test_get_connection_is_idempotent_across_calls(tmp_path):
    db_path = tmp_path / "telemetry.db"

    storage.get_connection(db_path).close()
    storage.get_connection(db_path).close()  # must not raise on re-create


def test_get_connection_with_no_db_path_resolves_from_config(tmp_path, monkeypatch):
    monkeypatch.setattr("config.TELEMETRY_DB_PATH", str(tmp_path / "default.db"))

    conn = storage.get_connection()
    conn.close()

    assert (tmp_path / "default.db").exists()


def test_insert_llm_call_writes_a_row(tmp_path):
    db_path = tmp_path / "telemetry.db"

    storage.insert_llm_call(
        agent_id="company-kb-assistant", task_id="t1", turn_number=1,
        model="qwen3:0.6b", input_tokens=10, output_tokens=20,
        cached_tokens=0, reasoning_tokens=0, latency_ms=123.4,
        estimated_cost=0.000006, call_site="ask_llm", db_path=db_path,
    )

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT agent_id, task_id, turn_number, model, input_tokens, "
        "output_tokens, cached_tokens, reasoning_tokens, call_site "
        "FROM llm_calls"
    ).fetchone()
    conn.close()

    assert row == (
        "company-kb-assistant", "t1", 1, "qwen3:0.6b", 10, 20, 0, 0, "ask_llm",
    )


def test_insert_llm_call_defaults_timestamp_to_now_iso(tmp_path):
    db_path = tmp_path / "telemetry.db"

    storage.insert_llm_call(
        agent_id="a", task_id="t1", turn_number=1, model="m",
        input_tokens=1, output_tokens=1, cached_tokens=0, reasoning_tokens=0,
        latency_ms=1.0, estimated_cost=0.0, call_site="ask_llm", db_path=db_path,
    )

    conn = sqlite3.connect(str(db_path))
    timestamp = conn.execute("SELECT timestamp FROM llm_calls").fetchone()[0]
    conn.close()

    assert timestamp  # non-empty ISO string


def test_insert_tool_call_writes_a_row(tmp_path):
    db_path = tmp_path / "telemetry.db"

    storage.insert_tool_call(
        agent_id="company-kb-assistant", task_id="t1", turn_number=2,
        tool_name="list_documents", input_size=2, output_size=50,
        output_tokens=12, duration_ms=45.0, db_path=db_path,
    )

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT agent_id, task_id, turn_number, tool_name, input_size, "
        "output_size, output_tokens FROM tool_calls"
    ).fetchone()
    conn.close()

    assert row == ("company-kb-assistant", "t1", 2, "list_documents", 2, 50, 12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/telemetry/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telemetry.storage'`

- [ ] **Step 3: Implement storage.py**

```python
# src/telemetry/storage.py
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL,
    estimated_cost REAL NOT NULL,
    call_site TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    input_size INTEGER NOT NULL,
    output_size INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    duration_ms REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    response TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _default_db_path():
    # Local import (not a module-level `from config import ...`) so tests
    # can monkeypatch config.TELEMETRY_DB_PATH and have it take effect here —
    # mirrors rag/build_index.py's FTS_DB_PATH pattern.
    from config import TELEMETRY_DB_PATH
    src_dir = Path(__file__).parent.parent
    return src_dir / TELEMETRY_DB_PATH


def get_connection(db_path=None):
    """Open a new connection to the telemetry DB, creating the schema if it
    doesn't exist yet. Callers are responsible for closing it."""
    resolved = Path(db_path) if db_path else _default_db_path()
    conn = sqlite3.connect(str(resolved))
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def insert_llm_call(
    agent_id, task_id, turn_number, model, input_tokens, output_tokens,
    cached_tokens, reasoning_tokens, latency_ms, estimated_cost, call_site,
    timestamp=None, db_path=None,
):
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO llm_calls (
                timestamp, agent_id, task_id, turn_number, model,
                input_tokens, output_tokens, cached_tokens, reasoning_tokens,
                latency_ms, estimated_cost, call_site
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp or now_iso(), agent_id, task_id, turn_number, model,
                input_tokens, output_tokens, cached_tokens, reasoning_tokens,
                latency_ms, estimated_cost, call_site,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def insert_tool_call(
    agent_id, task_id, turn_number, tool_name, input_size, output_size,
    output_tokens, duration_ms, timestamp=None, db_path=None,
):
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO tool_calls (
                timestamp, agent_id, task_id, turn_number, tool_name,
                input_size, output_size, output_tokens, duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp or now_iso(), agent_id, task_id, turn_number,
                tool_name, input_size, output_size, output_tokens, duration_ms,
            ),
        )
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/telemetry/test_storage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/storage.py tests/telemetry/test_storage.py
git commit -m "feat: add telemetry.storage schema and row inserts"
```

---

### Task 4: `telemetry.pricing` — cost estimation

**Files:**
- Create: `src/telemetry/pricing.py`
- Test: `tests/telemetry/test_pricing.py`

**Interfaces:**
- Produces: `pricing.PRICING` (dict), `pricing.DEFAULT_PRICING` (dict), `pricing.estimate_cost(model, input_tokens, output_tokens) -> float`. `llm_middleware` (Task 6) calls `estimate_cost` with exactly these 3 positional args.

- [ ] **Step 1: Write the failing tests**

```python
# tests/telemetry/test_pricing.py
from telemetry import pricing


def test_estimate_cost_for_known_model():
    cost = pricing.estimate_cost("qwen3:0.6b", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 0.40  # $0.20/M in + $0.20/M out


def test_estimate_cost_zero_tokens_is_zero():
    assert pricing.estimate_cost("qwen3:0.6b", 0, 0) == 0.0


def test_estimate_cost_unknown_model_defaults_to_zero(monkeypatch, capsys):
    monkeypatch.setattr(pricing, "_warned_models", set())

    cost = pricing.estimate_cost("some-other-model", 1_000_000, 1_000_000)

    assert cost == 0.0
    assert "some-other-model" in capsys.readouterr().out


def test_estimate_cost_unknown_model_warns_only_once(monkeypatch, capsys):
    monkeypatch.setattr(pricing, "_warned_models", set())

    pricing.estimate_cost("some-other-model", 1, 1)
    pricing.estimate_cost("some-other-model", 1, 1)

    assert capsys.readouterr().out.count("some-other-model") == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/telemetry/test_pricing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telemetry.pricing'`

- [ ] **Step 3: Implement pricing.py**

```python
# src/telemetry/pricing.py
PRICING = {
    "qwen3:0.6b": {"input": 0.20, "output": 0.20},  # Fireworks AI via OpenRouter, Sep 2026
}
DEFAULT_PRICING = {"input": 0.0, "output": 0.0}

_warned_models = set()


def estimate_cost(model, input_tokens, output_tokens):
    """$ = input_tokens/1e6 * price_in + output_tokens/1e6 * price_out.
    Unknown models default to $0.00 with a one-time warning rather than
    raising."""
    prices = PRICING.get(model)
    if prices is None:
        if model not in _warned_models:
            print(f"⚠️  No pricing data for model '{model}'; defaulting to $0.00")
            _warned_models.add(model)
        prices = DEFAULT_PRICING

    return (input_tokens / 1_000_000) * prices["input"] + (output_tokens / 1_000_000) * prices["output"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/telemetry/test_pricing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/pricing.py tests/telemetry/test_pricing.py
git commit -m "feat: add telemetry.pricing cost estimation with unknown-model fallback"
```

---

### Task 5: `telemetry.llm_cache` — normalized-match response cache

**Files:**
- Create: `src/telemetry/llm_cache.py`
- Test: `tests/telemetry/test_llm_cache.py`

**Interfaces:**
- Consumes: `rag.expand._normalize` (existing), `storage.get_connection` (Task 3).
- Produces: `llm_cache.get(prompt, model, temperature, db_path=None) -> dict | None` (dict has keys `response`, `input_tokens`, `output_tokens`), `llm_cache.put(prompt, model, temperature, response, input_tokens, output_tokens, db_path=None)`, `llm_cache.clear(db_path=None)`. `llm_middleware` (Task 6) and `build_index.py` (Task 10) call these by these exact names.

- [ ] **Step 1: Write the failing tests**

```python
# tests/telemetry/test_llm_cache.py
from telemetry import llm_cache


def test_get_returns_none_on_miss(tmp_path):
    db_path = tmp_path / "telemetry.db"
    assert llm_cache.get("hello", "qwen3:0.6b", None, db_path=db_path) is None


def test_put_then_get_is_a_hit(tmp_path):
    db_path = tmp_path / "telemetry.db"
    llm_cache.put("hello world", "qwen3:0.6b", None, "the answer", 10, 5, db_path=db_path)

    result = llm_cache.get("hello world", "qwen3:0.6b", None, db_path=db_path)

    assert result == {"response": "the answer", "input_tokens": 10, "output_tokens": 5}


def test_normalization_collapses_whitespace_case_and_punctuation(tmp_path):
    db_path = tmp_path / "telemetry.db"
    llm_cache.put("Hello, World!", "qwen3:0.6b", None, "answer", 1, 1, db_path=db_path)

    assert llm_cache.get("  hello world  ", "qwen3:0.6b", None, db_path=db_path) is not None


def test_normalization_does_not_conflate_different_questions(tmp_path):
    db_path = tmp_path / "telemetry.db"
    llm_cache.put("what is the vacation policy", "qwen3:0.6b", None, "a1", 1, 1, db_path=db_path)

    assert llm_cache.get("what is the expense policy", "qwen3:0.6b", None, db_path=db_path) is None


def test_different_model_is_a_miss(tmp_path):
    db_path = tmp_path / "telemetry.db"
    llm_cache.put("q", "qwen3:0.6b", None, "a", 1, 1, db_path=db_path)

    assert llm_cache.get("q", "other-model", None, db_path=db_path) is None


def test_different_temperature_is_a_miss(tmp_path):
    db_path = tmp_path / "telemetry.db"
    llm_cache.put("q", "qwen3:0.6b", 0.1, "a", 1, 1, db_path=db_path)

    assert llm_cache.get("q", "qwen3:0.6b", 0.9, db_path=db_path) is None


def test_clear_removes_all_entries(tmp_path):
    db_path = tmp_path / "telemetry.db"
    llm_cache.put("q", "qwen3:0.6b", None, "a", 1, 1, db_path=db_path)

    llm_cache.clear(db_path=db_path)

    assert llm_cache.get("q", "qwen3:0.6b", None, db_path=db_path) is None


def test_put_overwrites_an_existing_key(tmp_path):
    db_path = tmp_path / "telemetry.db"
    llm_cache.put("q", "qwen3:0.6b", None, "first", 1, 1, db_path=db_path)
    llm_cache.put("q", "qwen3:0.6b", None, "second", 2, 2, db_path=db_path)

    assert llm_cache.get("q", "qwen3:0.6b", None, db_path=db_path) == {
        "response": "second", "input_tokens": 2, "output_tokens": 2,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/telemetry/test_llm_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telemetry.llm_cache'`

- [ ] **Step 3: Implement llm_cache.py**

```python
# src/telemetry/llm_cache.py
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
# Reuses rag/expand.py's whitespace/case/punctuation collapse (per spec) so
# the cache and query-expansion dedup logic never drift apart.
from rag.expand import _normalize as normalize
from telemetry import storage


def _cache_key(prompt, model, temperature):
    raw = f"{normalize(prompt)}{model}{temperature}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get(prompt, model, temperature, db_path=None):
    """Look up a cached response. Returns a dict with response/input_tokens/
    output_tokens on hit, None on miss."""
    key = _cache_key(prompt, model, temperature)
    conn = storage.get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT response, input_tokens, output_tokens FROM llm_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    return {"response": row[0], "input_tokens": row[1], "output_tokens": row[2]}


def put(prompt, model, temperature, response, input_tokens, output_tokens, db_path=None):
    key = _cache_key(prompt, model, temperature)
    conn = storage.get_connection(db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache "
            "(cache_key, model, response, input_tokens, output_tokens, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (key, model, response, input_tokens, output_tokens, storage.now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def clear(db_path=None):
    conn = storage.get_connection(db_path)
    try:
        conn.execute("DELETE FROM llm_cache")
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/telemetry/test_llm_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/llm_cache.py tests/telemetry/test_llm_cache.py
git commit -m "feat: add telemetry.llm_cache normalized-match response cache"
```

---

### Task 6: `telemetry.llm_middleware` — the single LLM instrumentation point

**Files:**
- Create: `src/telemetry/llm_middleware.py`
- Test: `tests/telemetry/test_llm_middleware.py`

**Interfaces:**
- Consumes: `context.current_agent_id/current_task_id/current_turn_number` (Task 2), `llm_cache.get/put` (Task 5), `pricing.estimate_cost` (Task 4), `storage.insert_llm_call` (Task 3).
- Produces: `record_llm_call(call_site, model, prompt, temperature, fn, db_path=None) -> str`, where `fn` is a zero-arg callable returning `(response_text, input_tokens, output_tokens)`. Tasks 7, 8, and 9's LLM-side wiring all call this with exactly this signature.

- [ ] **Step 1: Write the failing tests**

```python
# tests/telemetry/test_llm_middleware.py
from telemetry import context
from telemetry.llm_middleware import record_llm_call


def test_cache_miss_calls_fn_and_records_a_row(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t1")
    context.next_turn()
    calls = []

    def fn():
        calls.append(1)
        return "the answer", 10, 20

    result = record_llm_call(
        call_site="ask_llm", model="qwen3:0.6b", prompt="what is the policy",
        temperature=None, fn=fn, db_path=db_path,
    )

    assert result == "the answer"
    assert calls == [1]  # fn called exactly once on a miss

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT task_id, turn_number, model, input_tokens, output_tokens, "
        "cached_tokens, call_site FROM llm_calls"
    ).fetchone()
    conn.close()
    assert row == ("t1", 1, "qwen3:0.6b", 10, 20, 0, "ask_llm")


def test_cache_hit_does_not_call_fn(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t2")
    context.next_turn()

    record_llm_call(
        call_site="ask_llm", model="qwen3:0.6b", prompt="repeat question",
        temperature=None, fn=lambda: ("first answer", 10, 20), db_path=db_path,
    )

    calls = []

    def fn_should_not_run():
        calls.append(1)
        return "should not happen", 0, 0

    result = record_llm_call(
        call_site="ask_llm", model="qwen3:0.6b", prompt="repeat question",
        temperature=None, fn=fn_should_not_run, db_path=db_path,
    )

    assert result == "first answer"
    assert calls == []


def test_cache_hit_records_cached_tokens_equal_to_input_tokens(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t3")
    context.next_turn()
    record_llm_call(
        call_site="ask_llm", model="qwen3:0.6b", prompt="q",
        temperature=None, fn=lambda: ("a", 15, 7), db_path=db_path,
    )

    record_llm_call(
        call_site="ask_llm", model="qwen3:0.6b", prompt="q",
        temperature=None, fn=lambda: ("unused", 0, 0), db_path=db_path,
    )

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT cached_tokens FROM llm_calls ORDER BY id").fetchall()
    conn.close()
    assert rows == [(0,), (15,)]


def test_records_estimated_cost_via_pricing(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t4")
    context.next_turn()

    record_llm_call(
        call_site="ask_llm", model="qwen3:0.6b", prompt="q",
        temperature=None, fn=lambda: ("a", 1_000_000, 1_000_000), db_path=db_path,
    )

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    cost = conn.execute("SELECT estimated_cost FROM llm_calls").fetchone()[0]
    conn.close()
    assert cost == 0.40


def test_records_reasoning_tokens_as_zero(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t5")
    context.next_turn()

    record_llm_call(
        call_site="ask_llm", model="qwen3:0.6b", prompt="q",
        temperature=None, fn=lambda: ("a", 1, 1), db_path=db_path,
    )

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    reasoning = conn.execute("SELECT reasoning_tokens FROM llm_calls").fetchone()[0]
    conn.close()
    assert reasoning == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/telemetry/test_llm_middleware.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telemetry.llm_middleware'`

- [ ] **Step 3: Implement llm_middleware.py**

```python
# src/telemetry/llm_middleware.py
import time

from telemetry import context, llm_cache, pricing, storage


def record_llm_call(call_site, model, prompt, temperature, fn, db_path=None):
    """Look up (prompt, model, temperature) in the response cache; on a
    miss, call fn() (a zero-arg callable returning
    (response_text, input_tokens, output_tokens)) and store the result.
    Either way, record one row in llm_calls tagged with call_site. Returns
    the response text."""
    cached = llm_cache.get(prompt, model, temperature, db_path=db_path)
    start = time.perf_counter()

    if cached is not None:
        response_text = cached["response"]
        input_tokens = cached["input_tokens"]
        output_tokens = cached["output_tokens"]
        cached_tokens = input_tokens
    else:
        response_text, input_tokens, output_tokens = fn()
        cached_tokens = 0
        llm_cache.put(
            prompt, model, temperature, response_text,
            input_tokens, output_tokens, db_path=db_path,
        )

    latency_ms = (time.perf_counter() - start) * 1000
    estimated_cost = pricing.estimate_cost(model, input_tokens, output_tokens)

    storage.insert_llm_call(
        agent_id=context.current_agent_id.get(),
        task_id=context.current_task_id.get(),
        turn_number=context.current_turn_number.get(),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=0,
        latency_ms=latency_ms,
        estimated_cost=estimated_cost,
        call_site=call_site,
        db_path=db_path,
    )

    return response_text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/telemetry/test_llm_middleware.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/llm_middleware.py tests/telemetry/test_llm_middleware.py
git commit -m "feat: add telemetry.llm_middleware record_llm_call instrumentation point"
```

---

### Task 7: Wire `ask_llm()` and `generate_keywords()` through `record_llm_call`

**Files:**
- Modify: `src/rag/query.py`
- Modify: `src/rag/expand.py`
- Test: `tests/rag/test_ask_llm.py` (extend)
- Test: `tests/rag/test_expand.py` (extend)

**Interfaces:**
- Consumes: `telemetry.llm_middleware.record_llm_call` (Task 6).
- Produces: `ask_llm(prompt, temperature=None, timeout=None, call_site="ask_llm")` — same return type (`str`) as before; `call_site` is new and defaults to preserve existing callers' behavior unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag/test_ask_llm.py — append to the existing file
#
# config.TELEMETRY_DB_PATH is an absolute tmp path once tests/conftest.py's
# isolated_telemetry_db autouse fixture has run for this test, so
# sqlite3.connect(config.TELEMETRY_DB_PATH) opens the exact file ask_llm()'s
# telemetry write used.
import config
import sqlite3


def test_ask_llm_default_call_site_is_ask_llm(monkeypatch):
    monkeypatch.setattr(query.requests, "post", lambda *a, **k: _FakeResponse({"response": "answer"}))

    query.ask_llm("hello")

    conn = sqlite3.connect(config.TELEMETRY_DB_PATH)
    row = conn.execute("SELECT call_site FROM llm_calls ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    assert row == ("ask_llm",)


def test_ask_llm_extracts_token_counts_from_ollama_response(monkeypatch):
    monkeypatch.setattr(
        query.requests, "post",
        lambda *a, **k: _FakeResponse({"response": "answer", "prompt_eval_count": 11, "eval_count": 22}),
    )

    query.ask_llm("hello")

    conn = sqlite3.connect(config.TELEMETRY_DB_PATH)
    row = conn.execute(
        "SELECT input_tokens, output_tokens FROM llm_calls ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row == (11, 22)


def test_ask_llm_accepts_an_explicit_call_site(monkeypatch):
    monkeypatch.setattr(query.requests, "post", lambda *a, **k: _FakeResponse({"response": "answer"}))

    query.ask_llm("hello", call_site="query_expansion")

    conn = sqlite3.connect(config.TELEMETRY_DB_PATH)
    row = conn.execute("SELECT call_site FROM llm_calls ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    assert row == ("query_expansion",)
```

```python
# tests/rag/test_expand.py — append to the existing file
def test_passes_query_expansion_call_site_to_ask_llm_fn():
    captured = {}

    def fake_ask_llm(prompt, **kwargs):
        captured["kwargs"] = kwargs
        return "keyword"

    generate_keywords("q", ask_llm_fn=fake_ask_llm)

    assert captured["kwargs"]["call_site"] == "query_expansion"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/rag/test_ask_llm.py tests/rag/test_expand.py -v`
Expected: FAIL — `test_ask_llm_*` fail because no `llm_calls` rows exist yet (`ask_llm` doesn't record telemetry); `test_passes_query_expansion_call_site_to_ask_llm_fn` fails with `KeyError: 'call_site'`

- [ ] **Step 3: Wire query.py's ask_llm**

```python
# src/rag/query.py — add import near the top, alongside the other rag imports
from telemetry.llm_middleware import record_llm_call
```

```python
# src/rag/query.py — replace the existing ask_llm() definition
def ask_llm(prompt, temperature=None, timeout=None, call_site="ask_llm"):
    """Query Ollama LLM through the telemetry middleware (cache lookup,
    timing, cost estimation, and a recorded llm_calls row) — see
    telemetry.llm_middleware.record_llm_call.

    temperature: if not None, sent as {"options": {"temperature": temperature}}.
    timeout: if not None, passed as requests.post(..., timeout=timeout).
             If None, blocks indefinitely (unchanged existing behavior).
    call_site: which of the spec's 3 tagged LLM call sites this is. Defaults
    to "ask_llm" (the direct final-answer call from ask()); generate_keywords()
    in rag/expand.py overrides it to "query_expansion".
    """
    def _do_request():
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        if temperature is not None:
            payload["options"] = {"temperature": temperature}
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        data = response.json()
        return data["response"], data.get("prompt_eval_count", 0), data.get("eval_count", 0)

    return record_llm_call(
        call_site=call_site,
        model=OLLAMA_MODEL,
        prompt=prompt,
        temperature=temperature,
        fn=_do_request,
    )
```

- [ ] **Step 4: Wire expand.py's generate_keywords to pass call_site**

```python
# src/rag/expand.py — the single existing ask_llm_fn call inside generate_keywords()
    try:
        response = ask_llm_fn(prompt, temperature=0.1, timeout=8.0, call_site="query_expansion")
    except Exception:
        return []
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/rag/test_ask_llm.py tests/rag/test_expand.py -v`
Expected: PASS

- [ ] **Step 6: Run the full test suite to check for regressions**

Run: `pytest -q`
Expected: PASS (existing `ask_llm` tests that don't reference telemetry still pass unmodified — `ask_llm` still returns just the response string)

- [ ] **Step 7: Commit**

```bash
git add src/rag/query.py src/rag/expand.py tests/rag/test_ask_llm.py tests/rag/test_expand.py
git commit -m "feat: route ask_llm and query expansion through telemetry middleware"
```

---

### Task 8: Wire `assistant.py` — mcp_decision call site + task/turn lifecycle

**Files:**
- Modify: `src/assistant.py`
- Test: `tests/test_assistant.py` (new)

**Interfaces:**
- Consumes: `telemetry.context` (Task 2), `telemetry.llm_middleware.record_llm_call` (Task 6).
- Produces: `CompanyKBAssistant.task_id` (str, set in `__init__`); `_llm_decide_mcp_usage` unchanged return type (`(tool_name_or_None, tool_args_or_None)`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_assistant.py
import assistant as assistant_module
from assistant import CompanyKBAssistant
from telemetry import context


class _FakeMCPClient:
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _make_bare_assistant(llm_client=None, mcp=None):
    instance = CompanyKBAssistant.__new__(CompanyKBAssistant)
    instance.llm_client = llm_client
    instance.mcp = mcp
    return instance


def test_llm_decide_mcp_usage_returns_none_when_no_mcp_client():
    instance = _make_bare_assistant(llm_client=None, mcp=None)
    assert instance._llm_decide_mcp_usage("q", []) == (None, None)


def test_llm_decide_mcp_usage_wraps_chat_through_record_llm_call(monkeypatch):
    captured = {}

    def fake_record_llm_call(call_site, model, prompt, temperature, fn):
        captured["call_site"] = call_site
        captured["model"] = model
        captured["prompt"] = prompt
        captured["temperature"] = temperature
        text, input_tokens, output_tokens = fn()
        captured["fn_result"] = (text, input_tokens, output_tokens)
        return text

    monkeypatch.setattr(assistant_module, "record_llm_call", fake_record_llm_call)

    class _FakeLLMClient:
        def chat(self, model, messages):
            captured["chat_model"] = model
            captured["chat_messages"] = messages
            return {
                "message": {"content": '{"use_mcp": true, "tool": "list_documents", "args": {}}'},
                "prompt_eval_count": 12,
                "eval_count": 5,
            }

    instance = _make_bare_assistant(llm_client=_FakeLLMClient(), mcp=object())

    tool_name, tool_args = instance._llm_decide_mcp_usage("what docs exist?", [])

    assert tool_name == "list_documents"
    assert tool_args == {}
    assert captured["call_site"] == "mcp_decision"
    assert "what docs exist?" in captured["prompt"]
    assert captured["temperature"] is None
    assert captured["fn_result"] == (
        '{"use_mcp": true, "tool": "list_documents", "args": {}}', 12, 5,
    )


def test_init_starts_a_task_in_ambient_context(monkeypatch):
    monkeypatch.setattr(assistant_module, "MCPClient", _FakeMCPClient)

    instance = CompanyKBAssistant()

    assert instance.task_id == context.current_task_id.get()
    assert context.current_turn_number.get() == 0


def test_query_advances_turn_number_once_per_call(monkeypatch):
    monkeypatch.setattr(assistant_module, "MCPClient", _FakeMCPClient)
    instance = CompanyKBAssistant()

    monkeypatch.setattr(assistant_module, "retrieve", lambda q: [])
    monkeypatch.setattr(assistant_module, "build_prompt", lambda q, c: "prompt")
    monkeypatch.setattr(assistant_module, "ask_llm", lambda prompt: "answer")
    monkeypatch.setattr(instance, "_llm_decide_mcp_usage", lambda q, c: (None, None))

    instance.query("first question")
    assert context.current_turn_number.get() == 1

    instance.query("second question")
    assert context.current_turn_number.get() == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_assistant.py -v`
Expected: FAIL — `ImportError`/`AttributeError` (`assistant_module.record_llm_call` doesn't exist; `instance.task_id` doesn't exist)

- [ ] **Step 3: Wire assistant.py**

```python
# src/assistant.py — add imports near the top, alongside the existing ones
from telemetry import context
from telemetry.llm_middleware import record_llm_call
```

```python
# src/assistant.py — __init__
    def __init__(self):
        self.llm_client = ollama.Client()
        self.mcp = None
        self.task_id = context.start_task()
        self._init_mcp()
```

```python
# src/assistant.py — _llm_decide_mcp_usage: replace the self.llm_client.chat(...)
# block through response_text extraction with:
        def _do_chat():
            response = self.llm_client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that decides when to use tools. Always respond with valid JSON only."},
                    {"role": "user", "content": decision_prompt}
                ]
            )
            return (
                response["message"]["content"],
                response.get("prompt_eval_count", 0),
                response.get("eval_count", 0),
            )

        try:
            response_text = record_llm_call(
                call_site="mcp_decision",
                model=OLLAMA_MODEL,
                prompt=decision_prompt,
                temperature=None,
                fn=_do_chat,
            )
            response_text = response_text.strip()

            # Clean up JSON if wrapped in markdown
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            decision = json.loads(response_text)

            if decision.get("use_mcp", False):
                tool_name = decision.get("tool")
                tool_args = decision.get("args", {})
                return tool_name, tool_args

            return None, None

        except Exception as e:
            # If LLM decision fails, don't use MCP
            return None, None
```

```python
# src/assistant.py — query(): first line of the method body
    def query(self, user_query: str, verbose=False):
        """Answer a question using RAG and optionally MCP tools."""
        context.next_turn()
        # Step 1: Retrieve from RAG
        contexts = retrieve(user_query)
        ...  # rest of the method unchanged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_assistant.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/assistant.py tests/test_assistant.py
git commit -m "feat: wire assistant.py through telemetry context and mcp_decision middleware"
```

---

### Task 9: `telemetry.tool_middleware` + wire `MCPClient.call_tool()`

**Files:**
- Create: `src/telemetry/tool_middleware.py`
- Modify: `src/mcp/client.py`
- Test: `tests/telemetry/test_tool_middleware.py`
- Test: `tests/mcp/test_client.py` (new)

**Interfaces:**
- Consumes: `context.current_agent_id/current_task_id/current_turn_number` (Task 2), `storage.insert_tool_call` (Task 3).
- Produces: `record_tool_call(tool_name, arguments, fn, db_path=None) -> Any` (returns `fn()`'s return value unchanged), where `fn` is a zero-arg callable returning the raw JSON-RPC response dict (with a `"result"` key). `MCPClient.call_tool(name, arguments)` return type unchanged.

- [ ] **Step 1: Write the failing tests for tool_middleware**

```python
# tests/telemetry/test_tool_middleware.py
import sqlite3

from telemetry import context
from telemetry.tool_middleware import record_tool_call


def test_records_a_tool_calls_row_and_returns_fn_result(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t1")
    context.next_turn()

    response = record_tool_call(
        "list_documents", {}, lambda: {"result": "doc1\ndoc2"}, db_path=db_path,
    )

    assert response == {"result": "doc1\ndoc2"}

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT task_id, turn_number, tool_name, output_tokens FROM tool_calls"
    ).fetchone()
    conn.close()
    assert row == ("t1", 1, "list_documents", len("doc1\ndoc2") // 4)


def test_input_size_is_byte_length_of_serialized_arguments(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t2")
    context.next_turn()

    record_tool_call(
        "read_document", {"file_path": "docs/x.md"}, lambda: {"result": "y"}, db_path=db_path,
    )

    import json
    expected_size = len(json.dumps({"file_path": "docs/x.md"}).encode("utf-8"))

    conn = sqlite3.connect(str(db_path))
    input_size = conn.execute("SELECT input_size FROM tool_calls").fetchone()[0]
    conn.close()
    assert input_size == expected_size


def test_output_size_is_byte_length_of_the_result_field_only(tmp_path):
    db_path = tmp_path / "telemetry.db"
    context.start_task("t3")
    context.next_turn()

    record_tool_call(
        "search_documents", {"query": "vacation"},
        lambda: {"jsonrpc": "2.0", "id": 1, "result": "found: vacation-policy.md"},
        db_path=db_path,
    )

    conn = sqlite3.connect(str(db_path))
    output_size = conn.execute("SELECT output_size FROM tool_calls").fetchone()[0]
    conn.close()
    assert output_size == len("found: vacation-policy.md".encode("utf-8"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/telemetry/test_tool_middleware.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telemetry.tool_middleware'`

- [ ] **Step 3: Implement tool_middleware.py**

```python
# src/telemetry/tool_middleware.py
import json
import time

from telemetry import context, storage


def record_tool_call(tool_name, arguments, fn, db_path=None):
    """Run fn() (a zero-arg callable invoking the actual MCP tool call and
    returning its raw JSON-RPC response dict) and record one row in
    tool_calls. output_size/output_tokens are sourced from the response's
    "result" field only, not the full JSON-RPC envelope. Returns fn()'s
    return value unchanged."""
    input_size = len(json.dumps(arguments).encode("utf-8"))

    start = time.perf_counter()
    response = fn()
    duration_ms = (time.perf_counter() - start) * 1000

    result = response.get("result", "") if isinstance(response, dict) else ""
    result_text = result if isinstance(result, str) else json.dumps(result)
    output_size = len(result_text.encode("utf-8"))
    output_tokens = len(result_text) // 4

    storage.insert_tool_call(
        agent_id=context.current_agent_id.get(),
        task_id=context.current_task_id.get(),
        turn_number=context.current_turn_number.get(),
        tool_name=tool_name,
        input_size=input_size,
        output_size=output_size,
        output_tokens=output_tokens,
        duration_ms=duration_ms,
        db_path=db_path,
    )

    return response
```

- [ ] **Step 4: Run tool_middleware tests to verify they pass**

Run: `pytest tests/telemetry/test_tool_middleware.py -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for MCPClient wiring**

```python
# tests/mcp/test_client.py
import threading

from mcp.client import MCPClient
from telemetry import context, storage


def _make_bare_client(send_fn):
    instance = MCPClient.__new__(MCPClient)
    instance.lock = threading.Lock()
    instance.next_id = 1
    instance._send = send_fn
    return instance


def test_call_tool_records_tool_telemetry():
    context.start_task("task-abc")
    context.next_turn()

    def fake_send(payload):
        assert payload["method"] == "tools/call"
        assert payload["params"]["name"] == "list_documents"
        return {"jsonrpc": "2.0", "id": payload["id"], "result": "doc1\ndoc2\ndoc3"}

    client = _make_bare_client(fake_send)

    response = client.call_tool("list_documents", {})

    assert response["result"] == "doc1\ndoc2\ndoc3"

    conn = storage.get_connection()
    try:
        row = conn.execute(
            "SELECT tool_name, task_id, output_tokens FROM tool_calls"
        ).fetchone()
    finally:
        conn.close()

    assert row == ("list_documents", "task-abc", len("doc1\ndoc2\ndoc3") // 4)


def test_call_tool_increments_next_id():
    def fake_send(payload):
        return {"jsonrpc": "2.0", "id": payload["id"], "result": ""}

    client = _make_bare_client(fake_send)
    assert client.next_id == 1

    client.call_tool("list_documents", {})

    assert client.next_id == 2
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/mcp/test_client.py -v`
Expected: FAIL — `test_call_tool_records_tool_telemetry` fails because no `tool_calls` row exists (`call_tool` doesn't record telemetry yet)

- [ ] **Step 7: Wire client.py**

```python
# src/mcp/client.py — add near the top, alongside the existing imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from telemetry.tool_middleware import record_tool_call
```

```python
# src/mcp/client.py — replace the existing call_tool() method
    def call_tool(self, name, arguments):
        """Call an MCP tool, recording telemetry for the call."""
        def _do_call():
            payload = {
                "jsonrpc": "2.0",
                "id": self.next_id,
                "method": "tools/call",
                "params": {
                    "name": name,
                    "arguments": arguments
                }
            }
            self.next_id += 1
            return self._send(payload)

        return record_tool_call(name, arguments, _do_call)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/mcp/test_client.py tests/telemetry/test_tool_middleware.py -v`
Expected: PASS

- [ ] **Step 9: Run the full test suite to check for regressions**

Run: `pytest -q`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add src/telemetry/tool_middleware.py src/mcp/client.py tests/telemetry/test_tool_middleware.py tests/mcp/test_client.py
git commit -m "feat: add telemetry.tool_middleware and wire MCPClient.call_tool"
```

---

### Task 10: Clear `llm_cache` on every `build_index()` run

**Files:**
- Modify: `src/rag/build_index.py`
- Test: `tests/rag/test_build_index.py` (extend)

**Interfaces:**
- Consumes: `telemetry.llm_cache.clear` (Task 5).

- [ ] **Step 1: Write the failing test**

```python
# tests/rag/test_build_index.py — append to the existing file
import numpy as np

from telemetry import llm_cache


def test_build_index_clears_llm_cache(tmp_path, monkeypatch):
    fake_docs = [{"path": "doc.txt", "text": "irrelevant"}]
    fake_chunks = [{"id": 0, "text": "invoice", "source": "doc.txt", "chunk_id": 0}]
    fake_embeddings = np.array([[1.0, 0.0]], dtype="float32")

    monkeypatch.setattr(build_index_module, "ingest_documents", lambda: fake_docs)
    monkeypatch.setattr(build_index_module, "chunk_documents", lambda docs: fake_chunks)
    monkeypatch.setattr(build_index_module, "embed_chunks", lambda chunks: fake_embeddings)
    monkeypatch.setattr(build_index_module, "FAISS_INDEX_PATH", str(tmp_path / "index.faiss"))
    monkeypatch.setattr(build_index_module, "CHUNKS_PATH", str(tmp_path / "chunks.pkl"))
    monkeypatch.setattr("config.FTS_DB_PATH", str(tmp_path / "fts.db"))

    llm_cache.put("some prompt", "qwen3:0.6b", None, "cached answer", 10, 5)
    assert llm_cache.get("some prompt", "qwen3:0.6b", None) is not None

    build_index_module.build_index()

    assert llm_cache.get("some prompt", "qwen3:0.6b", None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_build_index.py::test_build_index_clears_llm_cache -v`
Expected: FAIL — cache entry still present after `build_index()`

- [ ] **Step 3: Wire build_index.py**

```python
# src/rag/build_index.py — first line inside build_index(), before any printing
def build_index():
    """Build FAISS index from documents."""
    from telemetry import llm_cache
    llm_cache.clear()  # a rebuilt index can change which chunks are
                        # retrieved, invalidating any cached answer for the
                        # same normalized question text

    # Resolve paths relative to src directory
    src_dir = Path(__file__).parent.parent
    ...  # rest of the function unchanged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_build_index.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/rag/build_index.py tests/rag/test_build_index.py
git commit -m "feat: clear llm_cache on every build_index run"
```

---

### Task 11: `telemetry.dashboard` — aggregate + comparison view

**Files:**
- Create: `src/telemetry/dashboard.py`
- Test: `tests/telemetry/test_dashboard.py`

**Interfaces:**
- Consumes: `storage.get_connection`, `storage.insert_llm_call`, `storage.insert_tool_call` (Task 3).
- Produces: `compute_aggregate(db_path, task_ids=None) -> dict`, `compute_comparison(db_path, task_id_a, task_id_b) -> list[dict]`, `render_dashboard(aggregate) -> rich renderable`, `render_comparison(rows, task_id_a, task_id_b) -> rich.table.Table`. Task 13's CLI wiring calls these 4 functions by name.

- [ ] **Step 1: Write the failing tests**

```python
# tests/telemetry/test_dashboard.py
import pytest
from rich.console import Console

from telemetry import storage
from telemetry.dashboard import (
    compute_aggregate,
    compute_comparison,
    render_dashboard,
    render_comparison,
)


def _seed(db_path, task_id, turn_number, input_tokens, output_tokens, cached_tokens, cost, call_site="ask_llm"):
    storage.insert_llm_call(
        agent_id="a", task_id=task_id, turn_number=turn_number, model="qwen3:0.6b",
        input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=cached_tokens,
        reasoning_tokens=0, latency_ms=10.0, estimated_cost=cost, call_site=call_site,
        db_path=db_path,
    )


def test_compute_aggregate_on_empty_db_returns_zeros(tmp_path):
    db_path = tmp_path / "telemetry.db"
    storage.get_connection(db_path).close()

    result = compute_aggregate(db_path)

    assert result["tasks_completed"] == 0
    assert result["total_input_tokens"] == 0
    assert result["avg_tokens_per_task"] == 0.0
    assert result["cache_hit_rate"] == 0.0
    assert result["tool_breakdown"] == []


def test_compute_aggregate_single_task(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "t1", 1, 100, 50, 0, 0.03)
    _seed(db_path, "t1", 1, 20, 10, 20, 0.0, call_site="query_expansion")  # cache hit

    result = compute_aggregate(db_path)

    assert result["tasks_completed"] == 1
    assert result["total_input_tokens"] == 120
    assert result["total_output_tokens"] == 60
    assert result["total_cached_tokens"] == 20
    assert result["estimated_cost"] == 0.03
    assert result["avg_tokens_per_task"] == 180.0
    assert result["avg_turns_per_task"] == 1.0
    assert result["cache_hit_rate"] == 0.5  # 1 of 2 llm_calls rows was a hit


def test_compute_aggregate_multi_task_scoped_by_task_ids(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "t1", 1, 100, 50, 0, 0.03)
    _seed(db_path, "t2", 1, 200, 100, 0, 0.06)
    _seed(db_path, "t3", 1, 999, 999, 0, 9.99)  # excluded by task_ids filter

    result = compute_aggregate(db_path, task_ids=["t1", "t2"])

    assert result["tasks_completed"] == 2
    assert result["total_input_tokens"] == 300
    assert result["estimated_cost"] == 0.09


def test_compute_aggregate_tool_breakdown_sorted_by_total_duration_desc(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "t1", 1, 10, 10, 0, 0.0)
    storage.insert_tool_call(
        agent_id="a", task_id="t1", turn_number=1, tool_name="read_document",
        input_size=10, output_size=100, output_tokens=25, duration_ms=50.0, db_path=db_path,
    )
    storage.insert_tool_call(
        agent_id="a", task_id="t1", turn_number=1, tool_name="list_documents",
        input_size=2, output_size=20, output_tokens=5, duration_ms=200.0, db_path=db_path,
    )

    result = compute_aggregate(db_path)

    assert [row["tool_name"] for row in result["tool_breakdown"]] == ["list_documents", "read_document"]
    assert result["avg_tool_calls_per_task"] == 2.0


def test_compute_comparison_delta_increase_decrease_and_unchanged(tmp_path):
    db_path = tmp_path / "telemetry.db"
    _seed(db_path, "t1", 1, 100, 100, 0, 0.10)
    _seed(db_path, "t2", 1, 150, 100, 0, 0.05)

    rows = compute_comparison(db_path, "t1", "t2")
    by_metric = {row["metric"]: row for row in rows}

    assert by_metric["Input tokens"]["delta"] == 50                          # increase
    assert by_metric["Estimated cost"]["delta"] == pytest.approx(-0.05)      # decrease
    assert by_metric["Output tokens"]["delta"] == 0                         # unchanged


def test_render_dashboard_includes_key_metrics_as_text():
    aggregate = {
        "tasks_completed": 3, "total_input_tokens": 100, "total_output_tokens": 50,
        "total_cached_tokens": 10, "estimated_cost": 0.05, "avg_tokens_per_task": 50.0,
        "avg_turns_per_task": 2.0, "avg_tool_calls_per_task": 1.0, "cache_hit_rate": 0.25,
        "tool_breakdown": [],
    }

    console = Console(record=True, width=120)
    console.print(render_dashboard(aggregate))
    text = console.export_text()

    assert "Tasks completed" in text
    assert "3" in text
    assert "25.0%" in text


def test_render_comparison_includes_session_ids_and_delta_column():
    rows = [{"metric": "Input tokens", "session_a": 100, "session_b": 150, "delta": 50}]

    console = Console(record=True, width=120)
    console.print(render_comparison(rows, "t1", "t2"))
    text = console.export_text()

    assert "t1" in text
    assert "t2" in text
    assert "Input tokens" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/telemetry/test_dashboard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telemetry.dashboard'`

- [ ] **Step 3: Implement dashboard.py**

```python
# src/telemetry/dashboard.py
from rich.console import Group
from rich.table import Table

from telemetry import storage


def _task_filter_clause(task_ids):
    if not task_ids:
        return "", []
    placeholders = ",".join("?" for _ in task_ids)
    return f"WHERE task_id IN ({placeholders})", list(task_ids)


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def compute_aggregate(db_path, task_ids=None):
    """Aggregate stats across all history, or scoped to `task_ids` if given."""
    where_clause, params = _task_filter_clause(task_ids)

    conn = storage.get_connection(db_path)
    try:
        llm_row = conn.execute(
            f"""
            SELECT
                COUNT(DISTINCT task_id),
                COALESCE(SUM(input_tokens), 0),
                COALESCE(SUM(output_tokens), 0),
                COALESCE(SUM(cached_tokens), 0),
                COALESCE(SUM(estimated_cost), 0.0),
                COUNT(*),
                COALESCE(SUM(CASE WHEN cached_tokens > 0 THEN 1 ELSE 0 END), 0)
            FROM llm_calls {where_clause}
            """,
            params,
        ).fetchone()

        turns_row = conn.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT DISTINCT task_id, turn_number FROM llm_calls {where_clause}
            )
            """,
            params,
        ).fetchone()

        total_tool_calls = conn.execute(
            f"SELECT COUNT(*) FROM tool_calls {where_clause}", params
        ).fetchone()[0]

        tool_breakdown = conn.execute(
            f"""
            SELECT tool_name, COUNT(*), SUM(duration_ms), AVG(duration_ms)
            FROM tool_calls {where_clause}
            GROUP BY tool_name
            ORDER BY SUM(duration_ms) DESC
            """,
            params,
        ).fetchall()
    finally:
        conn.close()

    (
        tasks_completed, total_input_tokens, total_output_tokens,
        total_cached_tokens, total_estimated_cost, total_llm_calls, cache_hits,
    ) = llm_row
    total_turns = turns_row[0] or 0

    return {
        "tasks_completed": tasks_completed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cached_tokens": total_cached_tokens,
        "estimated_cost": total_estimated_cost,
        "avg_tokens_per_task": _safe_div(total_input_tokens + total_output_tokens, tasks_completed),
        "avg_turns_per_task": _safe_div(total_turns, tasks_completed),
        "avg_tool_calls_per_task": _safe_div(total_tool_calls, tasks_completed),
        "cache_hit_rate": _safe_div(cache_hits, total_llm_calls),
        "tool_breakdown": [
            {
                "tool_name": row[0], "call_count": row[1],
                "total_duration_ms": row[2], "avg_duration_ms": row[3],
            }
            for row in tool_breakdown
        ],
    }


def compute_comparison(db_path, task_id_a, task_id_b):
    """Exactly-2-task delta comparison: [{"metric", "session_a", "session_b", "delta"}, ...]."""
    agg_a = compute_aggregate(db_path, task_ids=[task_id_a])
    agg_b = compute_aggregate(db_path, task_ids=[task_id_b])

    metrics = [
        ("Input tokens", "total_input_tokens"),
        ("Output tokens", "total_output_tokens"),
        ("Cached tokens", "total_cached_tokens"),
        ("Estimated cost", "estimated_cost"),
        ("Turns", "avg_turns_per_task"),
        ("Tool calls", "avg_tool_calls_per_task"),
        ("Cache hit rate", "cache_hit_rate"),
    ]

    return [
        {
            "metric": label,
            "session_a": agg_a[key],
            "session_b": agg_b[key],
            "delta": agg_b[key] - agg_a[key],
        }
        for label, key in metrics
    ]


def render_dashboard(aggregate):
    table = Table(title="AI Agent — Aggregate Stats")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Tasks completed", str(aggregate["tasks_completed"]))
    table.add_row("Total input tokens", str(aggregate["total_input_tokens"]))
    table.add_row("Total output tokens", str(aggregate["total_output_tokens"]))
    table.add_row("Total cached tokens", str(aggregate["total_cached_tokens"]))
    table.add_row("Estimated cost", f"${aggregate['estimated_cost']:.4f}")
    table.add_row("Avg tokens/task", f"{aggregate['avg_tokens_per_task']:.1f}")
    table.add_row("Avg turns/task", f"{aggregate['avg_turns_per_task']:.1f}")
    table.add_row("Avg tool calls/task", f"{aggregate['avg_tool_calls_per_task']:.1f}")
    table.add_row("Cache hit rate", f"{aggregate['cache_hit_rate'] * 100:.1f}%")

    if not aggregate["tool_breakdown"]:
        return table

    breakdown_table = Table(title="Tool usage breakdown")
    breakdown_table.add_column("Tool")
    breakdown_table.add_column("Calls")
    breakdown_table.add_column("Total duration (ms)")
    breakdown_table.add_column("Avg duration (ms)")
    for row in aggregate["tool_breakdown"]:
        breakdown_table.add_row(
            row["tool_name"], str(row["call_count"]),
            f"{row['total_duration_ms']:.1f}", f"{row['avg_duration_ms']:.1f}",
        )

    return Group(table, breakdown_table)


def _format_metric_value(metric, value):
    if metric == "Estimated cost":
        return f"${value:.4f}"
    if metric == "Cache hit rate":
        return f"{value * 100:.1f}%"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def render_comparison(rows, task_id_a, task_id_b):
    table = Table(title="Session comparison")
    table.add_column("Metric")
    table.add_column(f"Session {task_id_a}")
    table.add_column(f"Session {task_id_b}")
    table.add_column("Δ")

    for row in rows:
        delta = row["delta"]
        sign = "+" if delta > 0 else ""
        table.add_row(
            row["metric"],
            _format_metric_value(row["metric"], row["session_a"]),
            _format_metric_value(row["metric"], row["session_b"]),
            f"{sign}{_format_metric_value(row['metric'], delta)}",
        )

    return table
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/telemetry/test_dashboard.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/dashboard.py tests/telemetry/test_dashboard.py
git commit -m "feat: add telemetry.dashboard aggregate and comparison view"
```

---

### Task 12: `telemetry.timeline` — per-task turn-by-turn view

**Files:**
- Create: `src/telemetry/timeline.py`
- Test: `tests/telemetry/test_timeline.py`

**Interfaces:**
- Consumes: `storage.get_connection` (Task 3).
- Produces: `fetch_timeline(db_path, task_id) -> list[dict]` (each dict has `turn_number`, `timestamp`, `type` (`"llm"`/`"tool"`), `label`, plus type-specific metric keys), `render_timeline(task_id, events) -> rich.table.Table`. Task 13's CLI wiring calls these 2 functions by name.

- [ ] **Step 1: Write the failing tests**

```python
# tests/telemetry/test_timeline.py
from rich.console import Console

from telemetry import storage
from telemetry.timeline import fetch_timeline, render_timeline


def test_fetch_timeline_orders_by_turn_then_timestamp(tmp_path):
    db_path = tmp_path / "telemetry.db"
    storage.insert_llm_call(
        agent_id="a", task_id="t1", turn_number=2, model="m", input_tokens=1,
        output_tokens=1, cached_tokens=0, reasoning_tokens=0, latency_ms=1.0,
        estimated_cost=0.0, call_site="ask_llm", timestamp="2026-09-05T10:00:02",
        db_path=db_path,
    )
    storage.insert_llm_call(
        agent_id="a", task_id="t1", turn_number=1, model="m", input_tokens=2,
        output_tokens=2, cached_tokens=0, reasoning_tokens=0, latency_ms=1.0,
        estimated_cost=0.0, call_site="mcp_decision", timestamp="2026-09-05T10:00:00",
        db_path=db_path,
    )
    storage.insert_tool_call(
        agent_id="a", task_id="t1", turn_number=1, tool_name="list_documents",
        input_size=2, output_size=10, output_tokens=3, duration_ms=5.0,
        timestamp="2026-09-05T10:00:01", db_path=db_path,
    )

    events = fetch_timeline(db_path, "t1")

    assert [e["turn_number"] for e in events] == [1, 1, 2]
    assert [e["type"] for e in events] == ["llm", "tool", "llm"]
    assert events[0]["label"] == "mcp_decision"
    assert events[1]["label"] == "list_documents"


def test_fetch_timeline_only_returns_the_requested_task(tmp_path):
    db_path = tmp_path / "telemetry.db"
    storage.insert_llm_call(
        agent_id="a", task_id="t1", turn_number=1, model="m", input_tokens=1,
        output_tokens=1, cached_tokens=0, reasoning_tokens=0, latency_ms=1.0,
        estimated_cost=0.0, call_site="ask_llm", db_path=db_path,
    )
    storage.insert_llm_call(
        agent_id="a", task_id="t2", turn_number=1, model="m", input_tokens=1,
        output_tokens=1, cached_tokens=0, reasoning_tokens=0, latency_ms=1.0,
        estimated_cost=0.0, call_site="ask_llm", db_path=db_path,
    )

    events = fetch_timeline(db_path, "t1")

    assert len(events) == 1


def test_render_timeline_lists_turn_and_type_for_each_event():
    events = [
        {
            "turn_number": 1, "timestamp": "t", "type": "llm", "label": "ask_llm",
            "input_tokens": 10, "output_tokens": 5, "cached_tokens": 0,
            "latency_ms": 12.3, "estimated_cost": 0.0001,
        },
        {
            "turn_number": 1, "timestamp": "t", "type": "tool", "label": "list_documents",
            "input_size": 2, "output_size": 20, "output_tokens": 5, "duration_ms": 8.0,
        },
    ]

    console = Console(record=True, width=120)
    console.print(render_timeline("t1", events))
    text = console.export_text()

    assert "Turn 1" in text
    assert "ask_llm" in text
    assert "list_documents" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/telemetry/test_timeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telemetry.timeline'`

- [ ] **Step 3: Implement timeline.py**

```python
# src/telemetry/timeline.py
from rich.table import Table

from telemetry import storage


def fetch_timeline(db_path, task_id):
    """Every llm_calls/tool_calls row for `task_id`, ordered by
    (turn_number, timestamp)."""
    conn = storage.get_connection(db_path)
    try:
        llm_rows = conn.execute(
            """
            SELECT turn_number, timestamp, call_site, input_tokens,
                   output_tokens, cached_tokens, latency_ms, estimated_cost
            FROM llm_calls WHERE task_id = ?
            """,
            (task_id,),
        ).fetchall()

        tool_rows = conn.execute(
            """
            SELECT turn_number, timestamp, tool_name, input_size,
                   output_size, output_tokens, duration_ms
            FROM tool_calls WHERE task_id = ?
            """,
            (task_id,),
        ).fetchall()
    finally:
        conn.close()

    events = [
        {
            "turn_number": row[0], "timestamp": row[1], "type": "llm", "label": row[2],
            "input_tokens": row[3], "output_tokens": row[4], "cached_tokens": row[5],
            "latency_ms": row[6], "estimated_cost": row[7],
        }
        for row in llm_rows
    ] + [
        {
            "turn_number": row[0], "timestamp": row[1], "type": "tool", "label": row[2],
            "input_size": row[3], "output_size": row[4], "output_tokens": row[5],
            "duration_ms": row[6],
        }
        for row in tool_rows
    ]

    events.sort(key=lambda e: (e["turn_number"], e["timestamp"]))
    return events


def render_timeline(task_id, events):
    table = Table(title=f"Timeline — task {task_id}")
    table.add_column("Turn")
    table.add_column("Type")
    table.add_column("Detail")
    table.add_column("Tokens/Size")
    table.add_column("Latency (ms)")
    table.add_column("Cost")

    for event in events:
        if event["type"] == "llm":
            tokens = f"{event['input_tokens']}in/{event['output_tokens']}out"
            if event["cached_tokens"] > 0:
                tokens += " (cached)"
            table.add_row(
                f"Turn {event['turn_number']}", "LLM", event["label"], tokens,
                f"{event['latency_ms']:.1f}", f"${event['estimated_cost']:.6f}",
            )
        else:
            tokens = f"{event['output_tokens']} tok / {event['output_size']}B"
            table.add_row(
                f"Turn {event['turn_number']}", "Tool", event["label"], tokens,
                f"{event['duration_ms']:.1f}", "-",
            )

    return table
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/telemetry/test_timeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/timeline.py tests/telemetry/test_timeline.py
git commit -m "feat: add telemetry.timeline per-task turn-by-turn view"
```

---

### Task 13: CLI wiring — `main.py` `dashboard`/`timeline` subcommands

**Files:**
- Modify: `src/main.py`
- Test: `tests/test_main.py` (new)

**Interfaces:**
- Consumes: `telemetry.dashboard.compute_aggregate/compute_comparison/render_dashboard/render_comparison` (Task 11), `telemetry.timeline.fetch_timeline/render_timeline` (Task 12), `assistant.CompanyKBAssistant.task_id` (Task 8).
- Produces: `build_parser() -> argparse.ArgumentParser` — the pure, testable seam; `main()` stays a thin dispatcher on top of it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_main.py
from main import build_parser


def test_no_args_means_interactive_mode():
    args = build_parser().parse_args([])
    assert args.command is None


def test_build_index_subcommand():
    args = build_parser().parse_args(["build-index"])
    assert args.command == "build-index"


def test_dashboard_subcommand_with_no_flags():
    args = build_parser().parse_args(["dashboard"])
    assert args.command == "dashboard"
    assert args.tasks is None
    assert args.compare is None


def test_dashboard_subcommand_with_tasks_flag():
    args = build_parser().parse_args(["dashboard", "--tasks", "184,185,190"])
    assert args.tasks == "184,185,190"


def test_dashboard_subcommand_with_compare_flag():
    args = build_parser().parse_args(["dashboard", "--compare", "184", "185"])
    assert args.compare == ["184", "185"]


def test_timeline_subcommand_requires_task_id():
    args = build_parser().parse_args(["timeline", "184"])
    assert args.command == "timeline"
    assert args.task_id == "184"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_main.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_parser' from 'main'`

- [ ] **Step 3: Refactor main.py to argparse + add dashboard/timeline dispatch**

```python
#!/usr/bin/env python3
"""
Company Knowledge Base Assistant - Main Entry Point
"""

import argparse
import sys

from assistant import CompanyKBAssistant


def build_parser():
    parser = argparse.ArgumentParser(prog="main.py")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("build-index")

    dashboard_parser = subparsers.add_parser("dashboard")
    dashboard_parser.add_argument(
        "--tasks", help="Comma-separated task_ids to scope the aggregate to",
    )
    dashboard_parser.add_argument(
        "--compare", nargs="+", metavar="TASK_ID",
        help="Exactly 2 task_ids to compare",
    )

    timeline_parser = subparsers.add_parser("timeline")
    timeline_parser.add_argument("task_id")

    return parser


def _run_dashboard(args):
    from telemetry.dashboard import (
        compute_aggregate, compute_comparison, render_dashboard, render_comparison,
    )
    from rich.console import Console
    console = Console()

    if args.compare:
        if len(args.compare) != 2:
            print("❌ --compare requires exactly 2 task_ids (use --tasks for more)")
            sys.exit(1)
        rows = compute_comparison(None, args.compare[0], args.compare[1])
        console.print(render_comparison(rows, args.compare[0], args.compare[1]))
        return

    task_ids = args.tasks.split(",") if args.tasks else None
    aggregate = compute_aggregate(None, task_ids=task_ids)
    console.print(render_dashboard(aggregate))


def _run_timeline(task_id):
    from telemetry.timeline import fetch_timeline, render_timeline
    from rich.console import Console
    events = fetch_timeline(None, task_id)
    Console().print(render_timeline(task_id, events))


def _run_interactive():
    assistant = CompanyKBAssistant()

    print("=" * 60)
    print("🤖 Company Knowledge Base Assistant")
    print("=" * 60)
    print(f"\n🆔 Task ID: {assistant.task_id}")
    print("   (pass this to `python main.py timeline <task_id>` afterward)")
    print("\nAsk questions about company policies, procedures, and documentation.")
    print("Type 'exit' or 'quit' to stop\n")

    try:
        while True:
            query = input("❓ Question: ").strip()

            if not query:
                continue

            if query.lower() in {"exit", "quit", "q"}:
                print("\n👋 Goodbye!")
                break

            print("\n" + "─" * 60)
            print("🤖 Answer:\n")

            try:
                result = assistant.query(query, verbose=True)
                print(result["answer"])

                if result["sources"]:
                    print("\n📚 Sources:")
                    for src in result["sources"]:
                        print(f"  • {src}")

                if result["mcp_used"]:
                    print(f"\n🔧 Used MCP tool: {result['mcp_tool']}")

            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

            print("─" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    finally:
        assistant.close()


def main():
    """Main entry point for the assistant."""
    args = build_parser().parse_args()

    if args.command == "build-index":
        from rag.build_index import build_index
        build_index()
        return

    if args.command == "dashboard":
        _run_dashboard(args)
        return

    if args.command == "timeline":
        _run_timeline(args.task_id)
        return

    _run_interactive()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_main.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `pytest -q`
Expected: PASS — full suite green

- [ ] **Step 6: Manual smoke test of the new CLI surface**

Run: `cd src && python main.py dashboard` (from repo root: `cd src && python main.py dashboard`)
Expected: prints an "AI Agent — Aggregate Stats" table (all zeros if `telemetry.db` doesn't exist yet — `get_connection` creates it empty)

Run: `cd src && python main.py dashboard --compare 1 2 3`
Expected: prints `❌ --compare requires exactly 2 task_ids (use --tasks for more)` and exits non-zero

- [ ] **Step 7: Commit**

```bash
git add src/main.py tests/test_main.py
git commit -m "feat: add dashboard/timeline CLI subcommands to main.py"
```

---

## Self-Review Notes (for the plan author, not a task)

- **Spec coverage:** all 3 LLM call sites (Tasks 7, 8) and the 1 tool call site (Task 9) are wrapped; cache + invalidation (Tasks 5, 10); pricing table + unknown-model fallback (Task 4); storage schema exactly matching spec (Task 3); `dashboard`/`timeline`/`--tasks`/`--compare` CLI surface (Task 13); context propagation incl. nested-call inheritance (Task 2, tested). Out-of-scope items (retention, concurrency, charts) are intentionally not tasked.
- **Type consistency check:** `record_llm_call(call_site, model, prompt, temperature, fn, db_path=None)` signature is identical across Tasks 6, 7, 8. `record_tool_call(tool_name, arguments, fn, db_path=None)` identical across Tasks 9. `storage.insert_llm_call`/`insert_tool_call` keyword names identical across Tasks 3, 6, 9, 11, 12.
