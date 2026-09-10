# Hybrid Search (FTS + Query Expansion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `src/rag/query.py`'s pure-vector `retrieve()` with a hybrid pipeline — LLM query expansion, parallel FAISS vector search + SQLite FTS5 full-text search, and Reciprocal Rank Fusion (RRF) — built test-first, one component at a time.

**Architecture:** Four new pure/near-pure modules (`fusion.py`, `fts_index.py`, `expand.py`, and changes to `chunk.py`) are unit-tested in isolation with fakes/injected dependencies. They're then wired together in `query.py`'s `retrieve()` and `build_index.py`, which get integration-style tests using monkeypatched globals so no real Ollama server, FAISS index, or SentenceTransformer download is required to run the suite (that import-time cost is unavoidable and paid once per test session — see Global Constraints).

**Tech Stack:** Python 3.10+, `sqlite3` (stdlib, FTS5), `faiss-cpu`, `concurrent.futures.ThreadPoolExecutor` (stdlib), `pytest>=7.4.0` (new dev dependency).

**Spec:** `spec/v1/SPEC.md` (this plan implements it section-by-section; read it first — this plan does not repeat its rationale, only its exact interfaces and the tests that pin them down).

## Global Constraints

- Baseline commit: `a6583d4e25f6f94eef96f1829ac4f72b9981c021`. No migration code: `src/index.faiss` and `src/chunks.pkl` are deleted and rebuilt via `python main.py build-index` once this plan is fully implemented (spec §4) — do not write backfill/compat code for the old chunk shape.
- New config constants (`src/config.py`, spec §6), values exact: `FTS_DB_PATH = "fts.db"`, `RRF_K = 60`, `NUM_QUERY_EXPANSIONS = 3`, `CANDIDATE_K = TOP_K * 3`. `TOP_K` (existing, `= 5`) becomes the *post-fusion* result count, not a raw search top-k.
- RRF (`rrf_fuse()`) is the single fusion implementation, reused for both intra-branch (per query variant) and inter-branch (vector vs FTS) fusion — do not write a second fusion function.
- Only `generate_keywords()`'s failure path must degrade gracefully (fallback to `[query]`, never raise). Do NOT add blanket try/except around FAISS or `ThreadPoolExecutor` calls elsewhere — a hard FAISS/SQLite failure is allowed to propagate (spec §8). This is a deliberate, smaller error-handling surface — don't over-defend.
- SQLite connections are not thread-safe: `search_fts()` must open a fresh `sqlite3.connect()` per call, never a shared/module-level connection (spec §7.2). FTS5 MATCH queries are sanitized by whitespace-splitting the query, double-quoting each token, and OR-ing them; a still-malformed MATCH must return `[]`, not raise.
- No new runtime dependency: `sqlite3`/FTS5 is stdlib and confirmed available in this environment (`sqlite3.sqlite_version == 3.53.0`). Add `pytest>=7.4.0` to `src/requirements.txt` for testing only.
- `src/mcp/server.py` and `src/assistant.py` are unchanged — out of scope (spec §1, §12).
- **Known test-environment cost:** importing `rag.query` (directly or transitively) triggers `SentenceTransformer(EMBEDDING_MODEL)` loading and one `_ensure_index_exists()` build attempt at module scope — pre-existing behavior, out of scope to refactor. With no `src/index.faiss`/`src/chunks.pkl` present and no `docs/` at the pytest rootdir, this build attempt fails safely (prints warnings, returns `False`, does not raise) and costs a few seconds once per test session (module caching), not once per test.

---

## Task 1: Test Infrastructure

**Files:**
- Create: `pytest.ini`
- Modify: `src/requirements.txt`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a working `pytest` invocation from the repo root with `src/` on `sys.path`, so every later task's tests can `import config`, `import rag.fusion`, etc. directly. All later tasks assume `pytest <path> -v` is run from the repo root.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
from config import TOP_K


def test_config_module_is_importable_via_pythonpath():
    assert TOP_K == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py -v` (from repo root)
Expected: FAIL — either `pytest`/`ModuleNotFoundError: No module named 'pytest'` if pytest isn't installed yet, or `ModuleNotFoundError: No module named 'config'` if pytest happens to already be globally installed (pythonpath isn't configured yet either way).

- [ ] **Step 3: Add pytest.ini and the pytest dependency**

```ini
# pytest.ini
[pytest]
pythonpath = src
testpaths = tests
```

Append to `src/requirements.txt`:

```
pytest>=7.4.0
```

Install it: `pip install -r src/requirements.txt`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pytest.ini src/requirements.txt tests/test_smoke.py
git commit -m "test: add pytest infra with src/ on pythonpath"
```

---

## Task 2: Config additions

**Files:**
- Modify: `src/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `config.FTS_DB_PATH` (str), `config.RRF_K` (int), `config.NUM_QUERY_EXPANSIONS` (int), `config.CANDIDATE_K` (int) — imported by name in Tasks 5, 6, 8, 9.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import config


def test_hybrid_search_config_constants():
    assert config.FTS_DB_PATH == "fts.db"
    assert config.RRF_K == 60
    assert config.NUM_QUERY_EXPANSIONS == 3
    assert config.CANDIDATE_K == config.TOP_K * 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `AttributeError: module 'config' has no attribute 'FTS_DB_PATH'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/config.py`:

```python
# Hybrid search configuration
FTS_DB_PATH = "fts.db"          # relative to src dir, mirrors FAISS_INDEX_PATH
RRF_K = 60                       # RRF constant k, per assignment's formula
NUM_QUERY_EXPANSIONS = 3         # max alternative phrases the LLM may produce
CANDIDATE_K = TOP_K * 3          # candidate pool size per method, before fusion
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add hybrid search config constants"
```

---

## Task 3: `chunk.py` — globally-unique chunk id

**Files:**
- Modify: `src/rag/chunk.py:25-38`
- Test: `tests/rag/test_chunk.py`

**Interfaces:**
- Consumes: nothing new (still takes `documents: list[dict]` with `path`/`text` keys).
- Produces: every dict returned by `chunk_documents()` now also has an `id` key (int, globally unique, 0-based, equal to the dict's position in the returned list) in addition to the existing `text`/`source`/`chunk_id` keys. Tasks 5, 8, 9 rely on `chunk["id"]` existing.

- [ ] **Step 1: Write the failing test**

```python
# tests/rag/test_chunk.py
from rag.chunk import chunk_documents


def test_chunk_documents_assigns_globally_unique_sequential_ids(monkeypatch):
    monkeypatch.setattr("rag.chunk.CHUNK_SIZE", 5)
    monkeypatch.setattr("rag.chunk.CHUNK_OVERLAP", 1)

    documents = [
        {"path": "doc1.txt", "text": "word " * 20},
        {"path": "doc2.txt", "text": "term " * 20},
    ]
    chunks = chunk_documents(documents)

    ids = [c["id"] for c in chunks]
    assert ids == list(range(len(chunks)))
    assert len(set(ids)) == len(chunks)

    doc1_chunk_ids = [c["chunk_id"] for c in chunks if c["source"] == "doc1.txt"]
    assert doc1_chunk_ids[0] == 0
    assert len(doc1_chunk_ids) > 1  # CHUNK_SIZE=5 must split "word "*20 into >1 chunk
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_chunk.py -v`
Expected: FAIL with `KeyError: 'id'`

- [ ] **Step 3: Write minimal implementation**

Replace `src/rag/chunk.py:25-38`:

```python
def chunk_documents(documents):
    """Chunk all documents into smaller pieces."""
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc["path"],
                "chunk_id": idx
            })

    for global_id, chunk in enumerate(all_chunks):
        chunk["id"] = global_id  # globally unique, used by FAISS + FTS + RRF

    return all_chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_chunk.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/chunk.py tests/rag/test_chunk.py
git commit -m "feat: assign globally-unique id to every chunk"
```

---

## Task 4: `fusion.py` — Reciprocal Rank Fusion

**Files:**
- Create: `src/rag/fusion.py`
- Test: `tests/rag/test_fusion.py`

**Interfaces:**
- Consumes: nothing (pure function, no dependency on other new modules).
- Produces: `rrf_fuse(ranked_lists: list[list[int]], k: int = 60) -> list[int]`. Used by Task 8 (`_hybrid_search_task`, `retrieve`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag/test_fusion.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.fusion'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/rag/fusion.py
def rrf_fuse(ranked_lists, k=60):
    """Merge N ranked lists of ids into one ranking via Reciprocal Rank Fusion.

    score(id) = sum, over every list L containing id, of 1 / (k + rank_L(id))
    where rank_L(id) is the 1-based position of id in L. Ties are broken by
    (1) the id's best rank in any single input list, then (2) the id's
    numeric value, for a fully deterministic order.
    """
    scores = {}
    best_rank = {}

    for ranked_list in ranked_lists:
        seen_in_this_list = set()
        for rank, doc_id in enumerate(ranked_list, start=1):
            if doc_id in seen_in_this_list:
                continue  # only the first (best-ranked) occurrence counts
            seen_in_this_list.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in best_rank or rank < best_rank[doc_id]:
                best_rank[doc_id] = rank

    return sorted(
        scores.keys(),
        key=lambda doc_id: (-scores[doc_id], best_rank[doc_id], doc_id),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_fusion.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rag/fusion.py tests/rag/test_fusion.py
git commit -m "feat: add rrf_fuse for reciprocal rank fusion"
```

---

## Task 5: `fts_index.py` — SQLite FTS5 index

**Files:**
- Create: `src/rag/fts_index.py`
- Test: `tests/rag/test_fts_index.py`

**Interfaces:**
- Consumes: chunk dicts with `id` (int) and `text` (str) keys (Task 3's output shape).
- Produces: `build_fts_index(chunks: list[dict], fts_db_path) -> None` and `search_fts(fts_db_path, query: str, top_k: int) -> list[int]`. Used by Task 8 (`retrieve`) and Task 9 (`build_index`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag/test_fts_index.py
import sqlite3

from rag.fts_index import build_fts_index, search_fts


def test_build_fts_index_creates_queryable_table(tmp_path):
    fts_db_path = tmp_path / "fts.db"
    chunks = [
        {"id": 1, "text": "invoice processing steps"},
        {"id": 2, "text": "vacation policy details"},
    ]

    build_fts_index(chunks, fts_db_path)

    conn = sqlite3.connect(str(fts_db_path))
    rows = conn.execute("SELECT id, text FROM chunks_fts ORDER BY id").fetchall()
    conn.close()
    assert rows == [(1, "invoice processing steps"), (2, "vacation policy details")]


def test_build_fts_index_overwrites_existing_file(tmp_path):
    fts_db_path = tmp_path / "fts.db"
    build_fts_index([{"id": 1, "text": "old content"}], fts_db_path)
    build_fts_index([{"id": 2, "text": "new content"}], fts_db_path)

    conn = sqlite3.connect(str(fts_db_path))
    rows = conn.execute("SELECT id, text FROM chunks_fts").fetchall()
    conn.close()
    assert rows == [(2, "new content")]


def test_search_fts_ranks_by_bm25_best_match_first(tmp_path):
    fts_db_path = tmp_path / "fts.db"
    chunks = [
        {"id": 1, "text": "invoice invoice invoice processing system with OAuth2 OAuth2 tokens OAuth2."},
        {"id": 2, "text": "Our company kitchen has a coffee machine and nothing else relevant here."},
        {"id": 3, "text": "A brief note mentioning invoice once."},
    ]
    build_fts_index(chunks, fts_db_path)

    result = search_fts(fts_db_path, "invoice OAuth2", top_k=10)

    assert result == [1, 3]  # id 2 has no matching terms, excluded entirely


def test_search_fts_respects_top_k_limit(tmp_path):
    fts_db_path = tmp_path / "fts.db"
    chunks = [{"id": i, "text": "invoice"} for i in range(5)]
    build_fts_index(chunks, fts_db_path)

    result = search_fts(fts_db_path, "invoice", top_k=2)

    assert len(result) == 2


def test_search_fts_returns_empty_list_for_blank_query(tmp_path):
    fts_db_path = tmp_path / "fts.db"
    build_fts_index([{"id": 1, "text": "invoice"}], fts_db_path)

    assert search_fts(fts_db_path, "", top_k=10) == []
    assert search_fts(fts_db_path, "   ", top_k=10) == []


def test_search_fts_returns_empty_list_on_malformed_query_instead_of_raising(tmp_path):
    fts_db_path = tmp_path / "fts.db"
    build_fts_index([{"id": 1, "text": "hello world"}], fts_db_path)

    assert search_fts(fts_db_path, '"', top_k=10) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_fts_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.fts_index'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/rag/fts_index.py
import sqlite3
from pathlib import Path


def build_fts_index(chunks, fts_db_path):
    """(Re)create the FTS5 virtual table from scratch and populate it from
    `chunks`. Overwrites any existing file at `fts_db_path`."""
    fts_db_path = Path(fts_db_path)
    if fts_db_path.exists():
        fts_db_path.unlink()

    conn = sqlite3.connect(str(fts_db_path))
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                id UNINDEXED,
                text,
                tokenize = 'porter unicode61'
            )
            """
        )
        conn.executemany(
            "INSERT INTO chunks_fts (id, text) VALUES (?, ?)",
            [(c["id"], c["text"]) for c in chunks],
        )
        conn.commit()
    finally:
        conn.close()


def search_fts(fts_db_path, query, top_k):
    """Run one FTS5 MATCH query and return up to top_k chunk ids, best match
    first. Opens a new connection per call (SQLite connections aren't
    thread-safe; this is called concurrently from a ThreadPoolExecutor)."""
    tokens = query.split()
    if not tokens:
        return []
    match_expr = " OR ".join(f'"{token}"' for token in tokens)

    conn = sqlite3.connect(str(fts_db_path))
    try:
        cursor = conn.execute(
            "SELECT id FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?",
            (match_expr, top_k),
        )
        return [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_fts_index.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rag/fts_index.py tests/rag/test_fts_index.py
git commit -m "feat: add SQLite FTS5 build_fts_index and search_fts"
```

---

## Task 6: `expand.py` — LLM query expansion

**Files:**
- Create: `src/rag/expand.py`
- Test: `tests/rag/test_expand.py`

**Interfaces:**
- Consumes: `config.NUM_QUERY_EXPANSIONS` (Task 2); an injectable `ask_llm_fn(prompt, temperature=..., timeout=...) -> str` (matches Task 7's `rag.query.ask_llm` signature, but tests always pass a fake — no import-time coupling to `rag.query`).
- Produces: `generate_keywords(query: str, ask_llm_fn=None) -> list[str]`, called from Task 8's `retrieve()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag/test_expand.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_expand.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.expand'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/rag/expand.py
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NUM_QUERY_EXPANSIONS

PROMPT_TEMPLATE = """Extract up to 3 short search keywords or phrases from this question.
Reply with ONLY a comma-separated list, nothing else.

Question: {query}

Keywords:
"""


def _normalize(text):
    return re.sub(r"[^\w\s]", "", text).strip().lower()


def generate_keywords(query, ask_llm_fn=None):
    """Ask the LLM for up to NUM_QUERY_EXPANSIONS alternative search phrases
    for `query`. Returns [] on ANY failure (network error, timeout,
    unparseable/empty response) — never raises. `ask_llm_fn` defaults to
    `rag.query.ask_llm`, imported lazily to avoid a circular import with
    query.py (which imports generate_keywords at module level)."""
    if ask_llm_fn is None:
        from rag.query import ask_llm as ask_llm_fn

    prompt = PROMPT_TEMPLATE.format(query=query)
    try:
        response = ask_llm_fn(prompt, temperature=0.1, timeout=8.0)
    except Exception:
        return []

    normalized_query = _normalize(query)
    keywords = []
    for piece in response.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if _normalize(piece) == normalized_query:
            continue
        keywords.append(piece)

    return keywords[:NUM_QUERY_EXPANSIONS]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_expand.py -v`
Expected: PASS (all 8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rag/expand.py tests/rag/test_expand.py
git commit -m "feat: add generate_keywords LLM query expansion with fallback"
```

---

## Task 7: `query.py` — extend `ask_llm()` with temperature/timeout

**Files:**
- Modify: `src/rag/query.py:128-138`
- Test: `tests/rag/test_ask_llm.py`

**Interfaces:**
- Consumes: `rag.query.requests` (existing module-level import), `config.OLLAMA_URL`/`OLLAMA_MODEL` (existing).
- Produces: `ask_llm(prompt, temperature=None, timeout=None) -> str`, backward compatible with existing callers (`ask()` in this file, `assistant.py`). Called by Task 6's `generate_keywords()` default path.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag/test_ask_llm.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_ask_llm.py -v`
Expected: FAIL with `TypeError: ask_llm() got an unexpected keyword argument 'temperature'`

- [ ] **Step 3: Write minimal implementation**

Replace `src/rag/query.py:128-138`:

```python
def ask_llm(prompt, temperature=None, timeout=None):
    """Query Ollama LLM.

    temperature: if not None, sent as {"options": {"temperature": temperature}}.
    timeout: if not None, passed as requests.post(..., timeout=timeout).
             If None, blocks indefinitely (unchanged existing behavior).
    """
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    if temperature is not None:
        payload["options"] = {"temperature": temperature}
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    return response.json()["response"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_ask_llm.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rag/query.py tests/rag/test_ask_llm.py
git commit -m "feat: extend ask_llm with optional temperature/timeout kwargs"
```

---

## Task 8: `query.py` — hybrid `retrieve()`

**Files:**
- Modify: `src/rag/query.py:1-23` (imports/module state), `src/rag/query.py:76-90` (replaces the old `retrieve()`)
- Test: `tests/rag/test_retrieve.py`

**Interfaces:**
- Consumes: `rag.fusion.rrf_fuse` (Task 4), `rag.fts_index.search_fts` (Task 5), `rag.expand.generate_keywords` (Task 6), `config.FTS_DB_PATH`/`RRF_K`/`NUM_QUERY_EXPANSIONS`/`CANDIDATE_K` (Task 2).
- Produces: `_vector_search(query, top_k) -> list[int]`, `_hybrid_search_task(search_fn, variants, top_k, rrf_k) -> list[int]`, and the rewritten `retrieve(query) -> list[dict]`. Also a new module-level `chunks_by_id: dict[int, dict]`, kept in sync with `chunks` — used by Task 10 (`benchmark.py`). No caller-facing change: `ask()` in this file and `assistant.py`'s `query()` keep working unmodified.

This task has three TDD cycles (one function each), committed separately.

### Cycle 8a: `_vector_search`

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag/test_retrieve.py
import numpy as np
import faiss

import rag.query as query


def test_vector_search_returns_ids_from_index_search(monkeypatch):
    class FakeModel:
        def encode(self, texts):
            return np.array([[0.1, 0.2]], dtype="float32")

    class FakeIndex:
        def search(self, q_emb, top_k):
            return np.array([[0.9, 0.5]]), np.array([[7, 3]])

    monkeypatch.setattr(query, "model", FakeModel())
    monkeypatch.setattr(query, "index", FakeIndex())
    monkeypatch.setattr(query, "chunks", [{"id": 7}, {"id": 3}])
    monkeypatch.setattr(faiss, "normalize_L2", lambda x: None)

    result = query._vector_search("cats", top_k=2)

    assert result == [7, 3]


def test_vector_search_returns_empty_list_when_index_not_loaded(monkeypatch):
    monkeypatch.setattr(query, "index", None)
    monkeypatch.setattr(query, "chunks", [])

    assert query._vector_search("cats", top_k=5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_retrieve.py -v`
Expected: FAIL with `AttributeError: module 'rag.query' has no attribute '_vector_search'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/rag/query.py — add this function, keep it above retrieve()
def _vector_search(query, top_k):
    """Encode `query`, normalize, search the FAISS index, return up to
    top_k chunk ids best-first. [] if index/chunks aren't loaded."""
    if index is None or len(chunks) == 0:
        return []

    q_emb = model.encode([query])
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, top_k)
    return [int(i) for i in ids[0] if i != -1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_retrieve.py -v`
Expected: PASS (the 2 `_vector_search` tests)

- [ ] **Step 5: Commit**

```bash
git add src/rag/query.py tests/rag/test_retrieve.py
git commit -m "feat: extract _vector_search helper from retrieve"
```

### Cycle 8b: `_hybrid_search_task`

- [ ] **Step 1: Write the failing test**

```python
# tests/rag/test_retrieve.py (append)
def test_hybrid_search_task_fuses_per_variant_rankings():
    calls = []

    def fake_search_fn(variant, top_k):
        calls.append((variant, top_k))
        return {"cats": [1, 2, 3], "kittens": [2, 1, 4]}[variant]

    result = query._hybrid_search_task(fake_search_fn, ["cats", "kittens"], top_k=10, rrf_k=60)

    assert calls == [("cats", 10), ("kittens", 10)]
    assert result == [1, 2, 3, 4]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_retrieve.py -v`
Expected: FAIL with `AttributeError: module 'rag.query' has no attribute '_hybrid_search_task'`

- [ ] **Step 3: Write minimal implementation**

Add near the top of `src/rag/query.py`, alongside the existing imports:

```python
from rag.fusion import rrf_fuse
```

Add the function, below `_vector_search`:

```python
def _hybrid_search_task(search_fn, variants, top_k, rrf_k):
    """Run search_fn(variant, top_k) for every variant, collect each result
    as one ranked list, and rrf_fuse() them into a single ranking."""
    per_variant_rankings = [search_fn(variant, top_k) for variant in variants]
    return rrf_fuse(per_variant_rankings, k=rrf_k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_retrieve.py -v`
Expected: PASS (3 tests so far)

- [ ] **Step 5: Commit**

```bash
git add src/rag/query.py tests/rag/test_retrieve.py
git commit -m "feat: add _hybrid_search_task to fuse per-variant rankings"
```

### Cycle 8c: hybrid `retrieve()`

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag/test_retrieve.py (append)
def test_retrieve_fuses_vector_and_fts_branches_and_falls_back_without_keywords(monkeypatch):
    fake_chunks = [
        {"id": 1, "text": "vector best", "source": "a.txt", "chunk_id": 0},
        {"id": 2, "text": "shared", "source": "b.txt", "chunk_id": 0},
        {"id": 3, "text": "fts best", "source": "c.txt", "chunk_id": 0},
    ]
    monkeypatch.setattr(query, "index", object())
    monkeypatch.setattr(query, "chunks", fake_chunks)
    monkeypatch.setattr(query, "chunks_by_id", {c["id"]: c for c in fake_chunks})
    monkeypatch.setattr(query, "generate_keywords", lambda q: [])

    monkeypatch.setattr(query, "_vector_search", lambda variant, top_k: [1, 2])
    monkeypatch.setattr(query, "search_fts", lambda fts_db_path, variant, top_k: [3, 2])

    result = query.retrieve("anything")

    # id 2 appears in both branches -> boosted above single-branch-only ids
    assert [c["id"] for c in result] == [2, 1, 3]


def test_retrieve_uses_generated_keywords_as_additional_query_variants(monkeypatch):
    fake_chunks = [
        {"id": 1, "text": "a", "source": "a.txt", "chunk_id": 0},
        {"id": 2, "text": "b", "source": "b.txt", "chunk_id": 0},
    ]
    monkeypatch.setattr(query, "index", object())
    monkeypatch.setattr(query, "chunks", fake_chunks)
    monkeypatch.setattr(query, "chunks_by_id", {c["id"]: c for c in fake_chunks})
    monkeypatch.setattr(query, "generate_keywords", lambda q: ["synonym"])

    vector_calls = []
    fts_calls = []

    def fake_vector_search(variant, top_k):
        vector_calls.append(variant)
        return [1, 2]

    def fake_search_fts(fts_db_path, variant, top_k):
        fts_calls.append(variant)
        return [2, 1]

    monkeypatch.setattr(query, "_vector_search", fake_vector_search)
    monkeypatch.setattr(query, "search_fts", fake_search_fts)

    query.retrieve("original question")

    assert vector_calls == ["original question", "synonym"]
    assert fts_calls == ["original question", "synonym"]


def test_retrieve_returns_empty_list_when_index_not_loaded(monkeypatch):
    monkeypatch.setattr(query, "index", None)
    monkeypatch.setattr(query, "chunks", [])
    monkeypatch.setattr(query, "_ensure_index_exists", lambda: False)

    assert query.retrieve("anything") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_retrieve.py -v`
Expected: FAIL — old `retrieve()` doesn't call `generate_keywords`/`search_fts`, so `vector_calls`/`fts_calls` stay empty and the first assertion mismatches (`AssertionError`), or `AttributeError` if `search_fts` isn't imported into `rag.query` yet.

- [ ] **Step 3: Write minimal implementation**

Add these imports near the top of `src/rag/query.py` (alongside the existing `from config import (...)` block and the `from rag.fusion import rrf_fuse` added in Cycle 8b):

```python
from concurrent.futures import ThreadPoolExecutor
from rag.fts_index import search_fts
from rag.expand import generate_keywords
from config import (
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    OLLAMA_URL,
    OLLAMA_MODEL,
    TOP_K,
    FTS_DB_PATH,
    RRF_K,
    NUM_QUERY_EXPANSIONS,
    CANDIDATE_K,
)
```

Add module-level state right after the existing `model = SentenceTransformer(EMBEDDING_MODEL)` line:

```python
src_dir = Path(__file__).parent.parent
fts_db_path = src_dir / FTS_DB_PATH

# Global variables for index and chunks
index = None
chunks = []
chunks_by_id = {}
```

(This replaces the old `index = None` / `chunks = []` pair — keep everything else in `_ensure_index_exists()` as-is except for the two spots noted below.)

In `_ensure_index_exists()`, change the `global` line and rebuild `chunks_by_id` wherever `chunks` is loaded:

```python
def _ensure_index_exists():
    """Ensure FAISS index exists, build it if it doesn't."""
    global index, chunks, chunks_by_id
```

and after each of the two existing `chunks = pickle.load(f)` lines, add:

```python
        chunks_by_id = {c["id"]: c for c in chunks}
```

Replace `src/rag/query.py:76-90` (the old `retrieve()`) with:

```python
def retrieve(query):
    """Hybrid retrieval: query expansion -> parallel vector+FTS -> RRF fusion."""
    if index is None or len(chunks) == 0:
        if not _ensure_index_exists():
            return []

    if index is None or len(chunks) == 0:
        return []

    keywords = generate_keywords(query)
    variants = [query] + keywords[:NUM_QUERY_EXPANSIONS]

    def fts_search_fn(variant, top_k):
        return search_fts(fts_db_path, variant, top_k)

    with ThreadPoolExecutor(max_workers=2) as executor:
        vector_future = executor.submit(
            _hybrid_search_task, _vector_search, variants, CANDIDATE_K, RRF_K
        )
        fts_future = executor.submit(
            _hybrid_search_task, fts_search_fn, variants, CANDIDATE_K, RRF_K
        )
        vector_fused = vector_future.result()
        fts_fused = fts_future.result()

    final_ids = rrf_fuse([vector_fused, fts_fused], k=RRF_K)[:TOP_K]
    return [chunks_by_id[i] for i in final_ids if i in chunks_by_id]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_retrieve.py -v`
Expected: PASS (all tests in this file)

- [ ] **Step 5: Commit**

```bash
git add src/rag/query.py tests/rag/test_retrieve.py
git commit -m "feat: rewrite retrieve() as hybrid vector+FTS+RRF pipeline"
```

---

## Task 9: `build_index.py` — IndexIDMap + FTS build

**Files:**
- Modify: `src/rag/build_index.py:1-47`
- Test: `tests/rag/test_build_index.py`

**Interfaces:**
- Consumes: `rag.chunk.chunk_documents` (Task 3's `id`-bearing output), `rag.fts_index.build_fts_index` (Task 5), `config.FTS_DB_PATH` (Task 2).
- Produces: `build_index()` — same public signature/behavior (no return value, writes files, prints progress), now also writes `fts.db` and stores FAISS ids via `IndexIDMap` instead of implicit row positions.

- [ ] **Step 1: Write the failing test**

```python
# tests/rag/test_build_index.py
import pickle

import numpy as np
import faiss

import rag.build_index as build_index_module
from rag.fts_index import search_fts


def test_build_index_writes_faiss_idmap_chunks_and_fts_index(tmp_path, monkeypatch):
    fake_docs = [{"path": "doc.txt", "text": "irrelevant, chunking is faked"}]
    fake_chunks = [
        {"id": 0, "text": "invoice processing steps", "source": "doc.txt", "chunk_id": 0},
        {"id": 1, "text": "vacation policy details", "source": "doc.txt", "chunk_id": 1},
    ]
    fake_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")

    monkeypatch.setattr(build_index_module, "ingest_documents", lambda: fake_docs)
    monkeypatch.setattr(build_index_module, "chunk_documents", lambda docs: fake_chunks)
    monkeypatch.setattr(build_index_module, "embed_chunks", lambda chunks: fake_embeddings)

    index_path = tmp_path / "index.faiss"
    chunks_path = tmp_path / "chunks.pkl"
    fts_db_path = tmp_path / "fts.db"
    # FAISS_INDEX_PATH/CHUNKS_PATH are imported at build_index.py's module top,
    # so patch the copies bound there. FTS_DB_PATH is imported *inside*
    # build_index() (a local import each call), so patch it at the source.
    monkeypatch.setattr(build_index_module, "FAISS_INDEX_PATH", str(index_path))
    monkeypatch.setattr(build_index_module, "CHUNKS_PATH", str(chunks_path))
    monkeypatch.setattr("config.FTS_DB_PATH", str(fts_db_path))

    build_index_module.build_index()

    loaded_index = faiss.read_index(str(index_path))
    query_emb = np.array([[1.0, 0.0]], dtype="float32")
    faiss.normalize_L2(query_emb)
    _, ids = loaded_index.search(query_emb, 1)
    assert ids[0][0] == 0  # IndexIDMap preserves chunk id 0, not just position

    with open(chunks_path, "rb") as f:
        loaded_chunks = pickle.load(f)
    assert loaded_chunks == fake_chunks

    assert search_fts(fts_db_path, "invoice", top_k=10) == [0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_build_index.py -v`
Expected: FAIL — `ids[0][0] == 0` fails (current code returns a plain `IndexFlatIP`, which happens to return position `0` too in this 2-item case, so instead expect the more telling failure: `fts.db` is never created, so `search_fts` raises `sqlite3.OperationalError: unable to open database file` or the assertion on its result fails since no FTS build call exists yet).

- [ ] **Step 3: Write minimal implementation**

Replace `src/rag/build_index.py` in full:

```python
import faiss
import numpy as np
import pickle
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.ingest import ingest_documents
from rag.chunk import chunk_documents
from rag.embed import embed_chunks
from config import FAISS_INDEX_PATH, CHUNKS_PATH


def build_index():
    """Build FAISS index from documents."""
    # Resolve paths relative to src directory
    src_dir = Path(__file__).parent.parent
    index_path = src_dir / FAISS_INDEX_PATH
    chunks_path = src_dir / CHUNKS_PATH

    print("📥 Loading documents...")
    documents = ingest_documents()

    if not documents:
        print("❌ No documents found. Please add documents to the docs directory.")
        return

    print("✂️ Chunking...")
    chunks = chunk_documents(documents)

    print("🧠 Generating embeddings...")
    embeddings = embed_chunks(chunks)

    print("📦 Creating FAISS index...")
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    ids = np.array([c["id"] for c in chunks], dtype=np.int64)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    index.add_with_ids(embeddings, ids)

    print("🔎 Creating FTS5 index...")
    from rag.fts_index import build_fts_index
    from config import FTS_DB_PATH
    fts_db_path = src_dir / FTS_DB_PATH
    build_fts_index(chunks, fts_db_path)

    print("💾 Saving...")
    faiss.write_index(index, str(index_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Indexing complete: {len(chunks)} chunks indexed")
    print(f"   Index saved to: {index_path}")
    print(f"   Chunks saved to: {chunks_path}")
    print(f"   FTS index saved to: {fts_db_path}")


if __name__ == "__main__":
    build_index()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_build_index.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/build_index.py tests/rag/test_build_index.py
git commit -m "feat: build IndexIDMap FAISS index and FTS5 index together"
```

---

## Task 10: `benchmark.py` — before/after comparison (bonus)

**Files:**
- Create: `src/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `rag.query._vector_search`, `rag.query.chunks_by_id`, `rag.query.retrieve` (accessed via `import rag.query as rag_query` and attribute lookup, never `from rag.query import chunks_by_id`, since that name is rebound — not mutated — whenever the index reloads; importing it directly would risk capturing a stale reference), `config.TOP_K`.
- Produces: `vector_only_retrieve(query) -> list[dict]`, `run_benchmark() -> None`. Not consumed by any other module — this is a standalone report script (spec §9), not part of the retrieval pipeline.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark.py
import benchmark


def test_vector_only_retrieve_uses_single_vector_search_no_fusion(monkeypatch):
    fake_chunks_by_id = {1: {"id": 1, "text": "t", "source": "s.txt", "chunk_id": 0}}
    monkeypatch.setattr(benchmark.rag_query, "chunks_by_id", fake_chunks_by_id)
    monkeypatch.setattr(benchmark.rag_query, "_vector_search", lambda q, top_k: [1])

    result = benchmark.vector_only_retrieve("anything")

    assert result == [fake_chunks_by_id[1]]


def test_run_benchmark_with_no_sample_queries_does_not_raise(monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "SAMPLE_QUERIES", [])

    benchmark.run_benchmark()

    assert "No SAMPLE_QUERIES configured" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/benchmark.py
"""
Compares vector-only retrieval (the pre-hybrid baseline) against the new
hybrid retrieve() on a small fixed set of sample queries, using whatever
documents currently exist in DOCUMENTS_DIR (build the index first via
`python main.py build-index` if needed — this script does not build it).

Usage: python src/benchmark.py
"""
from config import TOP_K
import rag.query as rag_query

# Each entry: a query designed to be hard for embeddings (rare terms, exact
# file/command names, abbreviations), and the substring expected to appear
# in a hit's source path/text. Populate once real documents exist in
# DOCUMENTS_DIR — this list ships empty since src/docs/ is currently empty.
SAMPLE_QUERIES = [
    # {"query": "...", "expected_substring": "..."},
]


def vector_only_retrieve(query):
    """Reimplements the pre-hybrid retrieve(): single vector search, no
    expansion, no FTS, no fusion."""
    ids = rag_query._vector_search(query, TOP_K)
    return [rag_query.chunks_by_id[i] for i in ids if i in rag_query.chunks_by_id]


def run_benchmark():
    """For each sample query: run vector_only_retrieve() and retrieve(),
    print both result sets' sources, and report hit/miss on
    expected_substring for each (qualitative demo, not a formal metric)."""
    if not SAMPLE_QUERIES:
        print(
            "No SAMPLE_QUERIES configured yet — add entries once documents "
            "exist in DOCUMENTS_DIR (see spec/v1/SPEC.md §9)."
        )
        return

    for case in SAMPLE_QUERIES:
        query_text = case["query"]
        expected = case["expected_substring"]

        vector_results = vector_only_retrieve(query_text)
        hybrid_results = rag_query.retrieve(query_text)

        vector_hit = any(expected in c["source"] or expected in c["text"] for c in vector_results)
        hybrid_hit = any(expected in c["source"] or expected in c["text"] for c in hybrid_results)

        print(f"\nQuery: {query_text!r}  (expecting {expected!r})")
        print(f"  vector-only: {'HIT' if vector_hit else 'MISS'} - {[c['source'] for c in vector_results]}")
        print(f"  hybrid:      {'HIT' if hybrid_hit else 'MISS'} - {[c['source'] for c in hybrid_results]}")


if __name__ == "__main__":
    run_benchmark()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/benchmark.py tests/test_benchmark.py
git commit -m "feat: add vector-only-vs-hybrid benchmark script (bonus)"
```

---

## Task 11: Full suite regression pass

**Files:** none new — this task only runs the whole suite and, if needed, deletes the stale index files per spec §4.

**Interfaces:** none — verification task.

- [ ] **Step 1: Run the entire test suite**

Run: `pytest -v` (from repo root)
Expected: PASS — every test from Tasks 1–10.

- [ ] **Step 2: Delete the stale pre-hybrid index (spec §4 — no migration)**

```bash
rm -f src/index.faiss src/chunks.pkl
```

(If they don't exist yet in your working copy, this is a no-op — confirmed absent at plan-writing time.)

- [ ] **Step 3: Rebuild and manually smoke-test**

Run: `cd src && python main.py build-index` (creates `src/index.faiss`, `src/chunks.pkl`, `src/fts.db` against whatever is in `src/docs/`)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: verify full hybrid search suite passes end to end"
```

---

## Self-Review

**Spec coverage:**
- §6 config additions → Task 2.
- §7.1 chunk id → Task 3.
- §7.2 fts_index.py → Task 5.
- §7.3 fusion.py → Task 4.
- §7.4 expand.py → Task 6.
- §7.5 ask_llm signature → Task 7.
- §7.6 hybrid retrieve() → Task 8.
- §7.7 build_index.py → Task 9.
- §8 fallback scope → enforced by Task 6's tests (generate_keywords never raises) and Task 8's fallback test; no extra try/except added anywhere else, per Global Constraints.
- §9 benchmark.py (bonus) → Task 10.
- §10 pytest dependency → Task 1.
- §11 testing requirements (tests/ mirroring src/ layout) → satisfied by `tests/rag/test_*.py` + `tests/test_*.py` throughout.
- §4 no-migration cleanup → Task 11.

**Placeholder scan:** every step has runnable code; no "TBD"/"add error handling"/"similar to Task N" phrasing anywhere in the code steps.

**Type/signature consistency check:**
- `rrf_fuse(ranked_lists, k=60)` — same signature used in Task 4's implementation and Task 8's `_hybrid_search_task`/`retrieve` calls.
- `search_fts(fts_db_path, query, top_k)` — same 3-positional-arg order in Task 5, Task 8's `fts_search_fn` closure, Task 9's test, and Task 10 (not directly called, but consistent with `rag_query`'s internal usage).
- `generate_keywords(query, ask_llm_fn=None)` — same signature in Task 6 and Task 8's `retrieve()` call (`generate_keywords(query)`, relying on the default).
- `_vector_search(query, top_k)` / `_hybrid_search_task(search_fn, variants, top_k, rrf_k)` — defined in Task 8, called consistently within Task 8 and Task 10.
- `chunks_by_id` — introduced in Task 8, consumed by Task 10 via module attribute access (not a direct name import, to avoid the stale-binding trap called out in Task 10's Interfaces section).

No gaps found.

---

**Plan complete and saved to `docs/superpowers/plans/2026-08-30-hybrid-search.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
