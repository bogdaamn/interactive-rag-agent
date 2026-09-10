# Hybrid Search (FTS + Query Expansion) — Specification

**Status:** Ready for implementation
**Repo:** local-rag-mcp
**Baseline commit:** `a6583d4e25f6f94eef96f1829ac4f72b9981c021`
**Source assignment:** "Improving a RAG System — Hybrid Search (FTS) and Query Expansion" homework (see `docs/` in this repo / assignment brief provided by instructor)

## 1. Goal

Replace the current pure-vector `retrieve()` in `src/rag/query.py` with a hybrid
pipeline that combines:

1. **Query expansion** — an LLM-generated set of alternative search phrases for
   the user's question, with fallback to the original query if generation fails.
2. **Parallel retrieval** — FAISS vector search and a new SQLite FTS5 full-text
   index, run concurrently so FTS does not add latency on top of vector search.
3. **Reciprocal Rank Fusion (RRF)** — merge the two methods' rankings into one
   final Top-K list of chunks, which flows into the existing prompt-building /
   LLM-answer code unchanged.

Every existing caller of `retrieve()` (`rag/query.py:ask()`, `assistant.py`'s
`CompanyKBAssistant.query()`) gets hybrid retrieval automatically — no caller-side
changes. `src/mcp/server.py`'s `search_documents` tool is filename search, not
document-content search, and is explicitly **out of scope**.

## 2. Current pipeline (baseline, for context)

```
query.py:retrieve(query)
  → SentenceTransformer.encode([query])
  → faiss.normalize_L2
  → IndexFlatIP.search(q_emb, TOP_K)      # TOP_K = 5, config.py
  → [chunks[i] for i in ids[0]]           # position-based lookup
```

`chunks.pkl` is a flat `list[dict]` with keys `text`, `source`, `chunk_id` (the
last is the chunk's index *within its source document*, reset per document —
**not** globally unique. FAISS `IndexFlatIP` returns row *positions*, which
happen to equal list indices only because chunks are added in the same order
they were built, with no explicit id mapping).

## 3. New pipeline

```
User Query
     │
     ▼
[generate_keywords(query)]  ── LLM, up to 3 phrases, [] on any failure
     │
     ▼
query_variants = [query] + keywords            (≤ 4 strings total)
     │
     ├────────────────────────────┬────────────────────────────┐
     ▼                            ▼                             │
[vector branch]              [fts branch]                (run concurrently
for v in variants:            for v in variants:            via ThreadPoolExecutor,
  vector_search(v, CANDIDATE_K)  search_fts(v, CANDIDATE_K)  2 workers)
  → ranked [chunk_id,...]        → ranked [chunk_id,...]
rrf_fuse(all vector rankings)  rrf_fuse(all fts rankings)
  → vector_fused                 → fts_fused
     │                            │
     └────────────┬───────────────┘
                  ▼
     rrf_fuse([vector_fused, fts_fused])[:TOP_K]
                  │
                  ▼
     [chunk dicts, via chunks_by_id lookup]
                  │
                  ▼
        build_prompt() / ask_llm()   (unchanged)
```

RRF is used **twice**, with the same function and the same `k`: once inside
each branch to combine that branch's per-variant rankings into one ranking,
and once more to fuse the two branches' rankings into the final list. This
keeps the fusion logic in a single, well-tested function.

## 4. Breaking changes / no migration

This is a from-scratch student project with no production data. Adding a
globally-unique chunk `id` changes the shape of `chunks.pkl`, `index.faiss`
gains an `IndexIDMap` wrapper, and a new `fts.db` file is introduced. **No
migration code is required.** Delete existing `src/index.faiss` and
`src/chunks.pkl` (if present) and rebuild via `python main.py build-index`
after implementing this spec. State this as a one-line note in the plan/PR,
don't write backfill code for it.

## 5. File-by-file changes

| File | Change |
|---|---|
| `src/config.py` | Add `FTS_DB_PATH`, `RRF_K`, `NUM_QUERY_EXPANSIONS`, `CANDIDATE_K` |
| `src/rag/chunk.py` | Add a globally-unique `id` field to every chunk dict |
| `src/rag/fts_index.py` | **New.** Build + query the SQLite FTS5 index |
| `src/rag/fusion.py` | **New.** `rrf_fuse()` |
| `src/rag/expand.py` | **New.** `generate_keywords()` (LLM query expansion) |
| `src/rag/build_index.py` | Build FAISS with `IndexIDMap` + build the FTS index |
| `src/rag/query.py` | `ask_llm()` gains optional `temperature`/`timeout` kwargs; `retrieve()` becomes the hybrid pipeline; internal helpers `_vector_search()`, `_hybrid_search_task()` split out |
| `src/assistant.py` | No change — already calls `retrieve()` |
| `src/mcp/server.py` | No change (out of scope, see §1) |
| `src/benchmark.py` | **New** (bonus, §9). Vector-only vs hybrid comparison script |

## 6. Config additions (`src/config.py`)

```python
# Hybrid search configuration
FTS_DB_PATH = "fts.db"          # relative to src dir, mirrors FAISS_INDEX_PATH
RRF_K = 60                       # RRF constant k, per assignment's formula
NUM_QUERY_EXPANSIONS = 3         # max alternative phrases the LLM may produce
CANDIDATE_K = TOP_K * 3          # candidate pool size per method, before fusion
```

`TOP_K` (existing, `= 5`) keeps its name but its meaning changes: it is now the
**final** number of chunks returned after RRF fusion, not the raw vector
search top-k. Each method internally retrieves `CANDIDATE_K` candidates per
query variant before any fusion narrows the pool down to `TOP_K`.

## 7. Component specs

### 7.1 `src/rag/chunk.py` — global chunk id

`chunk_documents()` already builds `all_chunks` in a single pass. Add a second
pass that assigns a globally-unique `id` (0-based, list position) to every
chunk, **in addition to** the existing per-document `chunk_id` field (left
untouched — nothing else in the codebase reads it, no need to remove it):

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
                "chunk_id": idx,   # existing field, per-document index — unchanged
            })

    for global_id, chunk in enumerate(all_chunks):
        chunk["id"] = global_id   # NEW: globally unique, used by FAISS + FTS + RRF

    return all_chunks
```

`embed_chunks()` in `embed.py` is unaffected (it only reads `c["text"]`).

### 7.2 `src/rag/fts_index.py` — SQLite FTS5 (new file)

```python
import sqlite3
from pathlib import Path


def build_fts_index(chunks: list[dict], fts_db_path: Path) -> None:
    """
    (Re)create the FTS5 virtual table from scratch and populate it from
    `chunks`. Overwrites any existing file at `fts_db_path`.

    Table schema:
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            id UNINDEXED,
            text,
            tokenize = 'porter unicode61'
        );

    `id` is stored as an UNINDEXED column (not tokenized, just carried through)
    so a MATCH query can SELECT it directly without a rowid join.
    """


def search_fts(fts_db_path: Path, query: str, top_k: int) -> list[int]:
    """
    Run one FTS5 MATCH query against `fts_db_path` and return up to `top_k`
    chunk ids, best match first.

    Opens a NEW sqlite3.connect() for this call (SQLite connections are not
    thread-safe by default; this function will be called concurrently from a
    ThreadPoolExecutor, so each call/thread gets its own connection — do not
    share a module-level connection object across threads).

    The raw `query` string is NOT passed to MATCH verbatim: FTS5's query
    syntax treats characters like `"`, `-`, `:`, `*` specially and will raise
    `sqlite3.OperationalError` on arbitrary user text (e.g. "how do I use
    OAuth2.0?" contains a bare `?` and a `.` that are safe, but "AND"/"OR"/"-"
    as literal words, or unmatched quotes, are not). Build a safe MATCH
    expression by splitting `query` on whitespace, discarding empty tokens,
    double-quoting each token, and OR-ing them:

        "invoice" OR "processing" OR "system"

    This gives keyword-OR matching (any token may match) ranked by bm25().
    If `query` has no tokens after splitting, or the resulting MATCH raises
    sqlite3.OperationalError, return [] (caller treats this the same as "no
    FTS results for this variant" — it must NOT raise).

    SQL: SELECT id FROM chunks_fts WHERE chunks_fts MATCH ?
         ORDER BY bm25(chunks_fts) LIMIT ?
         (bm25() ascending = best match first, per SQLite FTS5 docs — smaller
         bm25 score is a better match)
    """
```

### 7.3 `src/rag/fusion.py` — Reciprocal Rank Fusion (new file)

```python
def rrf_fuse(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    """
    Merge N ranked lists of chunk ids into one ranking using Reciprocal Rank
    Fusion:

        score(id) = sum over each list L that contains id of  1 / (k + rank_L(id))

    where rank_L(id) is the 1-based position of `id` within list L (best = 1).
    An id absent from a list contributes 0 for that list — it is NOT penalized
    beyond simply not getting that list's term.

    Returns chunk ids sorted by descending score. Ties are broken by: (1) the
    id's best (lowest) rank in any single input list, then (2) the id's
    numeric value, for a fully deterministic order (needed for reproducible
    tests).

    `ranked_lists` may contain empty lists (e.g. a branch that returned no
    results) — treat as contributing nothing, don't error.
    Duplicate ids WITHIN a single ranked list should not occur (caller's
    responsibility) but if they do, only the first (best-ranked) occurrence
    counts.
    """
```

### 7.4 `src/rag/expand.py` — LLM query expansion (new file)

```python
def generate_keywords(query: str, ask_llm_fn=None) -> list[str]:
    """
    Ask the LLM for up to NUM_QUERY_EXPANSIONS (3) alternative search phrases
    for `query`. Returns a list of 0-3 non-empty, stripped strings, best-effort
    order as returned by the model. Returns [] on ANY failure — network error,
    timeout, empty/unparseable response — never raises. Callers must treat []
    as "use the original query alone."

    `ask_llm_fn` defaults to `rag.query.ask_llm` (injected as a parameter, not
    imported at call time, so tests can pass a fake without monkeypatching
    module state).

    Prompt (kept short per the small-model tip — 0.6B/1.5B/3B Qwen):

        Extract up to 3 short search keywords or phrases from this question.
        Reply with ONLY a comma-separated list, nothing else.

        Question: {query}

        Keywords:

    Call pattern: `ask_llm_fn(prompt, temperature=0.1, timeout=8.0)`.

    Parsing: split the raw response on "," → strip each piece → drop empty
    strings → drop any piece that, case-insensitively stripped of
    punctuation, equals the original query verbatim (small models sometimes
    just echo the question back as "keyword #1") → truncate to
    NUM_QUERY_EXPANSIONS (3) items. If parsing yields zero usable phrases,
    return [].

    Failure modes to catch (all → return []):
      - requests.exceptions.RequestException (includes Timeout) raised by
        ask_llm_fn
      - any other Exception raised by ask_llm_fn (defensive: a small model's
        client library failing shouldn't take down retrieval)
      - a response that, after parsing, yields an empty list
    """
```

### 7.5 `src/rag/query.py` — `ask_llm()` signature change

Extend (not replace) the existing function so all current callers keep working
with zero changes:

```python
def ask_llm(prompt, temperature=None, timeout=None):
    """Query Ollama LLM.

    temperature: if not None, sent as {"options": {"temperature": temperature}}
                 in the request body (Ollama /api/generate accepts this).
    timeout: if not None, passed as `requests.post(..., timeout=timeout)`.
             If None, blocks indefinitely — unchanged existing behavior.
    """
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    if temperature is not None:
        payload["options"] = {"temperature": temperature}
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    return response.json()["response"]
```

### 7.6 `src/rag/query.py` — hybrid `retrieve()`

Module-level state changes:
- `index` is now built with `faiss.IndexIDMap(faiss.IndexFlatIP(dim))` (built in
  `build_index.py`, loaded here via `faiss.read_index` exactly as before — the
  ID-map wrapper is transparent to `read_index`/`write_index`).
- Add `chunks_by_id: dict[int, dict]`, rebuilt whenever `chunks` is (re)loaded:
  `{c["id"]: c for c in chunks}`.
- Add `fts_db_path`, resolved the same way `FAISS_INDEX_PATH`/`CHUNKS_PATH`
  already are (relative to `src/`).

New/changed functions:

```python
def _vector_search(query: str, top_k: int) -> list[int]:
    """Encode `query`, normalize, search the FAISS IndexIDMap, return up to
    top_k chunk ids best-first. Returns [] if index/chunks aren't loaded."""


def _hybrid_search_task(search_fn, variants: list[str], top_k: int, rrf_k: int) -> list[int]:
    """Run search_fn(variant, top_k) for every variant in `variants`
    sequentially, collect each result as one ranked list, and rrf_fuse() them
    into a single ranking. `search_fn` is either `_vector_search` or a
    closure over `search_fts(fts_db_path, ...)` — this function is what gets
    submitted to the ThreadPoolExecutor, once per method."""


def retrieve(query: str) -> list[dict]:
    """Hybrid retrieval: query expansion → parallel vector+FTS → RRF fusion.

    1. keywords = generate_keywords(query)   # [] on any failure
       variants = [query] + keywords[:NUM_QUERY_EXPANSIONS]
    2. Ensure index/chunks loaded (existing _ensure_index_exists() call,
       unchanged) — if unavailable, return [] exactly as today.
    3. Submit two tasks to a ThreadPoolExecutor(max_workers=2):
         - _hybrid_search_task(_vector_search, variants, CANDIDATE_K, RRF_K)
         - _hybrid_search_task(fts_search_fn, variants, CANDIDATE_K, RRF_K)
           where fts_search_fn(v, k) = search_fts(fts_db_path, v, k)
       `.result()` both (propagates exceptions from either branch — per §8,
       only the keyword-generation step is required to degrade gracefully;
       a hard FAISS/SQLite failure is allowed to raise).
    4. final_ids = rrf_fuse([vector_fused, fts_fused], RRF_K)[:TOP_K]
    5. return [chunks_by_id[i] for i in final_ids if i in chunks_by_id]
    """
```

`build_prompt()` and `ask()` are unchanged — they already only rely on
`c["source"]` / `c["text"]`, which every returned chunk dict still has.

### 7.7 `src/rag/build_index.py`

After the existing FAISS build, also build the FTS index:

```python
def build_index():
    ...
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
    ...
```

(`import numpy as np` needs adding to `build_index.py`'s imports.)

## 8. Fallback / error handling (task 4)

**In scope (must implement):** if `generate_keywords()` fails for any reason
— LLM unreachable, timeout, malformed/empty response — `retrieve()` must
proceed using `variants = [query]` only, and must NOT raise or crash. This is
enforced by `generate_keywords()` itself always returning `[]` rather than
raising (§7.4); `retrieve()` never needs its own try/except around the call.

**Out of scope (per spec decision, do not add defensive code for these):** a
hard failure inside `_vector_search` (e.g. corrupt `index.faiss`) or
`search_fts` beyond the "malformed MATCH query" case already handled in §7.2
is allowed to raise out of `retrieve()`. Do not add blanket try/except around
the FAISS or ThreadPoolExecutor calls — that's a deliberately smaller surface
than "handle everything," matching the assignment's bonus wording ("if the
LLM fails to return keywords... use the original query").

## 9. Benchmark (bonus, +10%)

**New file:** `src/benchmark.py`.

Purpose: demonstrate hybrid retrieval improves recall over vector-only search
on queries where vector embeddings are known to struggle — rare terms, exact
file/command names, abbreviations (per the assignment's stated motivation).

Structure:

```python
"""
Compares vector-only retrieval (the pre-hybrid baseline) against the new
hybrid retrieve() on a small fixed set of sample queries, using whatever
documents currently exist in DOCUMENTS_DIR (build the index first via
`python main.py build-index` if needed — this script does not build it).

Usage: python src/benchmark.py
"""

SAMPLE_QUERIES = [
    # Each entry: a query designed to be hard for embeddings, and the
    # substring expected to appear in a source path/text if hybrid search
    # is working. Populate `expected_substring` per whatever docs actually
    # exist in DOCUMENTS_DIR at benchmark time — this list is intentionally
    # a template, not fixed data, since the repo ships an empty docs/ dir.
]

def vector_only_retrieve(query: str) -> list[dict]:
    """Reimplements the OLD retrieve() behavior for comparison: single
    _vector_search(query, TOP_K) call, no expansion, no FTS, no fusion."""

def run_benchmark():
    """For each sample query: run vector_only_retrieve() and retrieve(),
    print both result sets' sources side by side, and report whether
    expected_substring appears in each method's results (simple hit/miss,
    not a formal precision/recall metric — this is a qualitative bonus
    demo, not a test)."""
```

This script is explicitly a demo/report tool, not a pytest test — it prints a
human-readable before/after comparison. The implementing agent should add at
least 3-5 `SAMPLE_QUERIES` once real documents exist in `DOCUMENTS_DIR`
(placeholder/sample `.md` files may be added under `src/docs/` for this
purpose if the directory is still empty — note in the PR description that
these are benchmark fixtures, not real company docs).

## 10. New dependency

None. `sqlite3` is Python stdlib (FTS5 confirmed available in this
environment: `sqlite3.sqlite_version == 3.53.0`, `CREATE VIRTUAL TABLE ...
USING fts5(...)` succeeds). No `requirements.txt` change needed.

For running tests (§11), add to `requirements.txt`:
```
pytest>=7.4.0
```

## 11. Testing requirements

No test suite exists in this repo today (no `tests/` dir, pytest not a
dependency). This spec requires one: `tests/`, mirroring `src/`'s layout
(`tests/rag/test_fusion.py`, `tests/rag/test_fts_index.py`,
`tests/rag/test_expand.py`, `tests/rag/test_query.py`, etc.). Each of
`rrf_fuse`, `build_fts_index`/`search_fts`, and `generate_keywords` is a pure
enough function (or takes an injectable `ask_llm_fn`) to unit test without a
running Ollama server or a real FAISS index. `retrieve()`'s hybrid assembly
should be tested with `_vector_search` and `search_fts` monkeypatched/faked so
the test doesn't require a built index.

See the accompanying implementation plan (`docs/superpowers/plans/`) for exact
test-by-test TDD steps.

## 12. Non-goals

- `src/mcp/server.py` is unchanged (§1).
- No migration path for old `chunks.pkl`/`index.faiss` (§4).
- No new config UI/CLI flags beyond the `config.py` constants in §6.
- No change to `assistant.py`'s MCP-tool-decision logic.
- No formal precision/recall metric computation in the benchmark (§9) — it's
  a qualitative before/after demo for the bonus points, not a test.

## 13. Acceptance criteria (maps to grading rubric)

| Rubric item | Satisfied by |
|---|---|
| Query Expansion (25%) | §7.4 `generate_keywords()`, short prompt, temperature 0.1, ≤3 phrases |
| Parallel FTS (35%) | §7.2 `search_fts()` (SQLite FTS5) + §7.6 `retrieve()`'s `ThreadPoolExecutor(max_workers=2)` running vector and FTS branches concurrently |
| Hybrid Fusion (25%) | §7.3 `rrf_fuse()`, k=60, used both intra-branch and inter-branch |
| Error Handling & Code Quality (15%) | §8 fallback scope, §7.4's catch-all in `generate_keywords`, injectable `ask_llm_fn` for testability |
| Bonus: before/after benchmark (+10%) | §9 `src/benchmark.py` |
