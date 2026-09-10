# RAG Retrieval, `search_documents` Tool & Agent Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the retrieval half of the pipeline — per-user vector search plus hybrid text search, RRF fusion, cross-encoder reranking, and the relevance threshold that enforces the no-hallucination rule — then expose it to the LLM as a `search_documents(query)` tool driven by a real multi-step agent loop.

**Architecture:** `retrieve.py` orchestrates: vector search (from the ingestion plan's `store.search_vectors`) fused with FTS5 keyword search (`textsearch.py`) via Reciprocal Rank Fusion (`fusion.py`), truncated to Top-K, reranked by a cross-encoder (`rerank.py`), then filtered by `RELEVANCE_THRESHOLD`. `tools.py` wraps `retrieve()` in a `Tool` bound to one `user_id` — the LLM's tool schema has no `user_id` parameter at all, so a model can't request another user's documents. `agent.py` is a step-budgeted loop that calls the LLM, invokes any requested tools, and feeds results back.

**Tech Stack:** Python 3.10+, stdlib `sqlite3` FTS5, `sentence-transformers` `CrossEncoder`, `numpy`. No new dependencies beyond what the ingestion plan already added.

**Spec:** `spec/v3/SPEC.md` §8 (retrieval, hybrid search, reranking), §11 (tool + agent), §13 (source attribution), §14 (no-hallucination rule)

**Depends on:** `docs/superpowers/plans/2026-09-10-document-ingestion-pipeline.md` — every task here consumes `UserDocsStore`, `embed_chunks`, and `userdocs.config`/`userdocs.errors` from that plan. Do not start this plan until that one is complete and green.

## Global Constraints

- `TOP_K = 5`, `CANDIDATE_K = 15`, `RRF_K = 60`, `RELEVANCE_THRESHOLD = 0.30` — all from `userdocs/config.py` (spec §6). Never hardcode these numbers in a module; import them.
- **The no-hallucination trigger is "no chunk survives `RELEVANCE_THRESHOLD` cosine similarity"**, not "retrieval returned zero rows" (spec §14). Vector search always returns *something*, so a zero-rows check would never fire for an irrelevant question.
- The threshold is applied to the **pre-rerank cosine similarity**, and reranking only reorders candidates that already passed (spec §8.3's note) — a cross-encoder's raw score is not a calibrated cosine value, so it must never be compared against `RELEVANCE_THRESHOLD`.
- Every retrieval path is `user_id`-scoped: `retrieve()` takes `user_id` and passes it to both `store.search_vectors` and `textsearch.search_fts` (spec §10). FTS5 has no per-row ACL, so `search_fts` narrows to this user's chunk ids first, exactly like `search_vectors` does.
- Reranking failing must degrade retrieval *quality*, never *availability* — `retrieve()` catches `RerankError` and falls back to the pre-rerank order (spec §8.3).
- The `search_documents` tool is built **per incoming message**, bound to that message's `user_id` in trusted code (spec §11.1) — never a long-lived global tool with a `user_id` argument the LLM fills in.
- TDD, strictly: write the failing test, run it and confirm the failure reason, write the minimal implementation, run it green, commit — one behavior per cycle.

---

### Task 1: RRF fusion

**Files:**
- Create: `src/userdocs/fusion.py`
- Test: `tests/userdocs/test_fusion.py`

**Interfaces:**
- Consumes: `userdocs.config.RRF_K`.
- Produces: `userdocs.fusion.rrf_fuse(ranked_lists: list[list[int]], k: int = RRF_K) -> list[int]` — a single fused list of chunk ids, best first.

> Note: `src/rag/fusion.py` already implements RRF for the *company-KB* pipeline. This is a deliberate second copy in `userdocs/` rather than an import, because the two subsystems are kept independent per spec §1 ("additive, not a rewrite") — importing across them would couple `userdocs` to `src/rag`'s config and module-load side effects (`src/rag/query.py` runs `_ensure_index_exists()` at import time). The implementation below is short enough that duplication costs less than that coupling.

- [ ] **Step 1: Write the failing test**

```python
# tests/userdocs/test_fusion.py
from userdocs.fusion import rrf_fuse


def test_rrf_fuse_single_list_preserves_order():
    assert rrf_fuse([[10, 20, 30]]) == [10, 20, 30]


def test_rrf_fuse_ranks_item_appearing_in_both_lists_highest():
    # id 20 is rank 2 in list A and rank 1 in list B -> best combined score
    fused = rrf_fuse([[10, 20, 30], [20, 40]])
    assert fused[0] == 20


def test_rrf_fuse_handles_empty_lists():
    assert rrf_fuse([[], []]) == []
    assert rrf_fuse([[], [5, 6]]) == [5, 6]


def test_rrf_fuse_breaks_ties_by_best_rank_then_id():
    # id 1 and id 2 both appear once at rank 1 in separate lists -> equal
    # scores; tie broken by numeric id ascending for determinism
    assert rrf_fuse([[2], [1]]) == [1, 2]


def test_rrf_fuse_deduplicates_within_a_single_list():
    fused = rrf_fuse([[7, 7, 8]])
    assert fused == [7, 8]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'userdocs.fusion'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/fusion.py
"""Reciprocal Rank Fusion for combining vector and text search results.

See spec/v3/SPEC.md §8.1 step 2. score(id) = sum over lists L of
1 / (k + rank_L(id)), where rank is 1-based. Higher score = better.
"""

from userdocs.config import RRF_K


def rrf_fuse(ranked_lists: list, k: int = RRF_K) -> list:
    scores: dict = {}
    best_rank: dict = {}

    for ranked_list in ranked_lists:
        seen_in_this_list = set()
        rank = 0
        for item_id in ranked_list:
            if item_id in seen_in_this_list:
                continue  # first occurrence wins within one list
            seen_in_this_list.add(item_id)
            rank += 1
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in best_rank or rank < best_rank[item_id]:
                best_rank[item_id] = rank

    # descending score, then ascending best rank, then ascending id (fully
    # deterministic — no dependence on dict insertion order)
    return sorted(
        scores.keys(),
        key=lambda item_id: (-scores[item_id], best_rank[item_id], item_id),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_fusion.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/fusion.py tests/userdocs/test_fusion.py
git commit -m "feat(userdocs): add RRF fusion for hybrid search result combination"
```

---

### Task 2: FTS5 index creation in the store

**Files:**
- Modify: `src/userdocs/store.py`
- Modify: `tests/userdocs/test_store.py`

**Interfaces:**
- Consumes: `UserDocsStore` from the ingestion plan (Tasks 9–13).
- Produces: a `chunks_fts` FTS5 virtual table created in `UserDocsStore.__init__`, populated by `insert_chunks_with_vectors` (extended here) and cleaned up by `delete_document` (extended here). No new public methods.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/userdocs/test_store.py
def test_insert_chunks_also_populates_fts_index(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    doc_id = store.insert_document(1, "policy.txt", ".txt")
    store.insert_chunks_with_vectors(
        doc_id,
        [{"text": "employees receive vacation days", "chunk_index": 0, "page": None}],
        np.zeros((1, 384), dtype=np.float32),
    )

    rows = store._conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'vacation'"
    ).fetchall()
    assert len(rows) == 1

    chunk_id = store._conn.execute("SELECT id FROM chunks").fetchone()[0]
    assert rows[0][0] == chunk_id
    store.close()


def test_delete_document_also_clears_fts_rows(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    doc_id = store.insert_document(1, "policy.txt", ".txt")
    store.insert_chunks_with_vectors(
        doc_id,
        [{"text": "employees receive vacation days", "chunk_index": 0, "page": None}],
        np.zeros((1, 384), dtype=np.float32),
    )

    store.delete_document(user_id=1, filename="policy.txt")

    rows = store._conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'vacation'"
    ).fetchall()
    assert rows == []
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_store.py -v`
Expected: FAIL with `sqlite3.OperationalError: no such table: chunks_fts`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/store.py — in __init__, after the chunk_vectors CREATE, add:

            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING "
                "fts5(text, tokenize='porter unicode61')"
            )
```

```python
# src/userdocs/store.py — in insert_chunks_with_vectors, inside the
# `for chunk, vector in zip(...)` loop, right after the chunk_vectors INSERT:

                self._conn.execute(
                    "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
                    (chunk_id, chunk["text"]),
                )
```

```python
# src/userdocs/store.py — in delete_document, inside the `if chunk_ids:` block,
# right after the chunk_vectors DELETE:

                self._conn.execute(
                    f"DELETE FROM chunks_fts WHERE rowid IN ({placeholders})", chunk_ids
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_store.py -v`
Expected: PASS (14 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/store.py tests/userdocs/test_store.py
git commit -m "feat(userdocs): maintain an FTS5 keyword index alongside chunks and vectors"
```

---

### Task 3: `search_fts` — per-user keyword search (bonus: hybrid search)

**Files:**
- Create: `src/userdocs/textsearch.py`
- Create: `tests/userdocs/test_textsearch.py`

**Interfaces:**
- Consumes: `UserDocsStore` (its `_conn`, plus the `chunks_fts` table from Task 2).
- Produces: `userdocs.textsearch.search_fts(store, user_id: int, query: str, k: int) -> list[int]` — chunk ids, best BM25 match first, `[]` on a malformed query (never raises).

- [ ] **Step 1: Write the failing test**

```python
# tests/userdocs/test_textsearch.py
import numpy as np

from userdocs.store import UserDocsStore
from userdocs.textsearch import search_fts


def _ingest(store, user_id, filename, texts):
    doc_id = store.insert_document(user_id, filename, ".txt")
    chunks = [{"text": t, "chunk_index": i, "page": None} for i, t in enumerate(texts)]
    store.insert_chunks_with_vectors(
        doc_id, chunks, np.zeros((len(texts), 384), dtype=np.float32)
    )
    return doc_id


def test_search_fts_finds_matching_chunk(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    _ingest(store, 1, "policy.txt", ["employees receive vacation days", "office hours are 9 to 5"])

    results = search_fts(store, user_id=1, query="vacation", k=5)

    assert len(results) == 1
    chunk_text = store.get_chunk(results[0])["text"]
    assert "vacation" in chunk_text
    store.close()


def test_search_fts_never_returns_another_users_chunks(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    _ingest(store, 1, "secret.txt", ["confidential vacation policy for executives"])
    _ingest(store, 2, "public.txt", ["general office information"])

    # user 2 searches for a term that only exists in user 1's document
    results = search_fts(store, user_id=2, query="vacation", k=10)

    assert results == []
    store.close()


def test_search_fts_returns_empty_for_user_with_no_documents(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    assert search_fts(store, user_id=999, query="anything", k=5) == []
    store.close()


def test_search_fts_returns_empty_on_malformed_query_instead_of_raising(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    _ingest(store, 1, "policy.txt", ["employees receive vacation days"])

    # unbalanced quote / FTS5 operator soup would be a syntax error if passed raw
    assert search_fts(store, user_id=1, query='"', k=5) == []
    assert search_fts(store, user_id=1, query="AND OR NEAR", k=5) == []
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_textsearch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'userdocs.textsearch'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/textsearch.py
"""FTS5 keyword search over user document chunks (bonus: hybrid search).

See spec/v3/SPEC.md §8.2. Mirrors src/rag/fts_index.py's query-sanitization
approach (split on whitespace, double-quote each token, OR them together) so a
user's raw question can never be interpreted as FTS5 operator syntax, but adds
the per-user chunk-id narrowing that spec §10 requires — FTS5 has no per-row
access control of its own.
"""

import sqlite3


def _safe_match_expression(query: str) -> str:
    """Turn arbitrary user text into a valid FTS5 MATCH expression.

    Each whitespace-separated token is double-quoted (which makes FTS5 treat
    it as a literal string, not an operator), embedded double quotes are
    doubled per FTS5's own escaping rules, and the tokens are OR-ed so any
    match counts. Returns "" if nothing usable remains.
    """
    tokens = []
    for raw_token in query.split():
        cleaned = raw_token.replace('"', '""').strip()
        if cleaned:
            tokens.append(f'"{cleaned}"')
    return " OR ".join(tokens)


def _user_chunk_ids(store, user_id: int) -> set:
    document_ids = [
        row[0]
        for row in store._conn.execute(
            "SELECT id FROM documents WHERE user_id = ?", (user_id,)
        ).fetchall()
    ]
    if not document_ids:
        return set()
    placeholders = ",".join("?" for _ in document_ids)
    return {
        row[0]
        for row in store._conn.execute(
            f"SELECT id FROM chunks WHERE document_id IN ({placeholders})", document_ids
        ).fetchall()
    }


def search_fts(store, user_id: int, query: str, k: int) -> list:
    chunk_ids = _user_chunk_ids(store, user_id)
    if not chunk_ids:
        return []

    match_expression = _safe_match_expression(query)
    if not match_expression:
        return []

    try:
        rows = store._conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?",
            (match_expression, max(k, len(chunk_ids))),
        ).fetchall()
    except sqlite3.OperationalError:
        # A malformed MATCH expression must degrade to "no keyword hits",
        # never crash the whole retrieval path.
        return []

    results = []
    for (rowid,) in rows:
        if rowid in chunk_ids:
            results.append(rowid)
        if len(results) >= k:
            break
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_textsearch.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/textsearch.py tests/userdocs/test_textsearch.py
git commit -m "feat(userdocs): add per-user FTS5 keyword search for hybrid retrieval"
```

---

### Task 4: `retrieve` — vector-only path with the relevance threshold

**Files:**
- Create: `src/userdocs/retrieve.py`
- Create: `tests/userdocs/test_retrieve.py`

**Interfaces:**
- Consumes: `UserDocsStore.{search_vectors, get_chunk}`, `userdocs.embed.embed_chunks`, `userdocs.config.{TOP_K, CANDIDATE_K, RELEVANCE_THRESHOLD}`.
- Produces: `userdocs.retrieve.RetrievedChunk` (dataclass: `text: str`, `filename: str`, `page: int | None`, `chunk_index: int`, `score: float`); `userdocs.retrieve.retrieve(store, user_id: int, query: str) -> list[RetrievedChunk]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/userdocs/test_retrieve.py
import numpy as np

from userdocs.retrieve import RetrievedChunk, retrieve
from userdocs.store import UserDocsStore


class FakeStore:
    """Stands in for UserDocsStore so this test needs no embedding model,
    no sqlite-vec, and no real database — retrieve()'s orchestration logic is
    what's under test here, not storage."""

    def __init__(self, vector_hits, chunks):
        self._vector_hits = vector_hits          # list[(chunk_id, cosine_similarity)]
        self._chunks = chunks                     # dict[chunk_id, dict]

    def search_vectors(self, user_id, query_vector, k):
        return self._vector_hits[:k]

    def get_chunk(self, chunk_id):
        return self._chunks[chunk_id]


def test_retrieve_returns_chunks_above_threshold_best_first(monkeypatch):
    import userdocs.retrieve as retrieve_module

    monkeypatch.setattr(
        retrieve_module, "embed_chunks", lambda texts: np.zeros((1, 384), dtype=np.float32)
    )
    monkeypatch.setattr(retrieve_module, "HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(retrieve_module, "RERANK_ENABLED", False)

    store = FakeStore(
        vector_hits=[(1, 0.91), (2, 0.55)],
        chunks={
            1: {"text": "25 vacation days", "chunk_index": 0, "page": 3, "filename": "policy.pdf"},
            2: {"text": "office hours", "chunk_index": 1, "page": 4, "filename": "policy.pdf"},
        },
    )

    results = retrieve(store, user_id=1, query="how many vacation days?")

    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert [r.text for r in results] == ["25 vacation days", "office hours"]
    assert results[0].filename == "policy.pdf"
    assert results[0].page == 3
    assert results[0].score == 0.91


def test_retrieve_returns_empty_when_all_scores_below_threshold(monkeypatch):
    import userdocs.retrieve as retrieve_module

    monkeypatch.setattr(
        retrieve_module, "embed_chunks", lambda texts: np.zeros((1, 384), dtype=np.float32)
    )
    monkeypatch.setattr(retrieve_module, "HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(retrieve_module, "RERANK_ENABLED", False)

    # Both chunks are technically the nearest neighbours, but neither is
    # actually relevant — this is the case the no-hallucination rule exists for.
    store = FakeStore(
        vector_hits=[(1, 0.12), (2, 0.05)],
        chunks={
            1: {"text": "unrelated text", "chunk_index": 0, "page": None, "filename": "other.txt"},
            2: {"text": "also unrelated", "chunk_index": 1, "page": None, "filename": "other.txt"},
        },
    )

    assert retrieve(store, user_id=1, query="parental leave policy") == []


def test_retrieve_returns_empty_when_user_has_no_documents(monkeypatch):
    import userdocs.retrieve as retrieve_module

    monkeypatch.setattr(
        retrieve_module, "embed_chunks", lambda texts: np.zeros((1, 384), dtype=np.float32)
    )
    monkeypatch.setattr(retrieve_module, "HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(retrieve_module, "RERANK_ENABLED", False)

    store = FakeStore(vector_hits=[], chunks={})
    assert retrieve(store, user_id=1, query="anything at all") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_retrieve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'userdocs.retrieve'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/retrieve.py
"""Retrieval orchestration. See spec/v3/SPEC.md §8.1.

The relevance threshold here is the no-hallucination rule's enforcement point
(spec §14): vector search always returns its nearest neighbours, however
irrelevant, so an empty-result check alone would never fire for a question the
documents don't answer. Filtering on cosine similarity is what makes
"I didn't find that in your documents" a real, testable behavior.
"""

from dataclasses import dataclass

from userdocs.config import CANDIDATE_K, RELEVANCE_THRESHOLD, TOP_K
from userdocs.embed import embed_chunks

HYBRID_SEARCH_ENABLED = False   # flipped on in Task 5
RERANK_ENABLED = False           # flipped on in Task 7


@dataclass
class RetrievedChunk:
    text: str
    filename: str
    page: int | None
    chunk_index: int
    score: float


def retrieve(store, user_id: int, query: str) -> list:
    query_vector = embed_chunks([query])[0]
    vector_hits = store.search_vectors(user_id, query_vector, CANDIDATE_K)

    candidates = []
    for chunk_id, score in vector_hits[:TOP_K]:
        chunk = store.get_chunk(chunk_id)
        candidates.append(
            RetrievedChunk(
                text=chunk["text"],
                filename=chunk["filename"],
                page=chunk["page"],
                chunk_index=chunk["chunk_index"],
                score=score,
            )
        )

    return [c for c in candidates if c.score >= RELEVANCE_THRESHOLD]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_retrieve.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/retrieve.py tests/userdocs/test_retrieve.py
git commit -m "feat(userdocs): add retrieve() with relevance-threshold no-hallucination gate"
```

---

### Task 5: Fuse vector + text search into `retrieve` (bonus: hybrid search)

**Files:**
- Modify: `src/userdocs/retrieve.py`
- Modify: `src/userdocs/config.py`
- Modify: `tests/userdocs/test_retrieve.py`

**Interfaces:**
- Consumes: `userdocs.fusion.rrf_fuse` (Task 1), `userdocs.textsearch.search_fts` (Task 3).
- Produces: `userdocs.config.HYBRID_SEARCH_ENABLED = True`; `retrieve()` now fuses both branches when enabled. The `RetrievedChunk.score` for a chunk that came only from the FTS branch (no vector hit) is its cosine similarity from a follow-up lookup — see the implementation note.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/userdocs/test_retrieve.py
def test_retrieve_includes_keyword_only_hits_when_hybrid_enabled(monkeypatch):
    import userdocs.retrieve as retrieve_module

    monkeypatch.setattr(
        retrieve_module, "embed_chunks", lambda texts: np.zeros((1, 384), dtype=np.float32)
    )
    monkeypatch.setattr(retrieve_module, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(retrieve_module, "RERANK_ENABLED", False)
    # chunk 3 is found ONLY by keyword search, not by vector search
    monkeypatch.setattr(retrieve_module, "search_fts", lambda store, uid, q, k: [3])

    store = FakeStore(
        vector_hits=[(1, 0.80), (3, 0.61)],
        chunks={
            1: {"text": "vector hit", "chunk_index": 0, "page": None, "filename": "a.txt"},
            3: {"text": "keyword hit", "chunk_index": 2, "page": None, "filename": "a.txt"},
        },
    )

    results = retrieve(store, user_id=1, query="some query")

    texts = [r.text for r in results]
    assert "keyword hit" in texts
    assert "vector hit" in texts


def test_retrieve_drops_fused_hits_that_have_no_similarity_score(monkeypatch):
    """A chunk surfaced only by keyword search whose cosine similarity is below
    the threshold must still be dropped — the threshold is the single gate for
    every branch, not just the vector one."""
    import userdocs.retrieve as retrieve_module

    monkeypatch.setattr(
        retrieve_module, "embed_chunks", lambda texts: np.zeros((1, 384), dtype=np.float32)
    )
    monkeypatch.setattr(retrieve_module, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(retrieve_module, "RERANK_ENABLED", False)
    monkeypatch.setattr(retrieve_module, "search_fts", lambda store, uid, q, k: [9])

    store = FakeStore(
        vector_hits=[(1, 0.80)],   # chunk 9 has no vector hit at all
        chunks={
            1: {"text": "relevant", "chunk_index": 0, "page": None, "filename": "a.txt"},
            9: {"text": "keyword only, irrelevant", "chunk_index": 8, "page": None, "filename": "a.txt"},
        },
    )

    results = retrieve(store, user_id=1, query="some query")

    assert [r.text for r in results] == ["relevant"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_retrieve.py -v`
Expected: FAIL — `AttributeError: <module 'userdocs.retrieve'> has no attribute 'search_fts'` (monkeypatch target doesn't exist yet)

- [ ] **Step 3: Write minimal implementation**

Add to `src/userdocs/config.py`:

```python
HYBRID_SEARCH_ENABLED = True
```

Replace `src/userdocs/retrieve.py`'s module body with:

```python
# src/userdocs/retrieve.py
"""Retrieval orchestration. See spec/v3/SPEC.md §8.1.

The relevance threshold here is the no-hallucination rule's enforcement point
(spec §14): vector search always returns its nearest neighbours, however
irrelevant, so an empty-result check alone would never fire for a question the
documents don't answer. Filtering on cosine similarity is what makes
"I didn't find that in your documents" a real, testable behavior.

Hybrid search (bonus) fuses the vector branch with an FTS5 keyword branch via
RRF. A chunk that only the keyword branch found has no cosine similarity from
the vector search, so it can't be threshold-checked — those are dropped rather
than admitted unscored, keeping the threshold the single gate for every branch.
"""

from dataclasses import dataclass

from userdocs.config import (
    CANDIDATE_K,
    HYBRID_SEARCH_ENABLED,
    RELEVANCE_THRESHOLD,
    TOP_K,
)
from userdocs.embed import embed_chunks
from userdocs.fusion import rrf_fuse
from userdocs.textsearch import search_fts

RERANK_ENABLED = False   # flipped on in Task 7


@dataclass
class RetrievedChunk:
    text: str
    filename: str
    page: int | None
    chunk_index: int
    score: float


def retrieve(store, user_id: int, query: str) -> list:
    query_vector = embed_chunks([query])[0]
    vector_hits = store.search_vectors(user_id, query_vector, CANDIDATE_K)
    similarity_by_chunk_id = {chunk_id: score for chunk_id, score in vector_hits}

    if HYBRID_SEARCH_ENABLED:
        keyword_hits = search_fts(store, user_id, query, CANDIDATE_K)
        ordered_ids = rrf_fuse([[cid for cid, _ in vector_hits], keyword_hits])
    else:
        ordered_ids = [cid for cid, _ in vector_hits]

    candidates = []
    for chunk_id in ordered_ids[:TOP_K]:
        score = similarity_by_chunk_id.get(chunk_id)
        if score is None:
            # Keyword-only hit: no cosine similarity available, so it can't
            # clear the relevance threshold. Dropped, not admitted unscored.
            continue
        chunk = store.get_chunk(chunk_id)
        candidates.append(
            RetrievedChunk(
                text=chunk["text"],
                filename=chunk["filename"],
                page=chunk["page"],
                chunk_index=chunk["chunk_index"],
                score=score,
            )
        )

    return [c for c in candidates if c.score >= RELEVANCE_THRESHOLD]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_retrieve.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/retrieve.py src/userdocs/config.py tests/userdocs/test_retrieve.py
git commit -m "feat(userdocs): fuse vector and keyword search results in retrieve()"
```

---

### Task 6: `rerank` — cross-encoder reranking (bonus)

**Files:**
- Create: `src/userdocs/rerank.py`
- Create: `tests/userdocs/test_rerank.py`

**Interfaces:**
- Consumes: `userdocs.config.RERANK_MODEL`, `userdocs.errors.RerankError`; `sentence_transformers.CrossEncoder`.
- Produces: `userdocs.rerank.rerank(query: str, candidates: list) -> list` — the same `RetrievedChunk` objects, reordered by cross-encoder score descending. Raises `RerankError` on model/predict failure.

- [ ] **Step 1: Write the failing test**

```python
# tests/userdocs/test_rerank.py
from userdocs.errors import RerankError
from userdocs.rerank import rerank
from userdocs.retrieve import RetrievedChunk


def _chunk(text, score):
    return RetrievedChunk(text=text, filename="a.txt", page=None, chunk_index=0, score=score)


def test_rerank_reorders_by_cross_encoder_score(monkeypatch):
    import userdocs.rerank as rerank_module

    class FakeCrossEncoder:
        def predict(self, pairs):
            # score by position of the word "answer" — the second candidate wins
            return [1.0 if "answer" in text else 0.1 for _query, text in pairs]

    monkeypatch.setattr(rerank_module, "_get_model", lambda: FakeCrossEncoder())

    candidates = [_chunk("unrelated filler", 0.9), _chunk("the actual answer", 0.5)]
    result = rerank("what is it?", candidates)

    assert [c.text for c in result] == ["the actual answer", "unrelated filler"]


def test_rerank_preserves_original_similarity_score(monkeypatch):
    """The cross-encoder only reorders. RetrievedChunk.score stays the cosine
    similarity so the relevance threshold (spec §14) keeps working against a
    calibrated value — a cross-encoder's raw output isn't comparable to it."""
    import userdocs.rerank as rerank_module

    class FakeCrossEncoder:
        def predict(self, pairs):
            return [0.99 for _ in pairs]

    monkeypatch.setattr(rerank_module, "_get_model", lambda: FakeCrossEncoder())

    candidates = [_chunk("first", 0.42)]
    result = rerank("q", candidates)

    assert result[0].score == 0.42


def test_rerank_raises_rerank_error_on_model_failure(monkeypatch):
    import userdocs.rerank as rerank_module

    class BoomCrossEncoder:
        def predict(self, pairs):
            raise RuntimeError("model exploded")

    monkeypatch.setattr(rerank_module, "_get_model", lambda: BoomCrossEncoder())

    try:
        rerank("q", [_chunk("x", 0.5)])
        assert False, "expected RerankError"
    except RerankError:
        pass


def test_rerank_empty_candidates_returns_empty_without_loading_model():
    # No monkeypatch: if this touched the model it would try a real download.
    assert rerank("q", []) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_rerank.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'userdocs.rerank'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/rerank.py
"""Cross-encoder reranking of retrieval candidates (bonus).

See spec/v3/SPEC.md §8.3. The cross-encoder reads (query, chunk) as a pair
rather than comparing two independently-computed embeddings, which makes it
more accurate for ranking — but its output is an uncalibrated relevance logit,
not a cosine similarity, so RetrievedChunk.score is deliberately left as the
original cosine value. Reranking changes the ORDER, never which candidates
cleared the relevance threshold.

Model is lazy-loaded so importing this module (or calling rerank with no
candidates) never triggers a model download.
"""

from sentence_transformers import CrossEncoder

from userdocs.config import RERANK_MODEL
from userdocs.errors import RerankError

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank(query: str, candidates: list) -> list:
    if not candidates:
        return []

    try:
        model = _get_model()
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = model.predict(pairs)
    except Exception as exc:
        raise RerankError(f"Reranking failed: {exc}") from exc

    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [candidate for candidate, _score in ranked]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_rerank.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/rerank.py tests/userdocs/test_rerank.py
git commit -m "feat(userdocs): add cross-encoder reranking of retrieval candidates"
```

---

### Task 7: Wire reranking into `retrieve` with graceful degradation

**Files:**
- Modify: `src/userdocs/retrieve.py`
- Modify: `src/userdocs/config.py`
- Modify: `tests/userdocs/test_retrieve.py`

**Interfaces:**
- Consumes: `userdocs.rerank.rerank` (Task 6), `userdocs.errors.RerankError`.
- Produces: `userdocs.config.RERANK_ENABLED = True`; `retrieve()` reranks the threshold-passing survivors and falls back to the unreranked order if `RerankError` is raised.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/userdocs/test_retrieve.py
from userdocs.errors import RerankError


def test_retrieve_applies_reranking_when_enabled(monkeypatch):
    import userdocs.retrieve as retrieve_module

    monkeypatch.setattr(
        retrieve_module, "embed_chunks", lambda texts: np.zeros((1, 384), dtype=np.float32)
    )
    monkeypatch.setattr(retrieve_module, "HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(retrieve_module, "RERANK_ENABLED", True)
    # rerank reverses the order, so a difference is observable
    monkeypatch.setattr(retrieve_module, "rerank", lambda query, candidates: list(reversed(candidates)))

    store = FakeStore(
        vector_hits=[(1, 0.90), (2, 0.80)],
        chunks={
            1: {"text": "first by vector", "chunk_index": 0, "page": None, "filename": "a.txt"},
            2: {"text": "second by vector", "chunk_index": 1, "page": None, "filename": "a.txt"},
        },
    )

    results = retrieve(store, user_id=1, query="q")

    assert [r.text for r in results] == ["second by vector", "first by vector"]


def test_retrieve_falls_back_to_unreranked_order_when_rerank_fails(monkeypatch):
    import userdocs.retrieve as retrieve_module

    monkeypatch.setattr(
        retrieve_module, "embed_chunks", lambda texts: np.zeros((1, 384), dtype=np.float32)
    )
    monkeypatch.setattr(retrieve_module, "HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(retrieve_module, "RERANK_ENABLED", True)

    def boom(query, candidates):
        raise RerankError("model unavailable")

    monkeypatch.setattr(retrieve_module, "rerank", boom)

    store = FakeStore(
        vector_hits=[(1, 0.90), (2, 0.80)],
        chunks={
            1: {"text": "first by vector", "chunk_index": 0, "page": None, "filename": "a.txt"},
            2: {"text": "second by vector", "chunk_index": 1, "page": None, "filename": "a.txt"},
        },
    )

    # A reranking failure must degrade quality, not availability.
    results = retrieve(store, user_id=1, query="q")
    assert [r.text for r in results] == ["first by vector", "second by vector"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_retrieve.py -v`
Expected: FAIL — `AttributeError: <module 'userdocs.retrieve'> has no attribute 'rerank'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/userdocs/config.py`:

```python
RERANK_ENABLED = True
```

In `src/userdocs/retrieve.py`, change the config import and remove the local flag:

```python
from userdocs.config import (
    CANDIDATE_K,
    HYBRID_SEARCH_ENABLED,
    RELEVANCE_THRESHOLD,
    RERANK_ENABLED,
    TOP_K,
)
from userdocs.embed import embed_chunks
from userdocs.errors import RerankError
from userdocs.fusion import rrf_fuse
from userdocs.rerank import rerank
from userdocs.textsearch import search_fts
```

Delete the `RERANK_ENABLED = False` module-level line, and replace `retrieve()`'s final `return` with:

```python
    survivors = [c for c in candidates if c.score >= RELEVANCE_THRESHOLD]

    if RERANK_ENABLED and survivors:
        try:
            survivors = rerank(query, survivors)
        except RerankError:
            # Reranking is a quality improvement, not a correctness
            # requirement — keep the pre-rerank order rather than losing the
            # answer entirely.
            pass

    return survivors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_retrieve.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/retrieve.py src/userdocs/config.py tests/userdocs/test_retrieve.py
git commit -m "feat(userdocs): rerank retrieval results with graceful degradation on failure"
```

---

### Task 8: `Tool` / `ToolRegistry` primitives

**Files:**
- Create: `src/userdocs/tool_registry.py`
- Create: `tests/userdocs/test_tool_registry.py`

**Interfaces:**
- Produces: `userdocs.tool_registry.Tool` (frozen dataclass: `name: str`, `description: str`, `parameters: dict`, `handler: Callable[..., Awaitable[str]]`, method `.schema() -> dict`); `userdocs.tool_registry.ToolRegistry(tools: list[Tool])` with `.schemas() -> list[dict]` and `async .invoke(call: dict) -> str`.

> Note: this is adapted from `../telegram-bot/tools/__init__.py`, which already implements exactly this shape for the previous assignment (spec §2's reuse table). The one behavioral difference: `invoke` returns `f"ERROR: {exc}"` on any handler exception so a tool failure becomes something the LLM can read and react to, rather than an exception that kills the whole turn.

- [ ] **Step 1: Write the failing test**

```python
# tests/userdocs/test_tool_registry.py
import asyncio

from userdocs.tool_registry import Tool, ToolRegistry


def _make_tool(handler):
    return Tool(
        name="echo",
        description="Echoes its input.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        handler=handler,
    )


def test_tool_schema_matches_ollama_function_format():
    tool = _make_tool(handler=None)
    assert tool.schema() == {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echoes its input.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }


def test_registry_invoke_calls_the_matching_handler():
    async def handler(text: str) -> str:
        return f"echoed: {text}"

    registry = ToolRegistry([_make_tool(handler)])
    result = asyncio.run(registry.invoke({"function": {"name": "echo", "arguments": {"text": "hi"}}}))
    assert result == "echoed: hi"


def test_registry_invoke_returns_error_string_for_unknown_tool():
    registry = ToolRegistry([])
    result = asyncio.run(registry.invoke({"function": {"name": "nope", "arguments": {}}}))
    assert result.startswith("ERROR:")
    assert "nope" in result


def test_registry_invoke_returns_error_string_instead_of_raising():
    async def handler(text: str) -> str:
        raise ValueError("handler blew up")

    registry = ToolRegistry([_make_tool(handler)])
    result = asyncio.run(registry.invoke({"function": {"name": "echo", "arguments": {"text": "hi"}}}))
    assert result.startswith("ERROR:")
    assert "handler blew up" in result


def test_registry_invoke_parses_json_string_arguments():
    """Ollama sometimes returns tool-call arguments as a JSON string rather
    than a decoded object — both forms must work."""

    async def handler(text: str) -> str:
        return f"echoed: {text}"

    registry = ToolRegistry([_make_tool(handler)])
    result = asyncio.run(
        registry.invoke({"function": {"name": "echo", "arguments": '{"text": "hi"}'}})
    )
    assert result == "echoed: hi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_tool_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'userdocs.tool_registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/tool_registry.py
"""Tool definition and dispatch for the agent loop.

Adapted from ../telegram-bot/tools/__init__.py (see spec/v3/SPEC.md §2's reuse
table). Handler exceptions become "ERROR: ..." strings the LLM can read and
react to, rather than exceptions that abort the whole turn.
"""

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Optional[Callable[..., Awaitable[str]]]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list):
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list:
        return [tool.schema() for tool in self._tools.values()]

    async def invoke(self, call: dict) -> str:
        function = call.get("function", {})
        name = function.get("name")
        tool = self._tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool {name!r}"

        arguments: Any = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return f"ERROR: could not parse arguments for {name!r}: {exc}"

        try:
            return await tool.handler(**arguments)
        except Exception as exc:
            return f"ERROR: {exc}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_tool_registry.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/tool_registry.py tests/userdocs/test_tool_registry.py
git commit -m "feat(userdocs): add Tool/ToolRegistry primitives for agent tool dispatch"
```

---

### Task 9: `search_documents` tool with source attribution

**Files:**
- Create: `src/userdocs/tools.py`
- Create: `tests/userdocs/test_tools.py`

**Interfaces:**
- Consumes: `userdocs.tool_registry.Tool` (Task 8), `userdocs.retrieve.retrieve` (Tasks 4–7).
- Produces: `userdocs.tools.NO_RESULTS_MESSAGE: str` (the literal `"No relevant information found in the user's documents."`); `userdocs.tools.build_search_documents_tool(store, user_id: int) -> Tool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/userdocs/test_tools.py
import asyncio

from userdocs.retrieve import RetrievedChunk
from userdocs.tools import NO_RESULTS_MESSAGE, build_search_documents_tool


def test_tool_schema_has_no_user_id_parameter():
    """user_id is bound in trusted code, never exposed to the LLM — otherwise a
    model could request another user's documents (spec §10, §11.1)."""
    tool = build_search_documents_tool(store=None, user_id=42)
    properties = tool.schema()["function"]["parameters"]["properties"]
    assert "query" in properties
    assert "user_id" not in properties


def test_tool_handler_formats_pdf_source_with_page(monkeypatch):
    import userdocs.tools as tools_module

    monkeypatch.setattr(
        tools_module,
        "retrieve",
        lambda store, user_id, query: [
            RetrievedChunk(
                text="Employees receive 25 vacation days.",
                filename="vacation_policy.pdf",
                page=12,
                chunk_index=37,
                score=0.88,
            )
        ],
    )

    tool = build_search_documents_tool(store=None, user_id=1)
    result = asyncio.run(tool.handler(query="how many vacation days?"))

    assert "[Source: vacation_policy.pdf, page 12]" in result
    assert "Employees receive 25 vacation days." in result


def test_tool_handler_formats_non_pdf_source_without_page(monkeypatch):
    import userdocs.tools as tools_module

    monkeypatch.setattr(
        tools_module,
        "retrieve",
        lambda store, user_id, query: [
            RetrievedChunk(
                text="Wellness stipend is $50/month.",
                filename="benefits.md",
                page=None,
                chunk_index=2,
                score=0.75,
            )
        ],
    )

    tool = build_search_documents_tool(store=None, user_id=1)
    result = asyncio.run(tool.handler(query="wellness stipend?"))

    assert "[Source: benefits.md]" in result
    assert "page" not in result


def test_tool_handler_returns_no_results_message_when_nothing_relevant(monkeypatch):
    import userdocs.tools as tools_module

    monkeypatch.setattr(tools_module, "retrieve", lambda store, user_id, query: [])

    tool = build_search_documents_tool(store=None, user_id=1)
    result = asyncio.run(tool.handler(query="parental leave?"))

    assert result == NO_RESULTS_MESSAGE


def test_tool_handler_passes_the_bound_user_id_to_retrieve(monkeypatch):
    import userdocs.tools as tools_module

    seen = {}

    def fake_retrieve(store, user_id, query):
        seen["user_id"] = user_id
        return []

    monkeypatch.setattr(tools_module, "retrieve", fake_retrieve)

    tool = build_search_documents_tool(store=None, user_id=777)
    asyncio.run(tool.handler(query="anything"))

    assert seen["user_id"] == 777
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'userdocs.tools'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/tools.py
"""The search_documents tool the agent calls. See spec/v3/SPEC.md §11.1, §13.

user_id is captured in the closure below, not declared as a tool parameter, so
the LLM has no way to search another user's documents even if it tried — the
binding happens in trusted code (spec §10's hard isolation requirement, agent
layer).

Source attribution (spec §13) is formatted here because this string is the only
thing the LLM ever sees about a retrieved chunk; the system prompt in agent.py
tells it to cite what appears in these brackets.
"""

from userdocs.retrieve import retrieve
from userdocs.tool_registry import Tool

NO_RESULTS_MESSAGE = "No relevant information found in the user's documents."

_DESCRIPTION = (
    "Search the user's uploaded documents for information relevant to a "
    "question. Use this whenever the user asks something that might be "
    "answered by a document they uploaded."
)

_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query, in the same language as the user's question.",
        }
    },
    "required": ["query"],
}


def _format_source(chunk) -> str:
    if chunk.page is not None:
        return f"[Source: {chunk.filename}, page {chunk.page}]"
    return f"[Source: {chunk.filename}]"


def build_search_documents_tool(store, user_id: int) -> Tool:
    async def _handler(query: str) -> str:
        chunks = retrieve(store, user_id, query)
        if not chunks:
            return NO_RESULTS_MESSAGE
        return "\n\n".join(f"{_format_source(c)}\n{c.text}" for c in chunks)

    return Tool(
        name="search_documents",
        description=_DESCRIPTION,
        parameters=_PARAMETERS,
        handler=_handler,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_tools.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/tools.py tests/userdocs/test_tools.py
git commit -m "feat(userdocs): add search_documents tool with source attribution"
```

---

### Task 10: The agent tool-use loop

**Files:**
- Create: `src/userdocs/agent.py`
- Create: `tests/userdocs/test_agent.py`

**Interfaces:**
- Consumes: `userdocs.tool_registry.ToolRegistry` (Task 8), `userdocs.config.AGENT_MAX_STEPS`.
- Produces: `userdocs.agent.SYSTEM_PROMPT: str`; `userdocs.agent.run(llm, registry, session, user_text: str, max_steps: int = AGENT_MAX_STEPS) -> str` (async). `llm` must have `async chat(messages: list[dict], tools: list[dict]) -> dict`; `session` must have `messages() -> list[dict]`, `append_user_message(str)`, `append_assistant_message(str)`, `append_tool_result(call: dict, result: str)` — the concrete `SessionStore` implementing this lives in the conversation-aware-RAG plan.

- [ ] **Step 1: Write the failing test**

```python
# tests/userdocs/test_agent.py
import asyncio

from userdocs.agent import SYSTEM_PROMPT, run
from userdocs.tool_registry import Tool, ToolRegistry


class FakeSession:
    def __init__(self):
        self._messages = []

    def messages(self):
        return list(self._messages)

    def append_user_message(self, text):
        self._messages.append({"role": "user", "content": text})

    def append_assistant_message(self, text):
        self._messages.append({"role": "assistant", "content": text})

    def append_tool_result(self, call, result):
        self._messages.append({"role": "tool", "content": result})


class ScriptedLLM:
    """Returns a pre-scripted sequence of assistant messages, one per chat()."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def chat(self, messages, tools):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self._responses.pop(0)


def _echo_registry(result_text="the tool result"):
    async def handler(query: str) -> str:
        return result_text

    return ToolRegistry([
        Tool(
            name="search_documents",
            description="d",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            handler=handler,
        )
    ])


def test_run_returns_content_directly_when_no_tool_calls():
    llm = ScriptedLLM([{"content": "Hello!", "tool_calls": []}])
    session = FakeSession()

    answer = asyncio.run(run(llm, _echo_registry(), session, "hi"))

    assert answer == "Hello!"
    assert len(llm.calls) == 1


def test_run_invokes_tool_then_returns_the_followup_answer():
    llm = ScriptedLLM([
        {
            "content": "",
            "tool_calls": [
                {"function": {"name": "search_documents", "arguments": {"query": "vacation"}}}
            ],
        },
        {"content": "You get 25 days. Source: policy.pdf", "tool_calls": []},
    ])
    session = FakeSession()

    answer = asyncio.run(run(llm, _echo_registry("25 days"), session, "how many vacation days?"))

    assert answer == "You get 25 days. Source: policy.pdf"
    assert len(llm.calls) == 2
    # the tool's result was fed back into the conversation before the 2nd call
    assert any(m["role"] == "tool" for m in llm.calls[1]["messages"])


def test_run_prepends_the_system_prompt_on_the_first_call():
    llm = ScriptedLLM([{"content": "ok", "tool_calls": []}])
    session = FakeSession()

    asyncio.run(run(llm, _echo_registry(), session, "hi"))

    first_message = llm.calls[0]["messages"][0]
    assert first_message["role"] == "system"
    assert first_message["content"] == SYSTEM_PROMPT


def test_run_stops_at_max_steps_and_returns_a_fallback_message():
    # The LLM never stops asking for tools; the step budget must break the loop.
    endless_tool_call = {
        "content": "",
        "tool_calls": [
            {"function": {"name": "search_documents", "arguments": {"query": "x"}}}
        ],
    }
    llm = ScriptedLLM([endless_tool_call] * 3)
    session = FakeSession()

    answer = asyncio.run(run(llm, _echo_registry(), session, "hi", max_steps=3))

    assert len(llm.calls) == 3
    assert "rephrasing" in answer.lower()


def test_system_prompt_instructs_against_answering_from_general_knowledge():
    lowered = SYSTEM_PROMPT.lower()
    assert "general knowledge" in lowered
    assert "search_documents" in lowered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'userdocs.agent'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/agent.py
"""Multi-step tool-use loop. See spec/v3/SPEC.md §11.2.

Adapted from ../telegram-bot/agent.py (spec §2's reuse table) — the loop shape
is the same; the system prompt is userdocs-specific and carries two hard
behavioral requirements from the assignment: always cite the source (§13), and
never substitute general knowledge for a document lookup (§14).

This module deliberately depends only on duck-typed llm/registry/session
objects, so it can be tested with fakes and reused by any caller (the Telegram
handler is the only real one today).
"""

from userdocs.config import AGENT_MAX_STEPS

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about documents the "
    "user has uploaded. You have access to a search_documents tool — use it "
    "whenever the question might be answered by their documents. If "
    "search_documents returns 'No relevant information found in the user's "
    "documents.', or the retrieved text doesn't actually answer the question, "
    "tell the user you did not find that information in their documents — "
    "never answer from your own general knowledge instead. Always cite the "
    "source filename (and page, if given) from the tool result in your final "
    "answer."
)

_STEP_BUDGET_EXHAUSTED = (
    "I wasn't able to finish processing that — please try rephrasing your question."
)


async def run(llm, registry, session, user_text: str, max_steps: int = AGENT_MAX_STEPS) -> str:
    for _ in range(max_steps):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + session.messages()
        assistant_message = await llm.chat(messages, tools=registry.schemas())

        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            return assistant_message.get("content", "")

        for call in tool_calls:
            result = await registry.invoke(call)
            session.append_tool_result(call, result)

    return _STEP_BUDGET_EXHAUSTED
```

> Note for the implementer: `run()` does **not** call
> `session.append_user_message(user_text)` itself — the caller does that before
> invoking `run` (see the Telegram chat handler in the Telegram-integration
> plan), so the user's message is already in `session.messages()` on the first
> iteration. The `user_text` parameter is kept in the signature because the
> conversation-aware-RAG plan's follow-up handling reads it, and because it
> matches the sibling repo's signature the loop was adapted from.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_agent.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/agent.py tests/userdocs/test_agent.py
git commit -m "feat(userdocs): add multi-step agent tool-use loop with no-hallucination prompt"
```

---

## Summary

At the end of this plan the whole retrieval side exists and is unit-tested with no live model or database required for the orchestration logic:

- `fusion.py` + `textsearch.py` + `retrieve.py`'s fusion step → **bonus +1 hybrid search**
- `rerank.py` + `retrieve.py`'s rerank step (with `RerankError` fallback) → **bonus +1 reranking**
- `tools.py`'s `_format_source` → **bonus +1 PDF page numbers** in attribution (the page data itself comes from the ingestion plan)
- `retrieve.py`'s `RELEVANCE_THRESHOLD` filter → the **no-hallucination rule** (assignment §14), tested directly by `test_retrieve_returns_empty_when_all_scores_below_threshold`
- `tools.py`'s closure-bound `user_id` + `textsearch.py`'s chunk-id narrowing → the **per-user isolation** requirement at the retrieval and agent layers, each with its own negative test
- `agent.py` → assignment §8's "the agent decides when to use the tool"

Still to come: `docs/superpowers/plans/2026-09-10-telegram-integration.md` (the bot, its commands, progress messages, and error-message mapping), `2026-09-10-conversation-aware-rag.md` (the real `SessionStore` this plan's `agent.run` duck-types against), and `2026-09-10-testing-and-evaluation.md` (end-to-end test, eval dataset, README).
