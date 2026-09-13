# Interactive Documents RAG (Telegram) — Design Spec

**Status:** Draft for review
**Date:** 2026-09-10
**Related:** `AI Agent Assignment.md` (repo root, source brief, Russian) / [[index]] in
the project's Obsidian vault (`~/Obsidian/interactive-rag-agent/`)

## 0. How to read this spec

Every place the assignment brief leaves a concrete engineering decision
unspecified is called out inline as:

> **Assumed default — confirm or override:** *value* — *one-line rationale*.

These are proposals, not blockers — the accompanying implementation plans
build against these defaults. Overriding one later only touches the specific
config constant/table column it names, not the overall architecture.

## 1. Purpose

Add a second, independent RAG subsystem to this repo: **per-user
document Q&A over a Telegram bot**. A user sends a document
(`.txt`/`.md`/`.docx`/`.pdf`) to the bot, the bot extracts/chunks/embeds it
into a **new SQLite + sqlite-vec store**, and the user can then ask
questions that the bot answers by having its LLM **agent** call a
`search_documents(query)` tool — never a giant always-on context dump — over
**only that user's own documents**, always citing a source, and refusing to
answer when nothing relevant was retrieved.

This is **additive**, not a rewrite of the existing company-knowledge-base
assistant:

- `src/rag/*` (FAISS + hybrid FTS5/RRF pipeline, §1 of `spec/v1/SPEC.md`) is
  untouched — it answers questions about the fixed company corpus in
  `DOCUMENTS_DIR`, unrelated to any Telegram user.
- `src/assistant.py`, `src/main.py`'s REPL, and `src/mcp/*` are untouched.
- `src/telemetry/*` (if merged from the `spec/v2` branch by the time this
  lands) is out of scope here — no telemetry wiring for the new subsystem in
  this spec.

The new subsystem lives entirely in two new top-level packages,
`src/userdocs/` (extraction → chunk → embed → store → retrieve → tool) and
`src/telegram_bot/` (aiogram long-polling bot, command handlers, per-user
session/history, LLM tool-use loop) — see §4 for the full file layout.

## 2. Reuse decision: what comes from `../telegram-bot`

Per the task's reuse-before-rebuild step, `../telegram-bot` (this user's
previous assignment, a working aiogram Telegram agent) was reviewed in full.
Findings and what they mean for this spec:

| Sibling module | What it does | Reuse decision |
|---|---|---|
| `agent.py` (66-line `run()` loop, `Protocol`-typed `ChatClient`/`Registry`/`Session`, native Ollama tool-calling via `/api/chat`, `for _ in range(max_steps)` loop) | Real multi-step tool-use loop | **Adapt directly.** This repo's own `src/assistant.py` only does a single-shot, non-looping tool decision (`_llm_decide_mcp_usage`) — weaker than what assignment §8 needs (the agent must *decide* to call `search_documents`, possibly mid-conversation). The sibling's loop shape is the better base; ported into `src/userdocs/agent.py` with `search_documents` as the (initially) only tool. |
| `tools/__init__.py` (`Tool` frozen dataclass + `ToolRegistry`, hand-built list, no decorators) | Tool registration pattern | **Adapt directly** — same dataclass/registry shape, new tool list. |
| `llm_client.py` (`OllamaClient` over `httpx`, native `/api/chat` tool-calling, 6 granular exception handlers → one `LLMError`) | LLM transport + error mapping | **Adapt directly**, becomes `src/telegram_bot/llm_client.py`. Its exception-handling granularity is the direct template for §9's "LLM error" and "timeout" categories. |
| `main.py` (aiogram `Dispatcher`, long polling, outer auth middleware keyed on `user_id` allowlist) | Bot bootstrap + auth | **Adapt the polling/middleware skeleton.** Auth middleware pattern (reject `from_user.id` not in an allowlist) is reused as-is if this bot needs an allowlist too — see §11's open decision. |
| Document/file upload handling | — | **Nothing to reuse — confirmed zero code exists.** The sibling has no `F.document` filter, no `bot.get_file`/`download` call anywhere; non-text messages get a generic "I can only understand text" reply. This entire path (§5) is new work. |
| `session.py` (`SessionStore` keyed by **`chat_id`**, append-only JSONL flat files, explicit non-goal: "no multi-user isolation") | Conversation history | **Not reusable as-is.** This spec's per-user isolation (§8) is a hard requirement the sibling deliberately doesn't have, and flat JSONL has no query/filter story. New store: SQLite table keyed by `user_id`, see §7. |
| `tools/exec_tool.py`, `tools/allowlist.py`, `skills/` | Shell-exec tool, progressive-disclosure skills | Not relevant to this feature — left alone. |

Net effect on scope: the tool-use loop, tool registry, and LLM client are
**ports**, not new designs (small deltas noted in the plans). Document
ingestion, storage, retrieval, and the Telegram document/command surface are
**new** — there is no prior art in either repo for the sqlite-vec storage
layer or per-user isolation.

## 3. Architecture

```text
                                   Telegram
                                      │
                    ┌─────────────────┴──────────────────┐
                    │                                     │
              document upload                       text message
                    │                                     │
                    ▼                                     ▼
         ┌─────────────────────┐               ┌───────────────────────┐
         │  ingest pipeline    │               │   agent tool-use loop │
         │  (src/userdocs/)    │               │   (src/userdocs/agent)│
         │                     │               └──────────┬────────────┘
         │ download            │                          │
         │   ↓                 │                  decides to call
         │ extract text        │                          │
         │   ↓                 │                          ▼
         │ chunk               │               search_documents(query, user_id)
         │   ↓                 │                          │
         │ embed               │                          ▼
         │   ↓                 │               ┌───────────────────────┐
         │ store (SQLite +     │◄──────────────►│ retrieve.py           │
         │  sqlite-vec)        │   reads         │ (vector + optional    │
         └─────────────────────┘   from same DB  │  text search + rerank)│
                                                 └──────────┬────────────┘
                                                            │
                                                     Top-K chunks
                                                     (+ source, page)
                                                            │
                                                            ▼
                                                       build context
                                                            │
                                                            ▼
                                                           LLM
                                                            │
                                                            ▼
                                                     answer + source
                                                            │
                                                            ▼
                                                        Telegram
```

Both branches share one SQLite database file (`userdocs.db`, gitignored,
analogous to the existing `*.db` pattern already used for `fts.db`/
`telemetry.db`) holding `documents`, `chunks`, and a `vec0` virtual table of
embeddings — see §7.

## 4. File-by-file changes

| File | Status | Purpose |
|---|---|---|
| `src/userdocs/__init__.py` | new | package marker |
| `src/userdocs/config.py` | new | new config constants (§6) |
| `src/userdocs/errors.py` | new | typed exceptions for §9's 10 error categories |
| `src/userdocs/extract.py` | new | `.txt`/`.md`/`.docx`/`.pdf` → text (+ page map for PDFs) |
| `src/userdocs/chunk.py` | new | text → chunks (token-based, reuses `tiktoken` already in `requirements.txt`) |
| `src/userdocs/embed.py` | new | chunks → embedding vectors (reuses `sentence-transformers`, already a dependency) |
| `src/userdocs/store.py` | new | SQLite + sqlite-vec schema, CRUD, per-user-filtered queries; also maintains the FTS5 index |
| `src/userdocs/textsearch.py` | new | FTS5 keyword search over the same chunks (bonus: hybrid search) |
| `src/userdocs/fusion.py` | new | `rrf_fuse()` — Reciprocal Rank Fusion of the vector and keyword branches (bonus: hybrid search) |
| `src/userdocs/rerank.py` | new | cross-encoder rerank of Top-K candidates (bonus) |
| `src/userdocs/retrieve.py` | new | orchestrates vector (+ text) search → fusion → Top-K → threshold → rerank, with source/page |
| `src/userdocs/tool_registry.py` | new | `Tool` frozen dataclass + `ToolRegistry` (adapted from `../telegram-bot/tools/__init__.py`) |
| `src/userdocs/tools.py` | new | `search_documents` `Tool` definition (schema + handler) |
| `src/userdocs/agent.py` | new | multi-step tool-use loop (ported from `../telegram-bot/agent.py`) |
| `src/userdocs/pipeline.py` | new | `ingest_document(user_id, filename, raw_bytes) -> IngestResult`, the one entrypoint the Telegram handler calls; wires extract→chunk→embed→store with progress callbacks (bonus) |
| `src/telegram_bot/__init__.py` | new | package marker |
| `src/telegram_bot/config.py` | new | bot token, allowed users, Ollama URL/model (env-var based, `python-dotenv`, mirrors sibling's `Config`) |
| `src/telegram_bot/llm_errors.py` | new | `LLMError` / `LLMTimeoutError` (§9 categories #8/#9), separate from `errors.py` to avoid a circular import |
| `src/telegram_bot/errors.py` | new | the one exception-type → user-message mapping table (§9) |
| `src/telegram_bot/llm_client.py` | new | `OllamaClient`, ported from sibling, native `/api/chat` tool-calling |
| `src/telegram_bot/session.py` | new | per-`user_id` conversation history in SQLite (bonus: conversation-aware RAG) |
| `src/telegram_bot/middleware.py` | new | allowlist auth middleware (adapted from sibling's `main.py`) |
| `src/telegram_bot/handlers/__init__.py` | new | package marker |
| `src/telegram_bot/handlers/documents.py` | new | document-upload handler: progress messages (bonus) → `pipeline.ingest_document` |
| `src/telegram_bot/handlers/commands.py` | new | `/documents`, `/delete <filename>` |
| `src/telegram_bot/handlers/chat.py` | new | plain-text handler → `userdocs.agent.run(...)` |
| `src/telegram_bot/main.py` | new | aiogram `Dispatcher`, auth middleware, polling entrypoint |
| `pytest.ini` | new | `asyncio_mode = auto` for `pytest-asyncio`, plus a registered `slow` marker for the tests that load a real embedding model |
| `.env.example` | new | documented template for the bot's env vars |
| `.gitignore` | modified | add `.env` (`*.db` is already covered repo-wide) |
| `src/config.py` | unchanged | existing company-KB constants stay put; new constants live in `userdocs/config.py`/`telegram_bot/config.py` instead of being appended here, to keep the two subsystems' config independently testable and importable without pulling in the other |
| `requirements.txt` | modified | add `sqlite-vec`, `aiogram`, `python-docx` *(already present — confirm)*, `pypdf` *(already present — confirm)*, `python-dotenv`, `httpx`, `sentence-transformers` cross-encoder extra is already covered by the base package |
| `tests/userdocs/*` | new | mirrors `src/userdocs/*`, one test file per module |
| `tests/telegram_bot/*` | new | mirrors `src/telegram_bot/*` (handlers tested with aiogram objects faked, no live Telegram/LLM) |
| `tests/fixtures/corpus/*` | new | four small real documents (one per supported format) shared by the end-to-end test and the eval runner, plus a committed generator script |
| `eval/userdocs_eval.json` | new | RAG evaluation dataset, ≥5 Q/A pairs (§14) |
| `scripts/run_userdocs_eval.py` | new | runs the eval dataset against `retrieve()`, reports hit/miss per question (bonus-adjacent, not a pytest test — same spirit as existing `src/benchmark.py`) |
| `README.md` | modified | new "User documents RAG" section per assignment §17 (architecture, chunking, embeddings, retrieval, storage, security, limitations) |

`requirements.txt` currently already has `pypdf`, `python-docx`,
`sentence-transformers`, `tiktoken`, `numpy`, and `pytest` (confirmed via the
repo-state verification pass) — the genuinely new packages are:

| Package | Needed for |
|---|---|
| `sqlite-vec` | the `vec0` virtual table (§7) |
| `aiogram` | the Telegram bot (§10) |
| `httpx` | the Ollama client's async HTTP (§11.2) |
| `python-dotenv` | env-var config |
| `pytest-asyncio` | testing the async handlers/agent loop |

No new package is needed for reranking — `sentence-transformers` already
provides `CrossEncoder` (§8.3) — or for the FTS5 keyword index, which is
stdlib `sqlite3` (§8.2). `reportlab` is used *once*, by a committed
fixture-generator script, and deliberately does **not** go in
`requirements.txt` (§17).

## 5. Document upload pipeline (assignment §1, §3, §4, §5, bonus +1 progress)

### 5.1 `src/userdocs/extract.py`

```python
SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}


class ExtractedDocument:
    """text: full concatenated text. pages: list[str] | None — per-page text
    for PDFs only (index 0 = page 1), so callers can compute which page a
    chunk's text came from. None for .txt/.md/.docx (no page concept)."""
    text: str
    pages: list[str] | None


def extract_text(filename: str, raw_bytes: bytes) -> ExtractedDocument:
    """
    Dispatch on the filename's extension (case-insensitive).

    - `.txt`/`.md`: `raw_bytes.decode("utf-8", errors="replace")`, pages=None.
    - `.docx`: `python_docx.Document(io.BytesIO(raw_bytes))`, join all
      paragraph texts with "\\n", pages=None (docx has no reliable page
      concept without a rendering pass — out of scope, see §15 Non-goals).
    - `.pdf`: `pypdf.PdfReader(io.BytesIO(raw_bytes))`, pages = [p.extract_text()
      or "" for p in reader.pages], text = "\\n".join(pages).

    Raises `UnsupportedFormatError` (§9) if the extension isn't in
    SUPPORTED_EXTENSIONS. Raises `CorruptDocumentError` (§9) if the
    underlying library raises while parsing (pypdf's `PdfReadError`,
    python-docx's `PackageNotFoundError`, or any other parse-time exception —
    caught broadly here since both libraries raise their own exception
    hierarchies and the caller only needs "this file is broken", not which
    library said so). Raises `EmptyDocumentError` (§9) if the extracted text,
    stripped, is empty.
    """
```

> **Assumed default — confirm or override:** `.docx` gets no page numbers
> (pages=None always) — python-docx exposes paragraphs, not rendered pages;
> computing real page breaks would need a layout engine, out of scope for
> this assignment. Only PDF gets page-level source attribution (§13).

### 5.2 `src/userdocs/chunk.py`

Reuses the existing repo's token-based approach (`tiktoken.get_encoding
("cl100k_base")`, already how `src/rag/chunk.py` works) rather than a new
scheme, for consistency and because it's already proven in this codebase.

```python
def chunk_text(text: str, pages: list[str] | None) -> list[dict]:
    """
    Split `text` into overlapping token windows of CHUNK_SIZE tokens with
    CHUNK_OVERLAP tokens of overlap (defaults below), sliding forward by
    (CHUNK_SIZE - CHUNK_OVERLAP) tokens each step. The final partial window is
    kept as-is (not padded, not dropped) if it has at least 1 token.

    Returns one dict per chunk:
        {"text": str, "chunk_index": int, "page": int | None}

    `chunk_index` is 0-based, in document order. `page` is set only when
    `pages` is not None: the 1-based page number containing the *start* of
    this chunk's text, computed by walking cumulative per-page token counts
    (a chunk that straddles a page boundary is attributed to the page its
    first token came from — good enough for "Source: doc.pdf, page N", not
    claimed to be exact for chunks spanning multiple pages).
    """
```

> **Assumed default — confirm or override:** `CHUNK_SIZE = 500` tokens,
> `CHUNK_OVERLAP = 75` tokens (new constants in `userdocs/config.py`, distinct
> from `src/config.py`'s existing `CHUNK_SIZE=700`/`CHUNK_OVERLAP=100`). User
> documents here (policies, handbooks) tend to have shorter self-contained
> sections than the general company KB corpus the existing constants were
> tuned for; a smaller window reduces the chance one chunk mixes two
> unrelated policy clauses (hurts precision of "cite the right source"),
> while 75 tokens (~15%) of overlap avoids splitting a sentence that carries
> the actual answer across a chunk boundary. Too small a chunk (e.g. <100
> tokens) risks losing surrounding context the LLM needs to phrase a correct
> answer; too large (e.g. >1500 tokens) dilutes the embedding's specificity
> and makes Top-K less precise — this tradeoff belongs in the README per
> assignment §4.

### 5.3 `src/userdocs/embed.py`

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # reuse the model already loaded by src/rag/embed.py
EMBEDDING_DIM = 384                      # this model's known output size — sqlite-vec needs a fixed dim at table-create time

def embed_chunks(chunk_texts: list[str]) -> np.ndarray:
    """
    Encode `chunk_texts` with a module-level SentenceTransformer(EMBEDDING_MODEL)
    (lazy-loaded on first call, not at import time — see §15, avoids paying the
    ~model load cost for callers that only need extract/chunk, e.g. tests),
    L2-normalize each row (`sentence_transformers`' `.encode(..., normalize_embeddings=True)`),
    return shape (len(chunk_texts), EMBEDDING_DIM) float32 array.

    Raises `EmbeddingError` (§9) if the model fails to load or `.encode()`
    raises for any reason (out-of-memory, corrupted model cache, etc.) —
    wrapped so callers get one exception type regardless of the underlying
    library's failure mode.
    """
```

> **Assumed default — confirm or override:** keep `all-MiniLM-L6-v2` (already
> a dependency, runs locally with no API key/cost, 384-dim, already proven in
> this repo's FAISS pipeline) rather than introducing a second embedding
> model just for user documents. Vectors are **L2-normalized before storage**
> so sqlite-vec's L2 distance is monotonic with cosine similarity (`L2² =
> 2 − 2·cos_sim` for unit vectors) — this lets the whole pipeline use plain
> L2 KNN without depending on a specific sqlite-vec version's cosine-metric
> option (see §7).

### 5.4 `src/userdocs/pipeline.py` — the upload entrypoint

```python
@dataclass
class IngestResult:
    document_id: int
    filename: str
    chunk_count: int


def ingest_document(
    store: "UserDocsStore",
    user_id: int,
    filename: str,
    raw_bytes: bytes,
    on_progress: Callable[[str], Awaitable[None]] | None = None,
) -> IngestResult:
    """
    Full pipeline for one uploaded document, called once per Telegram
    document message:

      1. len(raw_bytes) > MAX_DOCUMENT_BYTES → raise DocumentTooLargeError (§9)
         before doing any parsing work.
      2. await on_progress("⏳ Extracting text...") if on_progress is not None
      3. extracted = extract_text(filename, raw_bytes)   # §5.1, may raise
         UnsupportedFormatError / CorruptDocumentError / EmptyDocumentError
      4. await on_progress("✅ Text extracted")
      5. await on_progress("⏳ Creating chunks...")
      6. chunks = chunk_text(extracted.text, extracted.pages)   # §5.2
      7. await on_progress(f"✅ {len(chunks)} chunks created")
      8. await on_progress("⏳ Generating embeddings...")
      9. vectors = embed_chunks([c["text"] for c in chunks])   # §5.3, may raise EmbeddingError
     10. await on_progress("✅ Embeddings generated")
     11. document_id = store.insert_document(user_id, filename, file_type=Path(filename).suffix)
     12. store.insert_chunks_with_vectors(document_id, chunks, vectors)   # §7, may raise SQLiteStoreError
     13. return IngestResult(document_id, filename, len(chunks))

    `on_progress` is `None` in tests and in any non-Telegram caller (e.g. a
    future CLI) — every `await on_progress(...)` call is skipped entirely
    when it's None, so this function has no Telegram dependency itself; the
    Telegram handler (§10.1) supplies a closure that edits/sends a message.
    Any raised error here is caught by the *caller* (§10.1), never inside
    `ingest_document` — this function's job is only to do the work and raise
    a specific, already-caught-and-typed exception; it does not know about
    Telegram at all.
    """
```

This directly satisfies bonus **+1 Progress** (assignment §19): the six
`on_progress` calls above map 1:1 onto the mockup's six progress lines.

## 6. New config (`src/userdocs/config.py`)

```python
CHUNK_SIZE = 500                 # tokens, see §5.2
CHUNK_OVERLAP = 75               # tokens, see §5.2
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_K = 5                        # see §8's rationale
CANDIDATE_K = TOP_K * 3          # pre-rerank candidate pool, mirrors src/config.py's existing CANDIDATE_K pattern
RRF_K = 60                       # mirrors src/rag/fusion.py's existing constant, for hybrid search bonus
RELEVANCE_THRESHOLD = 0.30       # cosine similarity; below this, treat as "not found" — see §14
MAX_DOCUMENT_BYTES = 20 * 1024 * 1024   # 20 MB, see §9
USERDOCS_DB_PATH = "userdocs.db"        # relative to src/, gitignored like fts.db/telemetry.db
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # bonus reranking, see §8.3
CONVERSATION_HISTORY_TURNS = 3          # bonus conversation-aware RAG, see §12
AGENT_MAX_STEPS = 4                     # tool-use loop budget, mirrors sibling's AGENT_MAX_STEPS
```

> **Assumed default — confirm or override:** `TOP_K = 5` — matches the
> existing company-KB pipeline's default (`src/config.py::TOP_K`), a
> reasonable starting point with no evidence yet that user documents need a
> different value; easy to tune later since it's a single constant consumed
> only by `retrieve.py`.
> **Assumed default — confirm or override:** `MAX_DOCUMENT_BYTES = 20MB` —
> generous enough for a multi-hundred-page PDF policy document, small enough
> to bound embedding-time memory/latency on a laptop-class Ollama+
> SentenceTransformers setup with no GPU assumed.

## 7. Storage: SQLite + sqlite-vec (assignment §6, §10)

New file `src/userdocs/store.py`, one new database `userdocs.db`.

```sql
CREATE TABLE documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    filename    TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    created_at  TEXT NOT NULL      -- ISO 8601, UTC
);
CREATE INDEX idx_documents_user_id ON documents(user_id);

CREATE TABLE chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    text          TEXT NOT NULL,
    page          INTEGER          -- NULL unless the source document is a PDF
);
CREATE INDEX idx_chunks_document_id ON chunks(document_id);

-- sqlite-vec virtual table: one row per chunk, same rowid as chunks.id so a
-- vector search result's rowid joins straight back to `chunks` with no
-- separate mapping table.
CREATE VIRTUAL TABLE chunk_vectors USING vec0(
    embedding FLOAT[384]
);
```

`PRAGMA foreign_keys = ON;` is set on every connection (SQLite defaults it
off) so `ON DELETE CASCADE` on `chunks.document_id` actually fires —
this is how `/delete` (§11.2) removes chunks without a manual second query.
`chunk_vectors` has no foreign key of its own (sqlite-vec `vec0` tables don't
support them); deleting from `chunk_vectors` is done explicitly by
`chunk_id` alongside the `chunks` delete (see `delete_document` below) —
**this is the one place the cascade doesn't reach, and it is covered by a
dedicated test** (§16, user-isolation delete test) so it can't silently
regress into orphaned vectors.

```python
class UserDocsStore:
    """Owns one sqlite3 connection to USERDOCS_DB_PATH, opened with the
    sqlite-vec extension loaded (`conn.enable_load_extension(True)` +
    `sqlite_vec.load(conn)`, per sqlite-vec's documented Python usage) and
    `PRAGMA foreign_keys = ON`. Not thread-safe by itself — see §15 for the
    same "one connection per call from a thread pool" note as the existing
    src/rag/fts_index.py, since retrieve.py's vector+text branches also run
    concurrently (bonus hybrid search)."""

    def insert_document(self, user_id: int, filename: str, file_type: str) -> int:
        """INSERT INTO documents(...) VALUES (...); returns the new id."""

    def insert_chunks_with_vectors(self, document_id: int, chunks: list[dict], vectors: np.ndarray) -> None:
        """
        One transaction: INSERT INTO chunks(...) for each chunk (capturing
        each new chunks.id), then INSERT INTO chunk_vectors(rowid, embedding)
        VALUES (?, ?) using that same id as rowid for each corresponding
        vector row. len(chunks) must equal vectors.shape[0] — raises
        ValueError otherwise (a caller bug, not a runtime error category).
        """

    def search_vectors(self, user_id: int, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """
        Per-user vector KNN. Because sqlite-vec's vec0 MATCH clause has no
        built-in join to `documents`/`chunks` for a WHERE filter, this runs
        as two steps in one call:
          1. `document_ids = [row[0] for row in self._conn.execute(
                 "SELECT id FROM documents WHERE user_id = ?", (user_id,))]`
             — if empty, return [] immediately (§8's user-isolation guard;
             also the natural "no documents yet" case).
          2. `chunk_ids = [row[0] for row in self._conn.execute(
                 "SELECT id FROM chunks WHERE document_id IN ({placeholders})",
                 document_ids)]` — the candidate universe for this user.
          3. Run the sqlite-vec KNN query for the full candidate pool (sqlite-vec
             has no native "restrict to this rowid set" filter as of the
             pinned version, so retrieval over-fetches: `k' = max(k, len(chunk_ids))`
             is capped at a hard ceiling — see the Non-goals note below — and
             results are filtered in Python to `id in chunk_ids` after the
             KNN call, keeping only the first `k` that survive the filter):

                 SELECT rowid, distance FROM chunk_vectors
                 WHERE embedding MATCH ? AND k = ?
                 ORDER BY distance

          4. Returns up to `k` `(chunk_id, cosine_similarity)` pairs, best
             first, where `cosine_similarity = 1 - (distance ** 2) / 2`
             (§5.3's L2-normalized-vector identity) — every caller works in
             cosine similarity, not raw L2 distance, so §14's threshold is
             one number everywhere.

        This is the one place per-user isolation is enforced at the storage
        layer (§8) — `search_vectors` NEVER runs an unfiltered
        `SELECT ... FROM chunk_vectors` for all users; it always narrows to
        this user's chunk id set first.
        """

    def get_chunk(self, chunk_id: int) -> dict:
        """SELECT c.text, c.chunk_index, c.page, d.filename
           FROM chunks c JOIN documents d ON d.id = c.document_id
           WHERE c.id = ?
           — used by retrieve.py to turn ids back into displayable chunks."""

    def list_documents(self, user_id: int) -> list[dict]:
        """SELECT id, filename, file_type, created_at FROM documents
           WHERE user_id = ? ORDER BY created_at ASC."""

    def delete_document(self, user_id: int, filename: str) -> bool:
        """
        Delete one document AND its chunks AND its vectors, scoped to this
        user_id (never deletes another user's same-named file):

          1. row = SELECT id FROM documents WHERE user_id = ? AND filename = ?
             — if no row, return False (caller reports "not found").
          2. chunk_ids = SELECT id FROM chunks WHERE document_id = ?
          3. DELETE FROM chunk_vectors WHERE rowid IN (chunk_ids)   -- not reached by the FK cascade, see above
          4. DELETE FROM documents WHERE id = ?   -- cascades to `chunks` via ON DELETE CASCADE
          5. commit; return True
        """

    def close(self) -> None: ...
```

Every method that touches user data takes `user_id` explicitly and every SQL
statement that reads `documents`/`chunks` filters on it (directly, or via a
`document_id`/`chunk_id` set already derived from a `user_id`-filtered
query) — this is the concrete code-level answer to assignment §10's
"демонстрировать в коде" requirement, and §16 lists the exact test that
proves it (`test_user_isolation.py`).

> **Assumed default — confirm or override:** one shared `userdocs.db` file
> with `user_id`-filtered queries, not one SQLite file per user. Simpler
> operationally (one file to back up/inspect/delete-for-GDPR-style-requests
> would need per-user filtering either way at the app layer once files pile
> up), and isolation is enforced by the query layer above regardless — the
> assignment's own diagram (`user_id ↓ documents ↓ chunks ↓ vector search`)
> matches a single-DB, filtered-query design, not a sharded one.

## 8. Retrieval (assignment §7)

### 8.1 `src/userdocs/retrieve.py`

```python
@dataclass
class RetrievedChunk:
    text: str
    filename: str
    page: int | None
    chunk_index: int
    score: float          # cosine similarity, post-rerank if reranking ran


def retrieve(store: UserDocsStore, user_id: int, query: str) -> list[RetrievedChunk]:
    """
    1. vector_hits = store.search_vectors(user_id, embed_chunks([query])[0], CANDIDATE_K)
       — a list of (chunk_id, cosine_similarity). Keep a
       {chunk_id: cosine_similarity} map: this is the only calibrated score in
       the pipeline and step 4 needs it.
    2. If HYBRID_SEARCH_ENABLED (bonus, §8.2): ordered_ids = rrf_fuse([
           [cid for cid, _ in vector_hits], search_fts(store, user_id, query, CANDIDATE_K)
       ]) — same RRF shape as src/rag/fusion.py; else ordered_ids is just the
       vector hits' ids in order.
    3. For each of ordered_ids[:TOP_K], look up its cosine similarity from
       step 1's map and build a RetrievedChunk from store.get_chunk(cid). A
       keyword-only hit has no entry in that map, so it has no threshold-able
       score and is SKIPPED rather than admitted unscored — hybrid search
       widens the candidate ordering, it does not create an unchecked path
       past the relevance gate.
    4. Drop any candidate whose score < RELEVANCE_THRESHOLD (§14's
       no-hallucination gate). This happens AFTER Top-K selection but BEFORE
       reranking — see §8.3's note on why the order matters. A
       genuinely-empty-after-threshold result is distinguishable from "there
       were no candidates at all" only in logging, not in caller-visible
       behavior — both mean "return []".
    5. If RERANK_ENABLED (bonus, §8.3) and any survivors remain:
       survivors = rerank(query, survivors), catching RerankError and keeping
       the pre-rerank order on failure (reranking must degrade retrieval
       quality, never availability).
    6. Return the survivors.
    """
```

### 8.2 `src/userdocs/textsearch.py` (bonus +1 hybrid search)

The FTS5 index itself is created and maintained by `store.py`, not here — a
`chunks_fts(text, tokenize='porter unicode61')` virtual table in the *same*
`userdocs.db` connection (not a separate file like `src/rag`'s `fts.db`), with
`rowid = chunks.id`. `UserDocsStore.insert_chunks_with_vectors` inserts into it
in the same transaction as the chunk rows, and `delete_document` deletes from
it alongside `chunk_vectors`, so the keyword index can never drift out of sync
with the chunks it indexes.

```python
def search_fts(store: UserDocsStore, user_id: int, query: str, k: int) -> list[int]:
    """Same query-sanitization approach as src/rag/fts_index.py::search_fts
    (split on whitespace, double-quote each token, OR them, bm25() ranking)
    — but additionally filtered to this user_id's chunk ids first (same
    two-step pattern as store.search_vectors, §7), since FTS5 has no
    per-row ACL of its own either.

    Returns chunk ids only, best BM25 match first — no score. BM25 scores
    aren't comparable to cosine similarities, and RRF (§8.1 step 2) consumes
    rank order rather than score, so carrying a BM25 number forward would
    only invite someone to compare it against RELEVANCE_THRESHOLD by mistake.

    Returns [] (never raises) on a malformed query, exactly like the existing
    implementation."""
```

> **Assumed default — confirm or override:** hybrid search is **on by
> default** (`HYBRID_SEARCH_ENABLED = True` in `userdocs/config.py`) since
> it's a Bonus item worth building properly rather than gating behind a flag
> nobody flips — same architectural precedent as the existing `src/rag/`
> pipeline, which has no "vector-only" toggle either.

### 8.3 `src/userdocs/rerank.py` (bonus +1 reranking)

```python
def rerank(query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Cross-encoder rerank of `candidates` (§8.1's RetrievedChunk objects,
    whose `.score` is the pre-rerank cosine similarity).

    Lazily loads a module-level `CrossEncoder(RERANK_MODEL)`
    (sentence-transformers' cross-encoder API, same library already a
    dependency for embeddings — no new package) on first call; returns
    immediately for an empty candidate list so nothing is loaded needlessly.
    Scores every (query, candidate.text) pair with `.predict()` and returns
    the candidates sorted descending by that score.

    `.score` is deliberately LEFT AS the original cosine similarity rather
    than overwritten with the cross-encoder's output: the cross-encoder emits
    an uncalibrated relevance logit, not a 0–1 similarity, so overwriting
    would silently break any later comparison against RELEVANCE_THRESHOLD.
    Reranking changes the ORDER only. (Blending the two scores was
    considered and rejected — it would need its own calibration work this
    assignment doesn't call for.)

    Raises RerankError (§9) if the model fails to load or `.predict()`
    raises — callers (retrieve.py) catch this and fall back to the
    pre-rerank order rather than losing the answer entirely (reranking
    failing should degrade retrieval quality, not availability).
    """
```

> **Assumed default — confirm or override:**
> `cross-encoder/ms-marco-MiniLM-L-6-v2` — a small (~34M param), CPU-friendly,
> widely-used passage-reranking cross-encoder that runs locally with no API
> key, consistent with this repo's local-first design (same reasoning as the
> embedding model choice).
>
> **Note on §14's threshold interacting with reranking:** since a
> cross-encoder's raw output is not a calibrated 0–1 cosine similarity, but
> §14's `RELEVANCE_THRESHOLD` is defined in cosine-similarity terms, the
> threshold check in `retrieve()` step 5 above is applied to the
> **pre-rerank** cosine score, and reranking only reorders (never
> re-admits) candidates that already passed the threshold — recorded here
> explicitly since it's exactly the kind of easy-to-miss ordering-of-steps
> bug the assignment's error-handling emphasis calls out.

## 9. Error handling (assignment §12 — exact category → mechanism)

New file `src/userdocs/errors.py` defines one exception per category so
every call site catches a specific type, never a bare `except Exception` (a
deliberate contrast with the sibling's `ToolRegistry.invoke`'s broad catch —
here, each layer maps *its own* exception, and only the Telegram handler at
the very top does one final broad catch as a last-resort safety net, §10.1).

| # | Category (assignment wording) | Exception | Raised by | User-facing message |
|---|---|---|---|---|
| 1 | Неподдерживаемый формат файла | `UnsupportedFormatError` | `extract.py` | "❌ Unsupported file format. Please send .txt, .md, .docx, or .pdf." |
| 2 | Повреждённый PDF | `CorruptDocumentError` | `extract.py` (pypdf) | "❌ Couldn't process the document. Please make sure the file isn't corrupted." |
| 3 | Повреждённый DOCX | `CorruptDocumentError` (same type) | `extract.py` (python-docx) | same message as #2 |
| 4 | Пустой документ | `EmptyDocumentError` | `extract.py` | "❌ This document appears to be empty — nothing to index." |
| 5 | Слишком большой документ | `DocumentTooLargeError` | `pipeline.py` (before parsing) | "❌ This file is too large (max 20 MB). Please split it or send a smaller file." |
| 6 | Ошибка embedding model | `EmbeddingError` | `embed.py` | "❌ Something went wrong while indexing your document. Please try again." |
| 7 | Ошибка SQLite | `SQLiteStoreError` | `store.py` (wraps `sqlite3.Error`) | "❌ Something went wrong while saving your document. Please try again." |
| 8 | Ошибка LLM | `LLMError` (ported from sibling) | `telegram_bot/llm_client.py` | "❌ The assistant is temporarily unavailable. Please try again shortly." |
| 9 | Timeout | `LLMTimeoutError` (subclass of `LLMError`) | `telegram_bot/llm_client.py` (httpx `ReadTimeout`/`ConnectTimeout`) | "⏳ That took too long — please try again." |
| 10 | Ошибка Telegram API | caught at the aiogram handler level (`aiogram.exceptions.TelegramAPIError` and subclasses) | `handlers/*.py` | logged only — no reply is sent (there is no channel left to send it on if the Telegram call itself failed); mirrors the sibling's two-tier bulk→per-message degradation pattern (`main.py:224-233`) for the specific case of `/delete` needing to edit/send a confirmation |

Every category has:
1. A specific exception type (never a string check on `str(exc)`).
2. Exactly one place it's caught — the Telegram handler that called into the
   pipeline/agent (`handlers/documents.py` for #1–7, `handlers/chat.py` for
   #8–9, every handler's outermost `try` for #10) — and turned into the
   fixed user message from the table, never a raw stack trace.
3. A test asserting the exception → message mapping (§16).

> **Assumed default — confirm or override:** `DocumentTooLargeError`'s limit
> is `MAX_DOCUMENT_BYTES = 20MB` (§6) checked on raw upload bytes, before
> extraction — cheapest possible point to fail fast, and matches Telegram
> Bot API's own 20MB download limit for bots without a local Bot API server,
> so this cap is also a practical ceiling rather than an arbitrary one.

## 10. Telegram integration (assignment §1, §11, §19 bonus progress)

### 10.1 `src/telegram_bot/handlers/documents.py`

```python
async def handle_document(message: Message, store: UserDocsStore) -> None:
    """
    Registered on `F.document` (aiogram filter). Flow:

      1. await message.answer("📄 Документ получен.\\n\\nНачинаю обработку...")
         — the assignment's exact scripted line (§1), sent immediately so
         the user isn't staring at silence during download.
      2. file = await message.bot.get_file(message.document.file_id)
         raw_bytes = await message.bot.download_file(file.file_path)  # -> BytesIO
      3. progress_msg = await message.answer("⏳ Extracting text...")
         on_progress = lambda text: progress_msg.edit_text(text)   # edits IN PLACE, not one message per step — keeps the chat from being spammed with 6 separate messages for one upload
      4. try: result = pipeline.ingest_document(store, message.from_user.id,
             message.document.file_name, raw_bytes.read(), on_progress)
         except (UnsupportedFormatError, CorruptDocumentError, EmptyDocumentError,
                 DocumentTooLargeError, EmbeddingError, SQLiteStoreError) as exc:
             await message.answer(ERROR_MESSAGES[type(exc)])   # §9's table
             return
      5. await message.answer("✅ Документ готов.\\n\\nТеперь вы можете задавать вопросы по документу.")
         — the assignment's exact scripted line (§1).

    Any `TelegramAPIError` raised by the `message.answer`/`edit_text` calls
    themselves propagates to aiogram's own error middleware (there is no
    more specific action to take — the chat channel is the thing that
    failed); this is category #10 in §9's table.
    """
```

### 10.2 `src/telegram_bot/handlers/commands.py`

```python
async def handle_documents_command(message: Message, store: UserDocsStore) -> None:
    """/documents — list this user's documents.

    docs = store.list_documents(message.from_user.id)
    If empty: "📚 You haven't uploaded any documents yet."
    Else, numbered list matching the assignment's exact mockup:

        📚 Your documents:

        1. employee_handbook.pdf
        2. vacation_policy.pdf
        3. benefits.md
    """


async def handle_delete_command(message: Message, store: UserDocsStore) -> None:
    """/delete <filename> — e.g. "/delete vacation_policy.pdf".

    filename = message.text.removeprefix("/delete").strip()
    If filename == "": "Usage: /delete <filename>" (a plain usage error, not
    one of §9's categories — this is a malformed *command*, not a document
    processing failure).
    deleted = store.delete_document(message.from_user.id, filename)
    "✅ Deleted {filename}." if deleted else "❌ No document named {filename} found."
    """
```

### 10.3 `src/telegram_bot/handlers/chat.py`

```python
async def handle_text(message: Message, store: UserDocsStore, llm: OllamaClient,
                       sessions: SessionStore, registry: ToolRegistry) -> None:
    """
    Plain-text message → the agent tool-use loop (§11).

      session = sessions.get_or_create(message.from_user.id)
      session.append_user_message(message.text)
      try:
          answer = await userdocs.agent.run(llm, registry, session,
                                             message.text, max_steps=AGENT_MAX_STEPS)
      except LLMError as exc:
          await message.answer(ERROR_MESSAGES[type(exc)])   # §9 #8/#9
          return
      session.append_assistant_message(answer)
      await message.answer(answer)
    """
```

## 11. Agent tool-use wiring (assignment §8, §9)

### 11.1 `src/userdocs/tools.py`

```python
def build_search_documents_tool(store: UserDocsStore, user_id: int) -> Tool:
    """
    Returns a Tool (the same frozen dataclass shape as
    ../telegram-bot/tools/__init__.py::Tool) bound to this specific user_id —
    a fresh Tool/ToolRegistry pair is built per incoming message (§11.2),
    not shared across users, so the LLM's tool schema never needs a user_id
    parameter of its own (and can't be tricked into passing a different
    one — the binding happens in trusted code, not via an LLM-controlled
    argument, which is the actual enforcement point for §10's isolation
    requirement at the agent layer, complementing store.py's own filtering).

    name: "search_documents"
    description: "Search the user's uploaded documents for information
        relevant to a question. Use this whenever the user asks something
        that might be answered by a document they uploaded."
    parameters: {"type": "object", "properties": {"query": {"type": "string",
        "description": "The search query, in the same language as the user's question."}},
        "required": ["query"]}
    handler: async def _handler(query: str) -> str:
        chunks = retrieve(store, user_id, query)   # §8.1
        if not chunks:
            return "No relevant information found in the user's documents."
        return "\\n\\n".join(
            f"[Source: {c.filename}{f', page {c.page}' if c.page else ''}]\\n{c.text}"
            for c in chunks
        )
    """
```

### 11.2 `src/userdocs/agent.py`

```python
async def run(llm: ChatClient, registry: Registry, session: Session,
               user_text: str, max_steps: int = 4) -> str:
    """
    Ported from ../telegram-bot/agent.py with one behavioral difference: the
    system prompt (prepended to session.messages() before the first LLM
    call) is userdocs-specific:

        "You are a helpful assistant that answers questions about documents
        the user has uploaded. You have access to a search_documents tool —
        use it whenever the question might be answered by their documents.
        If search_documents returns 'No relevant information found in the
        user's documents.', or the retrieved text doesn't actually answer
        the question, tell the user you don't have that information in
        their documents — never answer from general knowledge instead.
        Always cite the source filename (and page, if given) from the tool
        result in your final answer."

    Loop shape (unchanged from sibling):
        for _ in range(max_steps):
            assistant_msg = await llm.chat(session.messages(), tools=registry.schemas())
            if not assistant_msg.get("tool_calls"):
                return assistant_msg["content"]
            for call in assistant_msg["tool_calls"]:
                result = await registry.invoke(call)
                session.append_tool_result(call, result)
        return "I wasn't able to finish processing that — please try rephrasing your question."
    """
```

This is the concrete mechanism behind assignment §8's requirement that the
agent "самостоятельно определить, когда ему необходимо воспользоваться этим
инструментом" — the LLM sees `search_documents` in its tool schema on every
turn and decides whether to call it; the system prompt above is what steers
it toward calling it for document-shaped questions and away from ever
answering from parametric knowledge (§14).

> **Assumed default — confirm or override:** switch the LLM call for this
> subsystem to Ollama's **native tool-calling** (`/api/chat` with a `tools`
> array), via a ported `OllamaClient`, rather than reusing
> `src/rag/query.py::ask_llm`'s raw `/api/generate` call or
> `src/assistant.py`'s hand-written JSON-blob tool-decision prompt. Neither
> existing pattern in this repo actually loops or uses the model's native
> tool-calling format; the sibling's `agent.py`/`llm_client.py` already do,
> and assignment §8 specifically wants the *model* deciding, not a
> hand-parsed JSON convention.
> **Assumed default — confirm or override:** default model
> `qwen2.5:7b` (env-overridable, mirrors the sibling's `OLLAMA_MODEL` default)
> rather than this repo's existing `qwen3:0.6b` — a 0.6B model is a much
> weaker bet for reliably emitting well-formed tool calls turn after turn;
> the sibling's own working tool-loop was built and tested against 7b. This
> only affects the new `telegram_bot`/`userdocs` subsystem's config — the
> existing `src/rag/query.py::OLLAMA_MODEL` for the company-KB assistant is
> untouched.

## 12. Conversation-aware RAG (bonus +2)

`src/telegram_bot/session.py`:

```python
class SessionStore:
    """SQLite-backed, keyed by user_id (see §2's reuse-decision table — this
    is the one place the sibling's chat_id-keyed design is explicitly NOT
    reused). Table:

        CREATE TABLE conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,          -- "user" | "assistant" | "tool"
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

    Only conversational turns (role "user"/"assistant") are PERSISTED. The
    tool-call and tool-result messages of the turn currently in flight are
    held in memory on the Session and discarded when the next
    append_user_message() arrives — the agent loop needs them within a turn
    (step 2's LLM call must see step 1's tool output), but across turns
    they're per-call scaffolding that would balloon every subsequent prompt
    by a full Top-K of chunk text.

    get_or_create(user_id).messages() therefore returns:
        [last CONVERSATION_HISTORY_TURNS*2 persisted messages, oldest first]
      + [this turn's in-memory tool messages]
    which is exactly the `messages` list llm.chat() needs. The persisted half
    is bounded, so prompt size doesn't grow without limit however long the
    conversation runs.

    append_tool_result(call, result) appends an
    {"role": "assistant", "content": "", "tool_calls": [call]} message
    immediately before each {"role": "tool", "content": result} message.
    Ollama's /api/chat (following the OpenAI-shaped convention) requires a
    tool message to answer a preceding assistant message carrying the
    matching tool_calls; a bare tool message is an invalid sequence some
    models reject and others silently mishandle. One assistant/tool pair per
    call keeps the sequence valid for any number of calls in a turn.
    """
```

The follow-up example from the brief ("Сколько дней отпуска... / А можно
перенести их на следующий год?") works because the second question, plus the
prior Q&A pair, are all present in `session.messages()` when the LLM decides
whether/how to call `search_documents` — the LLM itself resolves "их" against
the visible history; no separate query-rewriting step is introduced (kept
consistent with §11's "the model decides" philosophy — a rewritten-query
approach was considered and rejected: it would duplicate logic the LLM
already does for free once history is in context).

> **Assumed default — confirm or override:** `CONVERSATION_HISTORY_TURNS = 3`
> (6 messages) — enough for the brief's own example (a 1-turn follow-up)
> with headroom for a couple more, without unboundedly growing every prompt
> as a conversation goes on.

## 13. Source attribution (assignment §13, bonus +1 page numbers)

Enforced at exactly one place — the `search_documents` tool handler
(§11.1) — since that's the only text the LLM ever sees about retrieved
chunks, and the system prompt (§11.2) instructs it to always cite what's in
that text. Format embedded in the tool result:

```text
[Source: vacation_policy.pdf, page 12]
<chunk text>
```

for a PDF chunk with a known page, or

```text
[Source: benefits.md]
<chunk text>
```

for a non-PDF chunk (`page` is `None`, §5.1). This is a minimum floor, not a
literal string the final answer must reproduce — the LLM composes the
final answer's own "Источник: ..." line from this bracketed hint, per the
system prompt's instruction. **Tested directly** (§16): a unit test on the
tool handler asserts the bracketed source line's exact format for both a
PDF-with-page and a non-PDF chunk, independent of what any LLM does with it.

## 14. No-hallucination rule (assignment §14) — exact trigger condition

> **Assumed default — confirm or override:** the trigger is **"no chunk
> survives `RELEVANCE_THRESHOLD = 0.30` cosine similarity"** (§8.1 step 5),
> not merely "retrieval returned zero rows." Rationale: an empty-database
> case (no documents uploaded yet) already naturally produces zero
> candidates before any threshold is applied, so a pure "empty result"
> check would work for that case alone — but the harder case the assignment
> is actually testing for (§14's own framing: "если релевантной информации
> в документах нет") is a *low-relevance but non-empty* result, e.g. a user
> asks about "parental leave" and the only uploaded document is an IT
> security policy — vector search will always return its Top-K nearest
> chunks (cosine similarity is never undefined, just low), so "zero rows"
> would never fire and the LLM would be handed irrelevant chunks it could
> still weave into a plausible-sounding but fabricated answer. A fixed
> similarity floor is what actually blocks that. `0.30` is a starting value
> chosen conservatively (all-MiniLM-L6-v2 unrelated-text pairs typically
> score well below this, closely-related paraphrases well above it) —
> tunable without any code change once real eval data (§16.5/§17) shows it's
> too strict or too lenient.

When triggered, `search_documents`'s handler (§11.1) returns the literal
string `"No relevant information found in the user's documents."`, and the
system prompt (§11.2) instructs the LLM to turn that into the assignment's
exact scripted phrase for the user:

```text
Я не нашёл этой информации в загруженных документах.
```

## 15. Non-goals

- Real per-page rendering/layout for `.docx` page numbers — not attempted
  (§5.1); only PDFs get page attribution.
- sqlite-vec native per-rowid-subset filtering for the vector KNN query —
  the pinned version is treated as not supporting it (§7's `search_vectors`
  over-fetches and filters in Python); if a newer sqlite-vec version adds
  this, `search_vectors` can be simplified, but that's a future
  optimization, not required here.
- No migration of the existing `src/rag/*` FAISS pipeline onto sqlite-vec —
  the two subsystems intentionally coexist (§1).
- No telemetry/cost tracking wired into the new subsystem (would duplicate
  `spec/v2`'s scope; out of scope here even if that spec has since merged).
- No multi-process concurrency story beyond SQLite's own file locking — same
  scope limitation the existing `spec/v2` design accepted for
  `telemetry.db`.
- No admin/moderation tooling (e.g. a global document count cap across all
  users, storage quota enforcement) — only the per-document `MAX_DOCUMENT_
  BYTES` cap (§9 #5) is in scope.
- No retry/backoff policy for LLM timeouts (§9 #9) beyond surfacing the
  error to the user and letting them resend their message — no automatic
  retry loop.

## 16. Testing requirements (assignment §15 — minimum 5, across levels)

Mirrors `src/`'s layout under `tests/userdocs/` and `tests/telegram_bot/`,
same convention as the existing `tests/rag/*`. At minimum, one test per
assignment-named level plus the isolation test that's explicitly called a
hard requirement:

1. **Document parsing** (`tests/userdocs/test_extract.py`) — `.txt`, `.md`,
   `.docx`, `.pdf` each → expected text; a corrupted-PDF fixture → 
   `CorruptDocumentError`; an empty file → `EmptyDocumentError`; an
   unsupported extension → `UnsupportedFormatError`.
2. **Chunking** (`tests/userdocs/test_chunk.py`) — known text → expected
   chunk count/overlap boundaries; a PDF's per-page text → correct `page`
   attribution for a chunk known to start on page 2.
3. **Retrieval** (`tests/userdocs/test_retrieve.py`) — a query with vector
   search and FTS search both faked/monkeypatched (no real Ollama/model
   load needed, same pattern as `tests/rag/test_query.py`) → expected
   Top-K ordering; a query with all scores below `RELEVANCE_THRESHOLD` →
   `[]`.
4. **User isolation** — the hard requirement, tested at each of the three
   layers that enforce it rather than once:
   - `tests/userdocs/test_store.py` — user A's and user B's documents both
     indexed; `search_vectors` for user B's `user_id` never returns any of
     user A's chunk ids (asserted directly on ids, a structural check, not
     "the answer looked right"), and `delete_document` for user B cannot
     delete user A's same-named file.
   - `tests/userdocs/test_textsearch.py` — the same negative assertion for
     the FTS5 branch, which has no per-row ACL of its own.
   - `tests/telegram_bot/test_session.py` — one user's conversation turns
     never appear in another's `messages()`.
   - `tests/userdocs/test_pipeline_e2e.py` — the same property through the
     real pipeline: user 2 searching a question that user 1's indexed
     document answers perfectly gets the "not found" result.
5. **End-to-end** (`tests/userdocs/test_pipeline_e2e.py`) — a real small
   `.txt` fixture → `ingest_document` → `retrieve` → tool handler's
   formatted string, with Ollama itself mocked/faked at the `llm.chat`
   boundary (same principle as the brief's own "Telegram API при этом можно
   замокать" note, extended to the LLM call too, so this test doesn't
   require a running Ollama server) — asserts the final tool-result string
   contains both the expected source filename and the expected chunk text.

Plus, since §9 names 10 error categories and each needs its own assertion:
`tests/userdocs/test_errors.py` (or colocated per-module) asserting each of
the 10 exception→message mappings from §9's table.

Full test-by-test TDD steps live in the accompanying plans under
`docs/superpowers/plans/`.

## 17. RAG Evaluation dataset (assignment §16)

`eval/userdocs_eval.json` — ≥5 entries, same shape as the brief's example:

```json
[
  {"question": "How many paid vacation days do employees get per year?", "expected_source": "vacation_policy.pdf"},
  {"question": "Can unused vacation days be carried over to the next year?", "expected_source": "vacation_policy.pdf"},
  {"question": "What notice period is required when resigning?", "expected_source": "employee_handbook.docx"},
  {"question": "How many paid sick days are covered per year?", "expected_source": "employee_handbook.docx"},
  {"question": "Are remote employees eligible for the wellness stipend?", "expected_source": "benefits.md"},
  {"question": "What are the office opening hours?", "expected_source": "office_info.txt"}
]
```

Six questions rather than the required five, spread across all four documents
(and therefore all four formats) — a dataset that only probes one document
would pass while telling you almost nothing about retrieval.

`scripts/run_userdocs_eval.py` loads this file, ingests the fixture corpus
into a throwaway database, runs `retrieve()` for each `question`, and reports
whether `expected_source` appears among the returned chunks' filenames — the
same qualitative hit/miss reporting style as the existing `src/benchmark.py`,
not a formal precision/recall metric (matching that prior art's own scope
decision). It exits non-zero if any question misses, so it doubles as a CI
gate and as the demo checklist's "RAG evaluation" item.

The fixture corpus lives in `tests/fixtures/corpus/` and is shared with the
end-to-end test: four small documents, **one per supported format**
(`vacation_policy.pdf`, `employee_handbook.docx`, `benefits.md`,
`office_info.txt`), generated once by a committed script and checked in. Two
constraints on it, each with its own guard test in
`tests/fixtures/test_corpus.py`:

- every document must **genuinely contain** the answer to the questions that
  name it — an eval dataset checked against documents that don't answer it
  measures nothing;
- `vacation_policy.pdf` must be **at least two pages**, with the carry-over
  rule on page 2, so PDF page attribution (bonus +1) is verifiable rather
  than trivially always "page 1".

## 18. Acceptance criteria (maps to assignment's own checklist)

| Rubric item (assignment's Acceptance Criteria list) | Satisfied by |
|---|---|
| Бот принимает `.txt`, `.md`, `.docx`, `.pdf` | §5.1 `extract.py` |
| Извлекается текст | §5.1 |
| Текст разбивается на chunks | §5.2 |
| Для chunks создаются embeddings | §5.3 |
| Embeddings сохраняются в SQLite + sqlite-vec | §7 |
| Реализован vector search | §7 `search_vectors`, §8.1 `retrieve` |
| Агент использует retrieval для ответа | §11 `search_documents` tool + agent loop |
| Можно загрузить несколько документов | §7 schema (`documents` 1-to-many `chunks`), §10.1 handler runs per upload |
| Документы разных пользователей изолированы | §7 `search_vectors`/`delete_document`'s `user_id` filtering, §16 isolation test |
| Есть `/documents` | §10.2 |
| Есть удаление документов | §10.2 `/delete` |
| При удалении удаляются chunks и embeddings | §7 `delete_document` (explicit `chunk_vectors` delete + FK cascade) |
| Ответ содержит источник | §13 |
| Агент не выдумывает информацию | §14 |
| Обрабатываются основные ошибки | §9 (all 10 categories) |
| Минимум 5 автоматических тестов | §16 (5 named categories, more granular tests within each) |
| Evaluation dataset из 5 вопросов | §17 |
| README описывает архитектуру | §4's README row + this spec's §5–§14 content, condensed |

### Bonus traceability

| Bonus item (points) | Spec section | Plan file → task |
|---|---|---|
| **+1 Progress messages** | §5.4 `pipeline.ingest_document`'s `on_progress` callback; §10.1 the in-place-edit wiring | `2026-09-10-document-ingestion-pipeline.md` → Task 14 (`on_progress`/`ingest_document_async`)<br>`2026-09-10-telegram-integration.md` → Task 5 (one edited message per stage, incl. a test that a failed edit can't abort the upload) |
| **+1 PDF page numbers** | §5.1 `ExtractedDocument.pages`; §5.2 `chunk_text`'s `page` field; §13's source format | `2026-09-10-document-ingestion-pipeline.md` → Task 4 (per-page extraction), Task 7 (chunk→page attribution)<br>`2026-09-10-rag-retrieval-tool.md` → Task 9 (`_format_source` emits `, page N`)<br>`2026-09-10-testing-and-evaluation.md` → Task 1 (a deliberately 2-page PDF fixture so this is verifiable, not vacuous), Task 2 (`test_pdf_source_attribution_includes_a_page_number`) |
| **+1 Hybrid search** | §8.2 `textsearch.py`; §8.1 step 2's RRF fusion | `2026-09-10-rag-retrieval-tool.md` → Task 1 (`fusion.py`), Task 2 (FTS5 index in the store), Task 3 (`search_fts`, per-user filtered), Task 5 (fusion wired into `retrieve()`) |
| **+1 Reranking** | §8.3 `rerank.py`; §8.1 step 5 | `2026-09-10-rag-retrieval-tool.md` → Task 6 (`rerank.py`), Task 7 (wired into `retrieve()` with `RerankError` fallback) |
| **+2 Conversation-aware RAG** | §12 `session.py`; §11.2's loop feeding `session.messages()` | `2026-09-10-conversation-aware-rag.md` → Task 1 (store), Task 3 (bounded window), Task 4 (ephemeral, Ollama-valid tool messages), Task 5 (acceptance test reproducing the brief's own "перенести **их**" follow-up) |

No gap: every Bonus item has a spec section and at least one named plan task,
and each has at least one test that fails if the feature regresses.

## 19. Open decisions — summary (all also inline above as "Assumed default")

1. Embedding model/dimension: `all-MiniLM-L6-v2`, 384-dim (§5.3).
2. Chunk size/overlap: 500/75 tokens, distinct from the existing
   `src/config.py` constants (§5.2).
3. Top-K: 5 (§6).
4. LLM provider/model for this subsystem: Ollama, native tool-calling,
   default `qwen2.5:7b` (§11.2) — a deliberate departure from this repo's
   existing `qwen3:0.6b` for the reasons given there.
5. Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (§8.3).
6. No-hallucination trigger: cosine similarity < 0.30, applied post-Top-K,
   pre-rerank (§14, §8.3's note on ordering).
7. Distance metric: L2 on normalized vectors (≡ cosine ranking) — chosen for
   sqlite-vec version portability (§5.3, §7).
8. Storage sharding: one shared DB, `user_id`-filtered queries, not
   per-user files (§7).
9. Document size cap: 20 MB (§6, §9 #5).
10. Conversation history window: last 3 turns / 6 messages (§12).
11. Telegram-bot reuse scope: tool-loop/registry/LLM-client patterns
    ported/adapted; document upload, storage, and session identity model
    are new (§2).

## 20. Usage/observability stats (added 2026-09-13, post-launch)

Not part of the original assignment brief — added on request, to give the
user visibility into LLM token consumption and ingestion volume while
testing the bot. Decisions below were confirmed directly with the user.

**Surface:** a new `/stats` command, per-user (consistent with this
subsystem's hard per-user isolation elsewhere — no cross-user or global
view). No footer on every reply; kept opt-in so normal answers aren't
cluttered.

**Token scope:** summed per *turn* (one user text message → one
`agent.run()` call), not per individual LLM call. A turn's tool-use loop can
call `llm.chat()` more than once (initial call, then a follow-up after a
tool result); `/stats` reports the turn total, matching what the user
experiences as "one request" rather than exposing the loop's internal call
count.

**Source of the numbers:** Ollama's `/api/chat` response already returns
`prompt_eval_count` and `eval_count` (input/output token counts) —
currently discarded by `OllamaClient.chat()`. No new instrumentation is
needed, just threading these existing fields through.

**Persistence:** in-memory only, per running process. Resets to zero on
every bot restart. Chosen over a SQLite table because this is a
development-time visibility feature, not a product requirement with
durability expectations — revisit if that changes.

**Tracked per user:**
- `prompt_tokens`, `completion_tokens` (cumulative, summed across turns)
- `turns` (count of completed agent turns)
- `documents_indexed`, `chunks_indexed` (cumulative from ingestion; one
  vector per chunk, so `chunks_indexed` ≡ vectors created)
- `errors`: dict of exception class name → count, covering both ingestion
  errors (§9 categories #1-7) and LLM errors (§9 categories #8-9) — an
  operational health view alongside the usage numbers.

**Explicitly out of scope for this addition** (may be revisited later):
retrieval-stage stats (candidate pool size, hybrid search hit breakdown,
rerank effect), tool-call counts, LLM latency. The user considered these
and deferred them to keep this addition tightly scoped.
