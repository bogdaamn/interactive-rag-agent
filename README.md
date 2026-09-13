# Local RAG/MCP Knowledge Base Assistant

# 📋 The Problem

- **Growing Documentation**: Knowledge scattered across files
- **Information Retrieval**: Hard to find answers without keywords
- **Privacy Concerns**: Cloud solutions may not comply with policies

```
Users → Search → Answer = 😫
```

# ✨ The Solution

A **local, intelligent Q&A system** using:

- **RAG**: Semantic search over documentation
- **MCP**: Dynamic document access
- **Local LLM**: Privacy-preserving answers (Ollama)

# ✨ Key Benefits

- ✅ Privacy-first (runs locally)
- ✅ No API costs
- ✅ Fast semantic search
- ✅ Intelligent document access
- ✅ Complete data control

# 🏗️ Architecture - Top Level

```
┌──────────────────────┐
│   User Interface     │ (CLI)
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  [RAG]       [MCP]
   Query      Tools
     │           │
     └─────┬─────┘
           ▼
    [Ollama LLM]
```

# 🏗️ Architecture - Storage

```
┌────────────────┐
│  FAISS Index   │ Vector Database
│  + MCP Tools   │
└────────┬───────┘
         │
    ┌────▼─────┐
    │   docs/  │
    │directory │
    └──────────┘
```

# 🔍 RAG Pipeline

1. Document Loading → Read .md, .txt, .pdf, .docx
2. Chunking → Split into 700-token chunks (100-token overlap)
3. Embedding → Use SentenceTransformers
4. Indexing → Build FAISS vector index
5. Query → Retrieve top 5 similar chunks
6. Prompt Building → Create context-aware prompt
7. LLM Generation → Get answer from model

# 🔍 Chunking

- **Chunk size:** 700 tokens (`tiktoken`, `cl100k_base`) — `src/config.py::CHUNK_SIZE`
- **Overlap:** 100 tokens (~14%) — `src/config.py::CHUNK_OVERLAP`

Chosen for this corpus's `src/docs/*.txt` files: 600–2000-word technical
write-ups (BM25 ranking, full-text search, sentence embeddings, SQLite)
where a single idea is usually explained across several paragraphs rather
than in one self-contained clause. 700 tokens is roughly a few paragraphs —
enough to keep a multi-sentence explanation of one concept together in a
single chunk without also pulling in the document's next, unrelated
sub-topic. The overlap exists so a sentence that continues an idea right at
a chunk boundary (e.g. "as described above, X does Y") still has its
antecedent nearby in at least one of the two overlapping chunks, instead of
being retrievable but unintelligible on its own.

Failure modes at the extremes: chunks that are **too small** (roughly under
150–200 tokens) truncate mid-explanation, so the embedding captures a
fragment of an idea rather than the idea itself and the LLM has to guess at
context that was cut away; retrieval also gets noisier, since many small,
similar-looking fragments end up competing for the same query. Chunks that
are **too large** (over ~1500–2000 tokens) start covering multiple
sub-topics in one vector (e.g. one chunk spanning both "how BM25 scores
documents" and "BM25 vs TF-IDF"), which blurs the embedding's meaning and
makes nearest-neighbour search less discriminating — the model may retrieve
a technically-matching chunk where the actually relevant sentence is diluted
among mostly unrelated text, spending prompt tokens on padding instead of
directly relevant content.

# 🔍 Embeddings

- **Model:** `all-MiniLM-L6-v2` (`sentence-transformers`) — `src/config.py::EMBEDDING_MODEL`
- **Vector size:** 384 dimensions

Embedding generation can run either locally or against a hosted API (e.g.
OpenAI's `text-embedding-3-small`). This project runs it **locally**:
`all-MiniLM-L6-v2` is a small (~80MB) general-purpose sentence-embedding
model that needs no API key, no per-request cost, and no network round-trip
per chunk — consistent with the rest of the stack (FAISS, Ollama) being
local-first, and it means company documents never leave the machine just to
get embedded. Its retrieval quality is adequate at this corpus's scale (a
handful of documents, a few thousand chunks) — a larger model (e.g.
`all-mpnet-base-v2`, 768 dimensions) would improve ranking marginally at
roughly 2-3x the inference cost per chunk, a trade-off that isn't worth it
here but would be revisited for a much larger or more ambiguous corpus.

# 🔍 Why FAISS?

- Fast vector similarity search
- Lightweight and memory-efficient
- No external dependencies
- Perfect for local deployments
- Millions of vectors supported

# 🔧 MCP - Model Context Protocol

MCP provides **standardized interface** for LLM tool access:

```python
read_document(file_path)
list_documents()
search_documents(query)
```

# 🔧 MCP Benefits

- Tool Use by LLM
- Real-time document access
- Standardized interface
- Easy to extend
- Local tool execution

# 💻 Tech Stack

```
Language:      Python 3.10+
Vector DB:     FAISS
Embeddings:    SentenceTransformers
LLM:           Ollama (local)
MCP:           FastMCP
```

# 📁 Project Structure

```
src/
├── config.py           Configuration
├── main.py             CLI entry point
├── assistant.py        Main orchestrator
├── rag/
│   ├── ingest.py      Load documents
│   ├── chunk.py       Split text
│   ├── embed.py       Generate embeddings
│   ├── build_index.py Build FAISS index
│   └── query.py       Retrieve & generate
├── mcp/
│   ├── server.py      MCP tool definitions
│   └── client.py      MCP client wrapper
└── docs/              Documentation
```

# 🚀 Index Building (Setup)

```
$ python main.py build-index

1. Load documents
  ↓
2. Split into chunks
  ↓
3. Generate embeddings
  ↓
4. Build FAISS index
  ↓
5. Save files
```

# 🚀 Query Processing (Runtime)

```
User Question
  ↓
Embed question
  ↓
Search FAISS → Top 5 chunks
  ↓
LLM decides: Use MCP tools?
  ↓
Build prompt + context
  ↓
Call Ollama
  ↓
Return answer + sources
```

# ✨ Core Features

- **Semantic Search**: Find by meaning, not keywords
- **Multi-format**: .md, .txt, .pdf, .docx files
- **Source Attribution**: Shows document sources
- **MCP Tools**: LLM can read full documents
- **No External APIs**: Runs locally only
- **Fast Retrieval**: Sub-second search

# ⚙️ Configuration Options

```python
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen3:0.6b"
TOP_K = 5
```

# 🎬 Live Demo - Starting

```bash
$ python main.py
```

Output:
```
🤖 Company Knowledge Base
Ask questions about documentation
Type 'exit' to stop
```

# 🎬 Demo - Query 1

```
❓ What are company values?

🤖 Innovation, integrity, collaboration

📚 Sources:
  • Loan Rangers Team.md
  • Info Security.md
```

# 🎬 Demo - Query 2

```
❓ What documents do we have?

🤖 [Uses MCP list_documents]
  • Loan Rangers Team.md
  • Information Security.md
  • Services.md
```

# 🎬 Demo - Query 3

```
❓ Full security policy?

🤖 [Uses MCP read_document]
[Full document content...]
```

# 🔐 Security - Local vs Cloud

**Cloud**: Data → Internet → Server
- ⚠️ Network transmission
- ⚠️ External storage
- ⚠️ Subscription costs

**Local**: Data → Local System
- ✅ No transmission
- ✅ Local storage only
- ✅ No costs

# 🔐 Implementation Safeguards

- **MCP Sandbox**: Prevents path traversal
- **Local Storage**: Documents stay on device
- **No Telemetry**: No tracking
- **Offline Ready**: Works without internet

# ⚡ Performance Benchmarks

```
Index Building:   ~30s (one-time)
Query Embedding:  ~50ms
FAISS Search:     ~5ms
LLM Generation:   2-5s
Total Cycle:      2-6s
```

# ⚡ Tuning for Speed

```python
# Faster (smaller model):
OLLAMA_MODEL = "qwen3:0.6b"

# Faster retrieval:
TOP_K = 3
CHUNK_SIZE = 500
```

# 🚢 Deployment - Single Machine

```
1. Install Ollama & Python deps
2. Copy docs/ to server
3. Build index
4. Run with nohup

$ nohup python main.py > log &
```

# 🚢 Scaling - Option 1: FastAPI

```
[HTTP Clients]			[HTTP Clients + Webllm]
       ↓        						 ↓
   [FastAPI]     				 [FastAPI]
       ↓         					 ↓
[Ollama + FAISS]      			  [FAISS]
```

# 🚢 Scaling - Option 2: Distributed

```
[Clients] → [Load Balancer]
             ↓
      [Multiple Retrievers]
```

# 🚢 Storage Scaling

```
Docs     Index      Build
10 MB    ~2 MB      ~5s
100 MB   ~20 MB     ~30s
1 GB     ~200 MB    ~5min
```

# 🔮 Phase 2: Enhanced Features

- ☐ Web UI (Streamlit)
- ☐ API endpoints
- ☐ Multi-language support
- ☐ Document versioning
- ☐ Fine-tuned embeddings

# 🔮 Phase 3: Advanced

- ☐ Conversation memory
- ☐ Multi-hop reasoning
- ☐ Metadata filtering
- ☐ Feedback loop
- ☐ Analytics dashboard

# 🔮 Phase 4: Enterprise

- ☐ User authentication
- ☐ Audit logging
- ☐ Role-based access
- ☐ LLM fine-tuning
- ☐ Cost analysis

# 📊 Why This Works

| Aspect | Traditional | Our RAG |
|--------|---|---|
| **Understanding** | Keywords | Semantic |
| **Answers** | Documents | Direct |
| **Privacy** | Cloud | Local |
| **Cost** | Subscription | One-time |
| **Speed** | Slow | Sub-second |

# ✅ What You Have Now

- Local privacy-first knowledge base
- Fast semantic search (FAISS)
- Intelligent tool use (MCP)
- Maintainable Python code
- Foundation for enterprise features

# 🙋 Quick Reference

```bash
# Build index
python main.py build-index

# Run interactively
python main.py

# Check config
cat config.py
```

# 📚 Resources

- **Code**: MobilaName/local-rag-mcp
- **FAISS**: facebook/faiss
- **Ollama**: ollama.ai
- **FastMCP**: github.com/jlowin/fastmcp
- **Transformers**: huggingface.co

**Thank You!**

## User documents RAG (Telegram)

A second, independent RAG subsystem: users send documents to a Telegram bot and
ask questions about them. Separate from the company-knowledge-base assistant
documented above — that one answers questions about a fixed corpus with FAISS;
this one indexes per-user uploads into SQLite + sqlite-vec.

Design spec: [`spec/v3/SPEC.md`](spec/v3/SPEC.md).

### Architecture

```text
Telegram
   ↓
Agent (multi-step tool-use loop, src/userdocs/agent.py)
   ↓  search_documents(query)
RAG (src/userdocs/retrieve.py — vector + keyword search, RRF fusion, rerank)
   ↓
SQLite + sqlite-vec (src/userdocs/store.py)
```

Upload path: Telegram document → download → `extract.py` → `chunk.py` →
`embed.py` → `store.py`, orchestrated by `pipeline.py::ingest_document` with
progress callbacks.

Query path: the agent sees `search_documents` in its tool schema on every turn
and decides for itself whether to call it. The retrieval result is fed back into
the conversation and the model composes the final answer from it — retrieved
text is never injected unconditionally into a system prompt.

### Chunking

- **Chunk size:** 500 tokens (`tiktoken`, `cl100k_base`)
- **Overlap:** 75 tokens (~15%)

Chosen because uploaded policy documents tend to have short, self-contained
sections. A window this size rarely mixes two unrelated clauses into one chunk,
which matters because a chunk is the unit of source attribution — a chunk
spanning two topics produces a citation that's technically correct and
practically misleading. The overlap keeps a sentence that carries the actual
answer from being split across a boundary.

Failure modes at the extremes: chunks that are **too small** (say under 100
tokens) lose the surrounding context the model needs to phrase a correct answer,
and produce embeddings dominated by whichever few words happen to be present;
chunks that are **too large** (over ~1500 tokens) average too many topics into
one vector, so the nearest-neighbour search becomes less discriminating and
Top-K fills up with broadly-related-but-not-answering passages.

### Embeddings

- **Model:** `all-MiniLM-L6-v2` (`sentence-transformers`)
- **Vector size:** 384 dimensions
- **Why:** runs locally with no API key or per-token cost, is already a
  dependency of this repo's other pipeline, and is fast enough on CPU to index a
  multi-hundred-page PDF interactively. Its quality is adequate for
  document-scale retrieval, which is what this assignment needs — a larger
  model would cost latency on every upload for a marginal ranking gain that
  the reranking stage below already recovers.

Vectors are **L2-normalized before storage**. That makes L2 distance monotonic
with cosine similarity (`L2² = 2 − 2·cos_sim` for unit vectors), which lets the
whole system use plain L2 KNN without depending on a specific `sqlite-vec`
version's cosine-metric option, while still reasoning in calibrated cosine
similarity everywhere above the storage layer.

### Retrieval

- **Distance metric:** L2 over L2-normalized vectors, converted to cosine
  similarity as `1 − L2²/2` at the storage boundary. Cosine is the right
  similarity for sentence embeddings (direction carries the meaning, magnitude
  doesn't), and normalizing at write time buys version-independence.
- **K:** 5, from a candidate pool of 15 before reranking.
  Five chunks is enough context for the model to answer and cite without
  diluting the prompt with marginal matches; the 3× candidate pool gives the
  reranker something to actually reorder.
- **Relevance threshold:** 0.30 cosine similarity. Anything
  below this is treated as "not found" — see Security/Limitations below for why
  this specific mechanism matters.
- **Hybrid search:** vector results are fused with FTS5/BM25 keyword results
  via Reciprocal Rank Fusion (k=60). Keyword search catches exact terms
  (a policy number, an unusual proper noun) that a 384-dimensional embedding
  smooths over.
- **Reranking:** the candidate pool is reordered by a
  `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder, which reads
  (question, chunk) as a pair rather than comparing two independently-computed
  vectors. Reranking only reorders — it never re-admits a chunk that failed the
  relevance threshold, because a cross-encoder's output is an uncalibrated
  relevance score, not a cosine similarity.

### Storage

One SQLite database (`userdocs.db`), three tables plus two virtual tables:

```text
documents            chunks                    chunk_vectors (vec0)
---------            ------                    --------------------
id            ←───── document_id               rowid  = chunks.id
user_id              id  ─────────────────────→ embedding FLOAT[384]
filename             chunk_index
file_type            text                      chunks_fts (fts5)
created_at           page                      ----------
                                               rowid  = chunks.id
                                               text
```

`chunk_vectors.rowid` and `chunks_fts.rowid` are both `chunks.id`, so a search
result in either index joins straight back to its chunk, its document, and
therefore its filename and page — the `vector → chunk → document → filename`
traceability the assignment requires, with no separate mapping table.

`conversation_messages(user_id, role, content, created_at)` lives in the same
file and holds the conversation history used for follow-up questions.

`chunks.document_id` has `ON DELETE CASCADE` and `PRAGMA foreign_keys = ON` is
set on every connection, so deleting a document removes its chunks. The two
virtual tables aren't reached by that cascade (vec0/fts5 tables don't support
foreign keys), so `delete_document` deletes from them explicitly — a dedicated
test covers this, since an orphaned vector would keep a "deleted" document
answering questions.

### Security

Per-user isolation is enforced at three independent layers, so no single change
can silently break it:

1. **Storage.** `search_vectors` and `search_fts` never query the vector or
   keyword index for all users. Each first resolves
   `user_id → documents → chunks`, then filters results to that chunk-id set.
   `delete_document` matches on `(user_id, filename)`, so two users can have a
   same-named file and neither can delete the other's.
2. **Agent.** The `search_documents` tool is built fresh per incoming message
   with the sender's `user_id` captured in a closure. The tool schema the LLM
   sees has exactly one parameter, `query` — there is no `user_id` argument for
   a model to fill in, guess, or be talked into changing.
3. **Conversation history.** `Session` is keyed by `user_id` and every read
   filters on it, so one user's prior turns can never appear in another's
   prompt.

Each layer has a dedicated negative test asserting user A's data is unreachable
from a user-B call — `tests/userdocs/test_store.py`,
`tests/userdocs/test_textsearch.py`, `tests/userdocs/test_pipeline_e2e.py`, and
`tests/telegram_bot/test_session.py`.

### Limitations

- **Grounding is enforced by a similarity threshold, not a proof.** If nothing
  clears 0.30 cosine similarity, the tool returns "no relevant
  information found" and the system prompt instructs the model to say so. But a
  chunk that *does* clear the threshold and is still not a real answer can
  still be reasoned over — the threshold reduces hallucination risk, it doesn't
  eliminate it. The threshold value itself is tuned against this repo's small
  fixture corpus and would want retuning against a real document set.
- **`.docx` files get no page numbers.** `python-docx` exposes paragraphs, not
  rendered pages; real page breaks would need a layout engine. Only PDFs get
  page-level attribution.
- **Chunk-boundary page attribution is approximate.** A chunk spanning a page
  break is credited to the page its first token came from.
- **Vector search over-fetches then filters in Python.** `sqlite-vec` (at the
  pinned version) has no "restrict KNN to this rowid set" filter, so the
  per-user narrowing happens after the KNN call. Correct, but it scales with
  total corpus size rather than the requesting user's corpus size.
- **20 MB upload cap**, matching Telegram's own bot download limit without a
  local Bot API server.
- **No retry on LLM timeout.** The user is told to try again; nothing retries
  automatically.
- **Single-process only.** Concurrency safety is whatever SQLite's file locking
  provides; there's no connection pool or writer queue.
- **Conversation history is bounded to 3 turns**, so
  a follow-up that depends on something said much earlier won't resolve.

### Running it

```bash
cp .env.example .env        # then set TELEGRAM_BOT_TOKEN
python3 -m venv .venv && .venv/bin/pip install -r src/requirements.txt

./scripts/start.sh          # background daemon: starts Ollama, warms up the
                             # model, launches the bot, writes bot.log/.bot.pid
./scripts/stop.sh           # stops the bot and Ollama
```

`start.sh`/`stop.sh` follow the same convention as the sibling `telegram-bot`
and `local-rag-mcp` projects — see the comments at the top of each script.
If a `start <target>` / `stop <target>` shell function is set up (see
`~/.zshrc`), use `start interactive-rag` / `stop interactive-rag` instead.

To run it in the foreground for debugging instead (Ctrl+C to stop):

```bash
PYTHONPATH=src .venv/bin/python -m telegram_bot.main
```

Note: `python src/telegram_bot/main.py` does **not** work directly — the
package's imports (`telegram_bot.*`, `userdocs.*`) resolve relative to `src/`,
so it must be run as a module with `src/` on `PYTHONPATH` (which is what both
`scripts/start.sh` and pytest's `pythonpath = src` setting already do).

Commands: send a `.txt`/`.md`/`.docx`/`.pdf` file to index it, then ask
questions. `/documents` lists your documents, `/delete <filename>` removes
one, `/stats` shows your usage (tokens spent, documents/chunks indexed,
errors encountered).

### Usage stats

`/stats` reports, for the requesting user only: cumulative LLM tokens spent
(prompt + completion, summed per agent turn — a single question can trigger
more than one LLM call via the tool-use loop, and `/stats` reports the turn
total rather than each internal call separately), documents and
chunks/vectors indexed, and error counts by category.

The numbers come from fields Ollama's `/api/chat` already returns
(`prompt_eval_count`/`eval_count`) and from the ingestion pipeline's own
chunk count — no extra instrumentation, just surfacing what was already
computed. Tracking is **in-memory only**: it resets to zero on every bot
restart, since this is a development-time visibility feature rather than a
durable product requirement. See `spec/v3/SPEC.md` §20 for the full design
rationale.

### Tests and evaluation

```bash
pytest                       # full suite
pytest -m "not slow"         # skip the tests that load a real embedding model
python scripts/run_userdocs_eval.py   # RAG evaluation report
```

Current state: 221 tests passing (213 with `pytest -m "not slow"`, skipping the
8 tests that load a real embedding model); the evaluation retrieves the
expected source document for 6/6 questions.
