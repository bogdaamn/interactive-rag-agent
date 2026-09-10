# LLM/Tool Telemetry & Monitoring — Design Spec

**Status:** Approved for planning
**Date:** 2026-09-05
**Related:** [[architecture]] / [[index]] in the project's Obsidian vault (`~/Obsidian/local-rag-mcp/`)

## Purpose

Add monitoring around every LLM call and every tool call the assistant makes,
so an aggregate dashboard and a per-session timeline can be produced from
collected data — matching the two mockups in the assignment ("AI AGENT"
summary block and the per-task turn-by-turn timeline).

This is scoped to **Part 1: collection + viewing**. It does not cover
alerting, retention/rotation policy, or multi-process/concurrent-session
safety beyond SQLite's own file locking.

## Current LLM/tool call sites (as of `f655ea0`)

1. `src/rag/query.py::ask_llm()` — raw `requests.post` to Ollama's
   `/api/generate`. Used directly for the final-answer generation, and
   indirectly (via `expand.py::generate_keywords`) for query expansion.
2. `src/assistant.py::CompanyKBAssistant._llm_decide_mcp_usage()` — uses the
   `ollama` Python package's `Client().chat()` for the "should I call an MCP
   tool" decision.
3. `src/mcp/client.py::MCPClient.call_tool()` — the one tool-call site,
   shelling out to the MCP server subprocess for `read_document` /
   `list_documents` / `search_documents`.

All three get wrapped. No new LLM/tool call sites are being added as part of
this work.

## Data model: agent_id / task_id / turn_number

- `agent_id` — constant string, `"company-kb-assistant"`.
- `task_id` — one per REPL session. Assigned once when `CompanyKBAssistant`
  is constructed (`main.py` interactive mode). A UUID4 string.
- `turn_number` — increments once per user question. Every LLM call and tool
  call made while answering *that* question (MCP-decision call, tool call if
  any, query-expansion call, final-answer call) shares the same
  `turn_number`.
- Propagation: a `contextvars.ContextVar`-based context
  (`src/telemetry/context.py`: `current_task_id`, `current_turn_number`,
  `current_agent_id`) set at session start and bumped at the top of
  `CompanyKBAssistant.query()` for each new question. Nested calls (e.g.
  `generate_keywords()` calling `ask_llm()`) read the ambient context instead
  of having it threaded through every function signature.

## Metrics sourcing

| Field | Source |
|---|---|
| `timestamp` | wall-clock at call time, ISO 8601 |
| `agent_id`, `task_id`, `turn_number` | ambient context (above) |
| `model` | passed to the call (`config.OLLAMA_MODEL`, currently `qwen3:0.6b`) |
| `input_tokens` | Ollama response's `prompt_eval_count` (both `/api/generate` and `ollama.Client().chat()` return this when non-streaming) |
| `output_tokens` | Ollama response's `eval_count` |
| `cached_tokens` | `input_tokens` on a cache hit (see below), `0` on a miss |
| `reasoning_tokens` | always `0` — Ollama exposes no reasoning-token signal for this model; column kept for forward-compat with a future reasoning-capable/paid model |
| `latency` | wall-clock around the call; near-zero on a cache hit since no network call happens |
| `estimated_cost` | `input_tokens/1e6 * price_in + output_tokens/1e6 * price_out`, looked up from a hardcoded pricing table (below); unknown models default to `$0.00` with a one-time warning rather than raising |
| `turn_number` | see above |
| `call_site` | which of the 3 call sites made this call — `"ask_llm"` / `"mcp_decision"` / `"query_expansion"` |

Tool calls:

| Field | Source |
|---|---|
| `tool_name` | the MCP tool name (`read_document` / `list_documents` / `search_documents`) |
| `input_size` | byte length of the serialized tool args |
| `output_size` | byte length of the serialized tool result |
| `output_tokens` | `len(result_text) // 4` heuristic — no tokenizer is available for arbitrary tool output text; documented in code as an approximation, not exact |
| `duration` | wall-clock around `MCPClient.call_tool()` |

### Pricing table

Hardcoded in `src/telemetry/pricing.py`, keyed by Ollama model name, $ per
million tokens:

```python
PRICING = {
    "qwen3:0.6b": {"input": 0.20, "output": 0.20},  # Fireworks AI via OpenRouter, Sep 2026
}
DEFAULT_PRICING = {"input": 0.0, "output": 0.0}
```

Source: [OpenRouter Qwen models](https://openrouter.ai/qwen),
[pricepertoken.com OpenRouter listing](https://pricepertoken.com/endpoints/openrouter)
— Qwen3-0.6B listed at $0.20/M input, $0.20/M output tokens via Fireworks AI.
This is the *exact* model this repo runs, so no family-substitution was
needed.

## The cache

A real response cache — hits skip the Ollama call entirely, not just a
bookkeeping stat.

- Table: `llm_cache` in `src/telemetry.db` (same DB as telemetry — one file,
  one connection per call).
- Cache key: `sha256(normalize(prompt) + model + str(temperature))`.
  `normalize()` reuses/extracts the existing whitespace/case/punctuation
  collapse from `src/rag/expand.py::_normalize()` — **not** semantic/fuzzy
  matching. Two prompts must be the same question modulo trivial phrasing to
  hit.
- On hit: return the stored response, record `cached_tokens = input_tokens`
  (the stored count from when it was first cached), `output_tokens` also
  taken from the stored row, skip calling Ollama.
- On miss: call Ollama normally, store
  `(cache_key, model, response, input_tokens, output_tokens, created_at)`,
  record `cached_tokens = 0`.
- **Invalidation:** `src/rag/build_index.py::build_index()` clears the
  entire `llm_cache` table on every run — a rebuilt index can change which
  chunks are retrieved, so a cached answer for the same normalized question
  text may no longer be valid.
- Realistic hit-rate expectation (documented so it's not a surprise later):
  the final-answer and MCP-decision prompts both embed retrieved-chunk text
  that varies per query, so they rarely repeat even normalized. The
  query-expansion prompt (`expand.py`'s `PROMPT_TEMPLATE.format(query=query)`)
  is just the user's question, so that's the call site where repeat/rephrased
  questions actually hit.

## Storage schema (`src/telemetry.db`)

```sql
CREATE TABLE llm_calls (
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

CREATE TABLE tool_calls (
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

CREATE TABLE llm_cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    response TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
```

`*.db` is already gitignored repo-wide (see `6ccc4f4`'s fix for `fts.db`), so
`telemetry.db` needs no new gitignore entry, just confirmation it matches the
existing `*.db` pattern.

## CLI surface

New subcommands dispatched from `main.py` (alongside the existing
`build-index` mode), rendered with `rich.table.Table` (already a dependency):

- `python main.py dashboard` — aggregate stats over all history: tasks
  completed, total tokens (input/output/cached), estimated cost,
  average-per-task (tokens/turns/tool calls), cache hit rate, most-expensive
  tools breakdown — matching the assignment's "AI AGENT" mockup.
- `python main.py dashboard --tasks 184,185,190` — same aggregate, scoped to
  those `task_id`s.
- `python main.py dashboard --compare 184 185` — exactly 2 task_ids →
  4-column table: `Metric | Session 184 | Session 185 | Δ`. More than 2 ids
  passed to `--compare` is a usage error (a single well-defined delta column
  only makes sense pairwise); use `--tasks` for >2.
- `python main.py timeline <task_id>` — per-turn breakdown for one session,
  ordered by timestamp, matching the assignment's
  `Turn N   LLM   X tokens   tool   Y` mockup.

## Module layout

```
src/telemetry/
  __init__.py
  context.py          # contextvars: current_task_id/turn_number/agent_id
  storage.py           # sqlite connection + schema creation/migration
  pricing.py            # PRICING table + estimate_cost()
  llm_cache.py           # normalize() reuse, cache key, get/put
  llm_middleware.py       # record_llm_call() wrapping ask_llm() + the ollama.Client().chat() decision call
  tool_middleware.py       # record_tool_call() wrapping MCPClient.call_tool()
  dashboard.py               # aggregate + comparison queries and rich rendering
  timeline.py                 # per-task query and rich rendering
```

New `config.py` constants: `TELEMETRY_DB_PATH`, `AGENT_ID`.

## Testing

Implementation follows **TDD** (per `superpowers:test-driven-development`),
matching this repo's existing convention (`tests/rag/test_*.py` mirrors
`src/rag/*.py`, one test file per module, established across the hybrid-search
work). Each new module above gets a corresponding `tests/telemetry/test_*.py`
written before its implementation. Key units that need explicit test coverage
because they're easy to get subtly wrong:

- `llm_cache`: normalization collapses whitespace/case/punctuation but does
  **not** treat different questions as equal; hit vs. miss; cache-clear on
  `build_index()`.
- `pricing.estimate_cost()`: known model, unknown model (defaults to $0,
  warns once), zero-token edge case.
- `dashboard`/`timeline` aggregation SQL: empty DB, single task, multi-task,
  the `--compare` delta arithmetic (increase vs. decrease vs. unchanged).
- `context` propagation: nested calls (e.g. `generate_keywords` → `ask_llm`)
  inherit the ambient `task_id`/`turn_number` correctly.

## Out of scope (for this spec)

- Retention/rotation of `telemetry.db` (grows unbounded for now).
- Concurrent multi-process writers (single interactive REPL process is the
  only writer today).
- Anything beyond the two views above (no web dashboard, no charts/graphs —
  plain `rich` tables matching the assignment's plain-text mockups).
