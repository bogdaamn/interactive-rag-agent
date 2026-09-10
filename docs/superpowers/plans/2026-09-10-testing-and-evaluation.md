# End-to-End Testing, RAG Evaluation & README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the assignment's remaining three requirements — an end-to-end test through the real pipeline (§15), a 5-question RAG evaluation dataset with a runner that proves retrieval returns the right documents (§16), and the README sections documenting every technical decision (§17).

**Architecture:** Real fixture documents (one `.md`, one `.pdf`, one `.docx`, one `.txt` — covering all four supported formats) live under `tests/fixtures/corpus/` and are used by both the end-to-end test and the evaluation runner, so the eval questions are checked against documents that genuinely contain their answers. The end-to-end test runs the true ingest → retrieve → tool-result chain against a real SQLite + sqlite-vec database and a real embedding model, mocking only the LLM. The eval runner is a reporting script (like the existing `src/benchmark.py`), not a pytest test.

**Tech Stack:** Python 3.10+, `pytest`, real `sentence-transformers` + `sqlite-vec` (no mocking at the storage/embedding layer here — that's the point of an end-to-end test).

**Spec:** `spec/v3/SPEC.md` §16 (testing requirements), §17 (evaluation dataset), §4's README row

**Depends on:** all three prior plans —
`2026-09-10-document-ingestion-pipeline.md`, `2026-09-10-rag-retrieval-tool.md`, `2026-09-10-telegram-integration.md`. This is the last plan; run it after the others are green.

## Global Constraints

- Fixture documents must **genuinely answer** their eval questions. An evaluation dataset checked against documents that don't contain the answers measures nothing.
- The end-to-end test uses a **real** `UserDocsStore` (real sqlite-vec) and a **real** embedding model. Only the LLM is faked. Assignment §15 explicitly allows mocking Telegram; this plan extends that to the LLM so the suite runs without a live Ollama, but not to embeddings or storage — those are the parts an end-to-end test exists to exercise.
- The eval runner is a **script**, not a test: it prints a hit/miss report and exits non-zero if any question misses, so it's usable as a demo (assignment's demo checklist item 11) and in CI.
- README must cover all seven §17 headings: Architecture, Chunking, Embeddings, Retrieval, Storage, Security, Limitations — each with the *rationale*, not just the value.
- The existing README content (about the company-KB FAISS assistant) must not be deleted. The new material is an added section.
- TDD, strictly, for the test/code tasks. The README task is documentation and has no red/green cycle — its "verification" step is running the eval and test suite to confirm every number the README quotes is real.

---

### Task 1: Fixture corpus covering all four formats

**Files:**
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/corpus/vacation_policy.pdf`
- Create: `tests/fixtures/corpus/employee_handbook.docx`
- Create: `tests/fixtures/corpus/benefits.md`
- Create: `tests/fixtures/corpus/office_info.txt`
- Create: `tests/fixtures/generate_corpus.py`
- Create: `tests/fixtures/corpus/README.md`
- Test: `tests/fixtures/test_corpus.py`

**Interfaces:**
- Produces: `tests/fixtures/CORPUS_DIR` (a `pathlib.Path` to the corpus directory) importable by the end-to-end test and the eval runner, so neither hardcodes the path.

- [ ] **Step 1: Write the failing test**

```python
# tests/fixtures/test_corpus.py
"""Guards the fixture corpus: every format is present, every file is
extractable, and each file actually contains the text the eval dataset expects
to find in it. A silently-empty or unparseable fixture would make the
end-to-end test and the eval both pass vacuously."""

import pytest

from tests.fixtures import CORPUS_DIR
from userdocs.extract import extract_text

_EXPECTED_CONTENT = {
    "vacation_policy.pdf": ["25", "vacation"],
    "employee_handbook.docx": ["notice", "resignation"],
    "benefits.md": ["wellness", "stipend"],
    "office_info.txt": ["office", "hours"],
}


def test_corpus_covers_all_four_supported_formats():
    suffixes = {path.suffix.lower() for path in CORPUS_DIR.iterdir() if path.is_file()}
    assert {".txt", ".md", ".docx", ".pdf"} <= suffixes


@pytest.mark.parametrize("filename", sorted(_EXPECTED_CONTENT))
def test_each_fixture_extracts_to_text_containing_its_expected_terms(filename):
    path = CORPUS_DIR / filename
    assert path.exists(), f"missing fixture: {filename}"

    extracted = extract_text(filename, path.read_bytes())
    lowered = extracted.text.lower()

    for term in _EXPECTED_CONTENT[filename]:
        assert term.lower() in lowered, f"{filename} does not mention {term!r}"


def test_pdf_fixture_has_more_than_one_page_so_page_attribution_is_testable():
    extracted = extract_text("vacation_policy.pdf", (CORPUS_DIR / "vacation_policy.pdf").read_bytes())
    assert extracted.pages is not None
    assert len(extracted.pages) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/fixtures/test_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.fixtures'`

- [ ] **Step 3: Write minimal implementation**

```python
# tests/fixtures/__init__.py
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"
```

```python
# tests/fixtures/generate_corpus.py
"""One-off generator for the fixture corpus. Run once, commit the output.

Requires two packages that are NOT runtime dependencies and must not be added
to requirements.txt:
    pip install reportlab python-docx

Usage:
    python tests/fixtures/generate_corpus.py
"""

from pathlib import Path

from docx import Document as DocxDocument
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

CORPUS_DIR = Path(__file__).parent / "corpus"

VACATION_PAGE_1 = [
    "Vacation Policy",
    "",
    "All full-time employees receive 25 days of paid vacation per year.",
    "Vacation accrues monthly and becomes available after the first 90 days",
    "of employment.",
]

VACATION_PAGE_2 = [
    "Carrying Over Unused Vacation",
    "",
    "Up to 5 unused vacation days may be carried over into the following",
    "calendar year. Any balance above 5 days is forfeited on December 31.",
    "Carried-over days must be used by March 31.",
]

HANDBOOK_PARAGRAPHS = [
    "Employee Handbook",
    "",
    "Resignation and notice period.",
    "Employees resigning from the company must give a notice period of "
    "four weeks in writing to their direct manager.",
    "",
    "Sick leave.",
    "The company covers 10 paid sick days per calendar year. A doctor's note "
    "is required for any absence longer than three consecutive days.",
]

BENEFITS_MARKDOWN = """# Benefits

## Wellness

Every employee receives a wellness stipend of $50 per month, which can be
spent on gym memberships, fitness classes, or mental-health services.

Remote employees are fully eligible for the wellness stipend on the same terms
as office-based employees.

## Equipment

New hires receive a laptop and a one-time $300 home-office budget.
"""

OFFICE_INFO_TEXT = """Office Information

The office is open from 9:00 to 18:00, Monday through Friday.
Badge access is required outside those office hours.
The nearest parking garage is on Second Street.
"""


def _write_pdf() -> None:
    pdf_canvas = canvas.Canvas(str(CORPUS_DIR / "vacation_policy.pdf"), pagesize=letter)
    for page_lines in (VACATION_PAGE_1, VACATION_PAGE_2):
        y = 720
        for line in page_lines:
            pdf_canvas.drawString(72, y, line)
            y -= 18
        pdf_canvas.showPage()
    pdf_canvas.save()


def _write_docx() -> None:
    document = DocxDocument()
    for paragraph in HANDBOOK_PARAGRAPHS:
        document.add_paragraph(paragraph)
    document.save(str(CORPUS_DIR / "employee_handbook.docx"))


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    _write_pdf()
    _write_docx()
    (CORPUS_DIR / "benefits.md").write_text(BENEFITS_MARKDOWN, encoding="utf-8")
    (CORPUS_DIR / "office_info.txt").write_text(OFFICE_INFO_TEXT, encoding="utf-8")
    print(f"Wrote fixture corpus to {CORPUS_DIR}")


if __name__ == "__main__":
    main()
```

```markdown
<!-- tests/fixtures/corpus/README.md -->
# Fixture corpus

Four small documents, one per supported format, used by
`tests/userdocs/test_pipeline_e2e.py` and `scripts/run_userdocs_eval.py`.

Regenerate with:

```bash
pip install reportlab python-docx
python tests/fixtures/generate_corpus.py
```

`reportlab` is only needed to generate the `.pdf` — it is not a runtime or test
dependency and must not be added to `requirements.txt`. The generated files are
committed so the test suite needs neither package nor a network connection.

`vacation_policy.pdf` is deliberately two pages, with the carry-over rule on
page 2, so PDF page attribution (bonus +1) is verifiable rather than trivially
always "page 1".
```

Then run the generator:

```bash
pip install reportlab python-docx
python tests/fixtures/generate_corpus.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/fixtures/test_corpus.py -v`
Expected: PASS (6 passed — 2 standalone + 4 parametrized)

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/__init__.py tests/fixtures/generate_corpus.py \
        tests/fixtures/test_corpus.py tests/fixtures/corpus/
git commit -m "test: add fixture corpus covering all four supported formats"
```

---

### Task 2: End-to-end pipeline test

**Files:**
- Create: `tests/userdocs/test_pipeline_e2e.py`

**Interfaces:**
- Consumes: `tests.fixtures.CORPUS_DIR` (Task 1), `userdocs.pipeline.ingest_document`, `userdocs.store.UserDocsStore`, `userdocs.tools.build_search_documents_tool`, `userdocs.agent.run`, `userdocs.tool_registry.ToolRegistry`.
- Produces: no production code — the assignment §15 "End-to-end" test level.

> This test loads the real embedding model and builds a real sqlite-vec index, so it is meaningfully slower than the unit tests (roughly a few seconds for the model load, once per session). It's marked `@pytest.mark.slow` so it can be deselected during fast iteration, and Task 3 registers that marker.

- [ ] **Step 1: Write the failing test**

```python
# tests/userdocs/test_pipeline_e2e.py
"""End-to-end test (assignment §15's fifth level):

    document -> index -> question -> retrieval -> LLM -> answer

Real extraction, real chunking, real embeddings, real SQLite + sqlite-vec. Only
the LLM is faked, so the suite needs no running Ollama — the same allowance the
assignment makes for the Telegram API, extended to the LLM.
"""

import pytest

from tests.fixtures import CORPUS_DIR
from userdocs.agent import run as agent_run
from userdocs.pipeline import ingest_document
from userdocs.store import UserDocsStore
from userdocs.tool_registry import ToolRegistry
from userdocs.tools import NO_RESULTS_MESSAGE, build_search_documents_tool

pytestmark = pytest.mark.slow


@pytest.fixture
def ingested_store(tmp_path):
    """A real store with the full fixture corpus ingested for user 1."""
    store = UserDocsStore(str(tmp_path / "e2e.db"))
    for path in sorted(CORPUS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".docx", ".pdf"}:
            ingest_document(store, user_id=1, filename=path.name, raw_bytes=path.read_bytes())
    yield store
    store.close()


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.seen_messages = []

    async def chat(self, messages, tools):
        self.seen_messages.append([dict(m) for m in messages])
        return self._responses.pop(0)


def test_ingesting_the_corpus_creates_documents_and_chunks(ingested_store):
    documents = ingested_store.list_documents(user_id=1)
    assert len(documents) == 4

    total_chunks = ingested_store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert total_chunks >= 4


@pytest.mark.asyncio
async def test_question_retrieves_the_correct_document_and_cites_it(ingested_store):
    tool = build_search_documents_tool(ingested_store, user_id=1)
    result = await tool.handler(query="How many paid vacation days do employees get?")

    assert "vacation_policy.pdf" in result
    assert "25" in result
    assert "[Source: vacation_policy.pdf" in result


@pytest.mark.asyncio
async def test_pdf_source_attribution_includes_a_page_number(ingested_store):
    tool = build_search_documents_tool(ingested_store, user_id=1)
    result = await tool.handler(query="Can unused vacation days be carried over?")

    assert "vacation_policy.pdf" in result
    assert "page" in result


@pytest.mark.asyncio
async def test_non_pdf_source_attribution_has_no_page_number(ingested_store):
    tool = build_search_documents_tool(ingested_store, user_id=1)
    result = await tool.handler(query="What is the monthly wellness stipend?")

    assert "[Source: benefits.md]" in result


@pytest.mark.asyncio
async def test_unanswerable_question_returns_the_no_results_message(ingested_store):
    tool = build_search_documents_tool(ingested_store, user_id=1)
    result = await tool.handler(
        query="What is the airspeed velocity of an unladen swallow in kilometres per hour?"
    )

    assert result == NO_RESULTS_MESSAGE


@pytest.mark.asyncio
async def test_full_agent_turn_produces_an_answer_grounded_in_the_tool_result(ingested_store):
    """The complete chain: the agent decides to call search_documents, the real
    retrieval runs against the real index, and the tool output is fed back for
    the final answer."""

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
            self._messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
            self._messages.append({"role": "tool", "content": result})

    question = "How many paid vacation days do employees get?"
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [
            {"function": {"name": "search_documents", "arguments": {"query": question}}}
        ]},
        {"content": "Employees get 25 days. Source: vacation_policy.pdf", "tool_calls": []},
    ])

    session = FakeSession()
    session.append_user_message(question)
    registry = ToolRegistry([build_search_documents_tool(ingested_store, user_id=1)])

    answer = await agent_run(llm, registry, session, question)

    assert answer == "Employees get 25 days. Source: vacation_policy.pdf"
    # the real retrieval output reached the LLM before it composed that answer
    second_prompt_contents = [m.get("content", "") for m in llm.seen_messages[1]]
    assert any("vacation_policy.pdf" in c for c in second_prompt_contents)
    assert any("25" in c for c in second_prompt_contents)


@pytest.mark.asyncio
async def test_user_isolation_holds_through_the_real_pipeline(tmp_path):
    """Assignment §15's fourth level, exercised end-to-end rather than at the
    store layer: user 2 uploads nothing, so their search finds nothing — even
    though user 1's indexed documents answer the question perfectly."""
    store = UserDocsStore(str(tmp_path / "isolation.db"))
    ingest_document(
        store,
        user_id=1,
        filename="vacation_policy.pdf",
        raw_bytes=(CORPUS_DIR / "vacation_policy.pdf").read_bytes(),
    )

    tool_for_user_2 = build_search_documents_tool(store, user_id=2)
    result = await tool_for_user_2.handler(query="How many paid vacation days do employees get?")

    assert result == NO_RESULTS_MESSAGE
    assert "vacation_policy.pdf" not in result
    store.close()


@pytest.mark.asyncio
async def test_deleted_document_no_longer_participates_in_search(ingested_store):
    """Assignment §11: after /delete, the document must be gone from RAG search."""
    tool = build_search_documents_tool(ingested_store, user_id=1)

    before = await tool.handler(query="How many paid vacation days do employees get?")
    assert "vacation_policy.pdf" in before

    assert ingested_store.delete_document(user_id=1, filename="vacation_policy.pdf") is True

    after = await tool.handler(query="How many paid vacation days do employees get?")
    assert "vacation_policy.pdf" not in after
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_pipeline_e2e.py -v`
Expected: FAIL with `PytestUnknownMarkWarning`/errors on the unregistered `slow` marker, and `ModuleNotFoundError: No module named 'tests'` if `tests/` isn't importable — both fixed in Task 3. Run Task 3 first if you prefer; either order works, but do not skip it.

- [ ] **Step 3: Confirm the test body is correct, then move to Task 3 for the config it needs**

No production code changes belong in this task — the pipeline is already built by the prior three plans. If a test here fails for a *behavioral* reason (wrong source format, isolation leak, threshold too strict to match the fixture wording), that is a real bug: fix the relevant module and note which. In particular, if `test_unanswerable_question_returns_the_no_results_message` fails, `RELEVANCE_THRESHOLD` is too low for this corpus — tune the constant in `userdocs/config.py` and record the new value for the README (Task 6).

- [ ] **Step 4: Run test to verify it passes (after Task 3's config lands)**

Run: `pytest tests/userdocs/test_pipeline_e2e.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/userdocs/test_pipeline_e2e.py
git commit -m "test(userdocs): add end-to-end pipeline test with real storage and embeddings"
```

---

### Task 3: Pytest configuration for the `slow` marker and `tests` package imports

**Files:**
- Modify: `pytest.ini`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: a registered `slow` marker (so `pytest -m "not slow"` works and no `PytestUnknownMarkWarning` is emitted), and `tests` as an importable package so `from tests.fixtures import CORPUS_DIR` resolves.

- [ ] **Step 1: Write the failing test**

There is no unit test for pytest configuration — the verification is the command itself. Record the expected behavior:

```bash
# Must list the e2e tests as deselected, not error:
pytest -m "not slow" --collect-only -q
```

- [ ] **Step 2: Run the command to verify it currently fails**

Run: `pytest -m "not slow" --collect-only -q 2>&1 | tail -20`
Expected: a `PytestUnknownMarkWarning: Unknown pytest.mark.slow` warning, and/or a collection error importing `tests.fixtures`

- [ ] **Step 3: Write minimal implementation**

```ini
# pytest.ini
[pytest]
asyncio_mode = auto
testpaths = tests
markers =
    slow: tests that load a real embedding model or build a real vector index
```

```python
# tests/__init__.py
```

> Note: `tests/__init__.py` makes `tests` a package so `from tests.fixtures import
> CORPUS_DIR` works when pytest is run from the repo root. `tests/userdocs/`,
> `tests/telegram_bot/`, `tests/telegram_bot/handlers/`, and `tests/fixtures/`
> already have their own `__init__.py` from the prior plans and Task 1.

- [ ] **Step 4: Run the commands to verify they pass**

Run: `pytest -m "not slow" -q`
Expected: PASS with the 8 e2e tests reported as deselected, no unknown-marker warning

Run: `pytest -q`
Expected: PASS, full suite including the e2e tests

- [ ] **Step 5: Commit**

```bash
git add pytest.ini tests/__init__.py
git commit -m "test: register the slow marker and make tests an importable package"
```

---

### Task 4: RAG evaluation dataset

**Files:**
- Create: `eval/userdocs_eval.json`
- Create: `tests/test_eval_dataset.py`

**Interfaces:**
- Produces: `eval/userdocs_eval.json` — a JSON array of `{"question": str, "expected_source": str}` objects, ≥5 entries, every `expected_source` naming a file that exists in the fixture corpus.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_dataset.py
"""Guards the evaluation dataset itself (assignment §16).

A dataset whose expected_source names a file that isn't in the corpus, or whose
questions duplicate each other, would let the eval runner report a
meaningless score.
"""

import json
from pathlib import Path

from tests.fixtures import CORPUS_DIR

EVAL_PATH = Path(__file__).parent.parent / "eval" / "userdocs_eval.json"


def _load():
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def test_dataset_has_at_least_five_questions():
    assert len(_load()) >= 5


def test_every_entry_has_a_question_and_an_expected_source():
    for entry in _load():
        assert entry["question"].strip()
        assert entry["expected_source"].strip()


def test_every_expected_source_exists_in_the_fixture_corpus():
    for entry in _load():
        path = CORPUS_DIR / entry["expected_source"]
        assert path.exists(), f"expected_source not in corpus: {entry['expected_source']}"


def test_questions_are_unique():
    questions = [entry["question"] for entry in _load()]
    assert len(set(questions)) == len(questions)


def test_dataset_covers_more_than_one_source_document():
    sources = {entry["expected_source"] for entry in _load()}
    assert len(sources) >= 3, "an eval that only probes one document tests very little"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_dataset.py -v`
Expected: FAIL with `FileNotFoundError` for `eval/userdocs_eval.json`

- [ ] **Step 3: Write minimal implementation**

```json
[
  {
    "question": "How many paid vacation days do employees get per year?",
    "expected_source": "vacation_policy.pdf"
  },
  {
    "question": "Can unused vacation days be carried over to the next year?",
    "expected_source": "vacation_policy.pdf"
  },
  {
    "question": "What notice period is required when resigning?",
    "expected_source": "employee_handbook.docx"
  },
  {
    "question": "How many paid sick days are covered per year?",
    "expected_source": "employee_handbook.docx"
  },
  {
    "question": "Are remote employees eligible for the wellness stipend?",
    "expected_source": "benefits.md"
  },
  {
    "question": "What are the office opening hours?",
    "expected_source": "office_info.txt"
  }
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_dataset.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add eval/userdocs_eval.json tests/test_eval_dataset.py
git commit -m "test: add RAG evaluation dataset with six question/source pairs"
```

---

### Task 5: Evaluation runner script

**Files:**
- Create: `scripts/run_userdocs_eval.py`
- Create: `tests/test_eval_runner.py`

**Interfaces:**
- Consumes: `eval/userdocs_eval.json` (Task 4), `tests.fixtures.CORPUS_DIR` (Task 1), `userdocs.{pipeline, retrieve, store}`.
- Produces: `scripts.run_userdocs_eval.EvalResult` (dataclass: `question: str`, `expected_source: str`, `retrieved_sources: list[str]`, `hit: bool`); `scripts.run_userdocs_eval.evaluate(store, user_id: int, dataset: list[dict]) -> list[EvalResult]`; `scripts.run_userdocs_eval.format_report(results: list[EvalResult]) -> str`; `scripts.run_userdocs_eval.main() -> int` (exit code: `0` if every question hit, `1` otherwise).

> `evaluate` and `format_report` are factored out from `main` so they're unit-testable with a fake store — `main` itself is the only part that ingests the real corpus and prints.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_runner.py
from scripts.run_userdocs_eval import EvalResult, evaluate, format_report


class FakeStore:
    pass


def test_evaluate_marks_a_question_as_hit_when_the_expected_source_is_retrieved(monkeypatch):
    import scripts.run_userdocs_eval as runner

    from userdocs.retrieve import RetrievedChunk

    monkeypatch.setattr(
        runner,
        "retrieve",
        lambda store, user_id, query: [
            RetrievedChunk(text="t", filename="vacation_policy.pdf", page=1, chunk_index=0, score=0.9)
        ],
    )

    results = evaluate(
        FakeStore(),
        user_id=1,
        dataset=[{"question": "q", "expected_source": "vacation_policy.pdf"}],
    )

    assert len(results) == 1
    assert results[0].hit is True
    assert results[0].retrieved_sources == ["vacation_policy.pdf"]


def test_evaluate_marks_a_question_as_miss_when_the_expected_source_is_absent(monkeypatch):
    import scripts.run_userdocs_eval as runner

    from userdocs.retrieve import RetrievedChunk

    monkeypatch.setattr(
        runner,
        "retrieve",
        lambda store, user_id, query: [
            RetrievedChunk(text="t", filename="benefits.md", page=None, chunk_index=0, score=0.9)
        ],
    )

    results = evaluate(
        FakeStore(),
        user_id=1,
        dataset=[{"question": "q", "expected_source": "vacation_policy.pdf"}],
    )

    assert results[0].hit is False


def test_evaluate_marks_an_empty_retrieval_as_miss(monkeypatch):
    import scripts.run_userdocs_eval as runner

    monkeypatch.setattr(runner, "retrieve", lambda store, user_id, query: [])

    results = evaluate(
        FakeStore(), user_id=1, dataset=[{"question": "q", "expected_source": "a.pdf"}]
    )

    assert results[0].hit is False
    assert results[0].retrieved_sources == []


def test_evaluate_deduplicates_retrieved_sources(monkeypatch):
    """Top-K often returns several chunks from the same file; the report should
    name each source once."""
    import scripts.run_userdocs_eval as runner

    from userdocs.retrieve import RetrievedChunk

    monkeypatch.setattr(
        runner,
        "retrieve",
        lambda store, user_id, query: [
            RetrievedChunk(text="a", filename="policy.pdf", page=1, chunk_index=0, score=0.9),
            RetrievedChunk(text="b", filename="policy.pdf", page=2, chunk_index=1, score=0.8),
        ],
    )

    results = evaluate(
        FakeStore(), user_id=1, dataset=[{"question": "q", "expected_source": "policy.pdf"}]
    )

    assert results[0].retrieved_sources == ["policy.pdf"]


def test_format_report_shows_a_marker_per_question_and_a_total():
    results = [
        EvalResult("q1", "a.pdf", ["a.pdf"], True),
        EvalResult("q2", "b.pdf", ["a.pdf"], False),
    ]

    report = format_report(results)

    assert "q1" in report
    assert "q2" in report
    assert "1/2" in report
    # a miss must show what WAS retrieved, so the failure is diagnosable
    assert "a.pdf" in report


def test_format_report_handles_an_empty_result_set():
    assert format_report([]).strip()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_userdocs_eval'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/__init__.py
```

```python
# scripts/run_userdocs_eval.py
"""RAG evaluation runner (assignment §16).

Ingests the fixture corpus into a throwaway database, runs every question in
eval/userdocs_eval.json through the real retrieval path, and reports whether
the expected source document was actually retrieved.

This is a reporting script, not a pytest test — same role as the existing
src/benchmark.py. It exits non-zero if any question misses, so it also works as
a CI gate and as a live demo (demo checklist item 11).

Usage:
    python scripts/run_userdocs_eval.py
"""

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tests.fixtures import CORPUS_DIR  # noqa: E402
from userdocs.pipeline import ingest_document  # noqa: E402
from userdocs.retrieve import retrieve  # noqa: E402
from userdocs.store import UserDocsStore  # noqa: E402

EVAL_PATH = Path(__file__).parent.parent / "eval" / "userdocs_eval.json"
EVAL_USER_ID = 1


@dataclass
class EvalResult:
    question: str
    expected_source: str
    retrieved_sources: list
    hit: bool


def evaluate(store, user_id: int, dataset: list) -> list:
    results = []
    for entry in dataset:
        chunks = retrieve(store, user_id, entry["question"])

        seen = []
        for chunk in chunks:
            if chunk.filename not in seen:
                seen.append(chunk.filename)

        results.append(
            EvalResult(
                question=entry["question"],
                expected_source=entry["expected_source"],
                retrieved_sources=seen,
                hit=entry["expected_source"] in seen,
            )
        )
    return results


def format_report(results: list) -> str:
    if not results:
        return "No evaluation questions were run."

    lines = ["RAG evaluation", "=" * 60, ""]
    for result in results:
        marker = "✅" if result.hit else "❌"
        lines.append(f"{marker} {result.question}")
        lines.append(f"     expected: {result.expected_source}")
        lines.append(f"     retrieved: {', '.join(result.retrieved_sources) or '(nothing)'}")
        lines.append("")

    hits = sum(1 for result in results if result.hit)
    lines.append("=" * 60)
    lines.append(f"Retrieved the expected source for {hits}/{len(results)} questions.")
    return "\n".join(lines)


def main() -> int:
    dataset = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmpdir:
        store = UserDocsStore(str(Path(tmpdir) / "eval.db"))
        try:
            for path in sorted(CORPUS_DIR.iterdir()):
                if path.is_file() and path.suffix.lower() in {".txt", ".md", ".docx", ".pdf"}:
                    ingest_document(
                        store, EVAL_USER_ID, filename=path.name, raw_bytes=path.read_bytes()
                    )
            results = evaluate(store, EVAL_USER_ID, dataset)
        finally:
            store.close()

    print(format_report(results))
    return 0 if all(result.hit for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes, then run the real evaluation**

Run: `pytest tests/test_eval_runner.py -v`
Expected: PASS (6 passed)

Run: `python scripts/run_userdocs_eval.py`
Expected: a report printed, exit code `0` (`6/6`). If any question misses, that's a real retrieval-quality finding — record the actual score, and either reword the question to match the fixture's wording, or adjust `RELEVANCE_THRESHOLD`/`TOP_K` in `userdocs/config.py`. Do **not** delete a missing question to make the score look better; note whichever change you made for the README (Task 6).

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/run_userdocs_eval.py tests/test_eval_runner.py
git commit -m "feat: add RAG evaluation runner reporting retrieval hits per question"
```

---

### Task 6: README documentation (assignment §17)

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the real, verified values from every prior task — the config constants actually in `userdocs/config.py`, and the actual eval score from Task 5.
- Produces: a new `## User documents RAG (Telegram)` section with all seven §17 subsections.

> No red/green cycle here — this is documentation. Its verification step is confirming every number quoted is the one actually in the code, and that the eval score quoted is the one the runner actually prints.

- [ ] **Step 1: Collect the real values to document**

Run these and record the output — the README must quote what the code actually does, not what this plan predicted:

```bash
grep -n "=" src/userdocs/config.py
python scripts/run_userdocs_eval.py | tail -3
pytest -q 2>&1 | tail -3
```

- [ ] **Step 2: Confirm the existing README's structure before editing**

Run: `grep -n "^#" README.md`
Expected: a list of the existing headings (the company-KB assistant's docs). The new section is **appended**; nothing existing is removed.

- [ ] **Step 3: Append the new section**

Append to `README.md` (replacing every `<...>` with the real value collected in Step 1):

```markdown
## User documents RAG (Telegram)

A second, independent RAG subsystem: users send documents to a Telegram bot and
ask questions about them. Separate from the company-knowledge-base assistant
documented above — that one answers questions about a fixed corpus with FAISS;
this one indexes per-user uploads into SQLite + sqlite-vec.

Design spec: [`spec/v3/SPEC.md`](spec/v3/SPEC.md).
Implementation plans: [`docs/superpowers/plans/2026-09-10-*`](docs/superpowers/plans/).

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

- **Chunk size:** <CHUNK_SIZE> tokens (`tiktoken`, `cl100k_base`)
- **Overlap:** <CHUNK_OVERLAP> tokens (~15%)

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
- **Vector size:** <EMBEDDING_DIM> dimensions
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
- **K:** <TOP_K>, from a candidate pool of <CANDIDATE_K> before reranking.
  Five chunks is enough context for the model to answer and cite without
  diluting the prompt with marginal matches; the 3× candidate pool gives the
  reranker something to actually reorder.
- **Relevance threshold:** <RELEVANCE_THRESHOLD> cosine similarity. Anything
  below this is treated as "not found" — see Security/Limitations below for why
  this specific mechanism matters.
- **Hybrid search:** vector results are fused with FTS5/BM25 keyword results
  via Reciprocal Rank Fusion (k=<RRF_K>). Keyword search catches exact terms
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
user_id              id  ─────────────────────→ embedding FLOAT[<EMBEDDING_DIM>]
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
  clears <RELEVANCE_THRESHOLD> cosine similarity, the tool returns "no relevant
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
- **Conversation history is bounded to <CONVERSATION_HISTORY_TURNS> turns**, so
  a follow-up that depends on something said much earlier won't resolve.

### Running it

```bash
cp .env.example .env        # then set TELEGRAM_BOT_TOKEN
pip install -r requirements.txt
python src/telegram_bot/main.py
```

Commands: send a `.txt`/`.md`/`.docx`/`.pdf` file to index it, then ask
questions. `/documents` lists your documents, `/delete <filename>` removes one.

### Tests and evaluation

```bash
pytest                       # full suite
pytest -m "not slow"         # skip the tests that load a real embedding model
python scripts/run_userdocs_eval.py   # RAG evaluation report
```

Current state: <N> tests passing; the evaluation retrieves the expected source
document for <HITS>/<TOTAL> questions.
```

- [ ] **Step 4: Verify every quoted value against the code**

Run: `pytest -q && python scripts/run_userdocs_eval.py`

Then re-read the appended section and confirm each `<...>` was replaced with the
real value: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_DIM`, `TOP_K`,
`CANDIDATE_K`, `RELEVANCE_THRESHOLD`, `RRF_K`, `CONVERSATION_HISTORY_TURNS`, the
test count, and the eval hit/total. A README that quotes a stale constant is
worse than one that omits it.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document the user-documents RAG subsystem per assignment §17"
```

---

## Summary

This plan closes the last three assignment requirements:

| Requirement | Delivered by |
|---|---|
| §15 — minimum 5 automated tests across levels | Document parsing (`test_extract.py`), chunking (`test_chunk.py`), retrieval (`test_retrieve.py`), user isolation (`test_store.py`, `test_textsearch.py`, `test_session.py`, and end-to-end in `test_pipeline_e2e.py`), end-to-end (`test_pipeline_e2e.py`) — plus error-handling coverage across `tests/telegram_bot/` |
| §16 — evaluation dataset of ≥5 questions, verified retrieval | `eval/userdocs_eval.json` (6 questions across 4 documents), `scripts/run_userdocs_eval.py`, guarded by `tests/test_eval_dataset.py` and `tests/test_eval_runner.py` |
| §17 — README | Task 6's seven subsections, every value verified against the code |

Also covered here, because an end-to-end test is where they're actually provable
rather than asserted in isolation: source attribution with and without page
numbers, the no-hallucination path (`test_unanswerable_question_returns_the_no_results_message`),
isolation through the real pipeline, and the requirement that a deleted document
stops participating in search.

## All four plans, in execution order

1. `2026-09-10-document-ingestion-pipeline.md` — 14 tasks
2. `2026-09-10-rag-retrieval-tool.md` — 10 tasks
3. `2026-09-10-conversation-aware-rag.md` — 5 tasks *(its Task 1 must land before plan 4's Task 8)*
4. `2026-09-10-telegram-integration.md` — 10 tasks
5. `2026-09-10-testing-and-evaluation.md` — 6 tasks (this plan)
