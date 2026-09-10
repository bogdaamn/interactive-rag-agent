# Document Ingestion Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the extract → chunk → embed → store (SQLite + sqlite-vec) pipeline for user-uploaded documents, with hard per-user isolation enforced at the storage layer, and the 10-category error taxonomy's ingestion-side exceptions (#1–7).

**Architecture:** A new `src/userdocs/` package. Each pipeline stage (`extract.py`, `chunk.py`, `embed.py`) is a pure function with no Telegram/LLM dependency, independently unit-testable. `store.py` owns the SQLite + sqlite-vec schema and is the only module that runs SQL — every read/write is `user_id`-scoped so isolation is enforced in one place. `pipeline.py::ingest_document` wires the stages together behind one entrypoint the (later, separate-plan) Telegram handler calls.

**Tech Stack:** Python 3.10+, stdlib `sqlite3`, `sqlite-vec` (new dependency), `pypdf`, `python-docx`, `tiktoken`, `sentence-transformers`, `numpy` — all already in `requirements.txt` except `sqlite-vec`.

**Spec:** `spec/v3/SPEC.md` §5 (upload pipeline), §6 (config), §7 (storage), §9 (error handling #1–7)

## Global Constraints

- Embedding model: `all-MiniLM-L6-v2`, 384-dim, L2-normalized before storage (spec §5.3).
- Chunk size/overlap: 500/75 tokens (spec §5.2), distinct from `src/config.py`'s existing 700/100.
- `MAX_DOCUMENT_BYTES = 20 * 1024 * 1024` (spec §6, §9 #5).
- Storage: one shared `userdocs.db`, every `documents`/`chunks`/`chunk_vectors` access filtered by `user_id` (spec §7, §10 hard requirement) — never write a query that reads `chunks`/`chunk_vectors` without first narrowing to this user's `document_id`/`chunk_id` set.
- `PRAGMA foreign_keys = ON` on every connection so `chunks.document_id`'s `ON DELETE CASCADE` fires.
- `.docx` gets no page numbers (`pages=None` always); only PDF gets page-level attribution (spec §5.1, bonus +1).
- Every new module lives under `src/userdocs/`; tests mirror it 1:1 under `tests/userdocs/`.
- TDD, strictly: for every task below, write the failing test, run it and confirm the failure reason matches what's expected, write the minimal implementation, run the test again and confirm it passes, then commit — before moving to the next task.

---

### Task 1: Config constants and typed exceptions

**Files:**
- Create: `src/userdocs/__init__.py`
- Create: `src/userdocs/config.py`
- Create: `src/userdocs/errors.py`
- Modify: `requirements.txt`
- Test: `tests/userdocs/__init__.py`
- Test: `tests/userdocs/test_config.py`
- Test: `tests/userdocs/test_errors.py`

**Interfaces:**
- Produces: `userdocs.config.{CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, EMBEDDING_DIM, TOP_K, CANDIDATE_K, RRF_K, RELEVANCE_THRESHOLD, MAX_DOCUMENT_BYTES, USERDOCS_DB_PATH, RERANK_MODEL, CONVERSATION_HISTORY_TURNS, AGENT_MAX_STEPS}` (`int`/`float`/`str` constants); `userdocs.errors.{UnsupportedFormatError, CorruptDocumentError, EmptyDocumentError, DocumentTooLargeError, EmbeddingError, SQLiteStoreError, RerankError}` (all `Exception` subclasses, each its own distinct type, no shared non-`Exception` base — callers `except` them individually per the spec's error table).

- [ ] **Step 1: Write the failing tests**

```python
# tests/userdocs/__init__.py
```

```python
# tests/userdocs/test_config.py
from userdocs import config


def test_chunk_overlap_smaller_than_chunk_size():
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE


def test_embedding_dim_matches_known_model():
    assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert config.EMBEDDING_DIM == 384


def test_relevance_threshold_is_a_valid_cosine_bound():
    assert 0.0 <= config.RELEVANCE_THRESHOLD <= 1.0


def test_max_document_bytes_is_20mb():
    assert config.MAX_DOCUMENT_BYTES == 20 * 1024 * 1024
```

```python
# tests/userdocs/test_errors.py
from userdocs import errors


def test_all_error_types_are_distinct_exception_subclasses():
    types = [
        errors.UnsupportedFormatError,
        errors.CorruptDocumentError,
        errors.EmptyDocumentError,
        errors.DocumentTooLargeError,
        errors.EmbeddingError,
        errors.SQLiteStoreError,
        errors.RerankError,
    ]
    for t in types:
        assert issubclass(t, Exception)
    assert len(set(types)) == len(types)


def test_errors_carry_a_message():
    exc = errors.CorruptDocumentError("bad.pdf could not be parsed")
    assert str(exc) == "bad.pdf could not be parsed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/userdocs/test_config.py tests/userdocs/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'userdocs'`

> **Before writing the implementation, check how `src/` gets onto `sys.path`.**
> The new modules live under `src/userdocs/` but the tests import them as
> `userdocs.*`, so something must put `src/` on the path. Run
> `cat conftest.py 2>/dev/null; cat tests/conftest.py 2>/dev/null; cat setup.cfg pyproject.toml pytest.ini 2>/dev/null | head -40`
> and check how the existing `tests/rag/test_*.py` files import `rag.*` —
> reuse that exact mechanism. If there is no such mechanism (the existing tests
> are run from inside `src/`), add a root `conftest.py`:
>
> ```python
> # conftest.py
> import sys
> from pathlib import Path
>
> sys.path.insert(0, str(Path(__file__).parent / "src"))
> ```
>
> and include it in this task's commit. Do not restructure the existing `src/`
> layout — the spec's non-goals keep the current pipeline's files untouched.

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/__init__.py
```

```python
# src/userdocs/config.py
"""Config constants for the user-documents RAG subsystem.

Deliberately separate from src/config.py (the existing company-KB pipeline's
config) so the two subsystems' constants can diverge (see spec/v3/SPEC.md §5.2)
without either importing the other.
"""

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

TOP_K = 5
CANDIDATE_K = TOP_K * 3
RRF_K = 60
RELEVANCE_THRESHOLD = 0.30

MAX_DOCUMENT_BYTES = 20 * 1024 * 1024

USERDOCS_DB_PATH = "userdocs.db"

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CONVERSATION_HISTORY_TURNS = 3
AGENT_MAX_STEPS = 4
```

```python
# src/userdocs/errors.py
"""Typed exceptions for the 10 error categories in spec/v3/SPEC.md §9.

Categories #8, #9, #10 (LLM error, timeout, Telegram API error) live in
src/telegram_bot/ instead — see docs/superpowers/plans/2026-09-10-telegram-integration.md.
"""


class UnsupportedFormatError(Exception):
    """Category #1: file extension is not one of .txt/.md/.docx/.pdf."""


class CorruptDocumentError(Exception):
    """Categories #2/#3: the PDF or DOCX parser raised while reading the file."""


class EmptyDocumentError(Exception):
    """Category #4: extracted text is empty after stripping whitespace."""


class DocumentTooLargeError(Exception):
    """Category #5: raw upload exceeds config.MAX_DOCUMENT_BYTES."""


class EmbeddingError(Exception):
    """Category #6: the embedding model failed to load or encode."""


class SQLiteStoreError(Exception):
    """Category #7: a sqlite3.Error was raised while reading/writing userdocs.db."""


class RerankError(Exception):
    """Bonus reranking failure (not one of the 10 numbered categories, but
    follows the same catch-and-degrade pattern — see retrieve.py)."""
```

Also append to `requirements.txt`:

```text
sqlite-vec
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/userdocs/test_config.py tests/userdocs/test_errors.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/__init__.py src/userdocs/config.py src/userdocs/errors.py \
        tests/userdocs/__init__.py tests/userdocs/test_config.py tests/userdocs/test_errors.py \
        requirements.txt
git commit -m "feat(userdocs): add config constants and typed error hierarchy"
```

---

### Task 2: `extract_text` for `.txt` / `.md`

**Files:**
- Create: `src/userdocs/extract.py`
- Test: `tests/userdocs/test_extract.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly (uses stdlib only for this step).
- Produces: `userdocs.extract.SUPPORTED_EXTENSIONS: set[str]`; `userdocs.extract.ExtractedDocument` (dataclass: `text: str`, `pages: list[str] | None`); `userdocs.extract.extract_text(filename: str, raw_bytes: bytes) -> ExtractedDocument`.

- [ ] **Step 1: Write the failing test**

```python
# tests/userdocs/test_extract.py
from userdocs.extract import extract_text


def test_extract_txt_returns_decoded_text_with_no_pages():
    raw = "Employees get 25 vacation days per year.".encode("utf-8")
    result = extract_text("policy.txt", raw)
    assert result.text == "Employees get 25 vacation days per year."
    assert result.pages is None


def test_extract_md_returns_decoded_text_with_no_pages():
    raw = "# Benefits\n\nWellness stipend: $50/month.".encode("utf-8")
    result = extract_text("benefits.md", raw)
    assert result.text == "# Benefits\n\nWellness stipend: $50/month."
    assert result.pages is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'userdocs.extract'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/extract.py
"""Text extraction for uploaded documents. See spec/v3/SPEC.md §5.1."""

from dataclasses import dataclass

from userdocs.errors import UnsupportedFormatError

SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}


@dataclass
class ExtractedDocument:
    text: str
    pages: list[str] | None


def _extension_of(filename: str) -> str:
    idx = filename.rfind(".")
    return filename[idx:].lower() if idx != -1 else ""


def extract_text(filename: str, raw_bytes: bytes) -> ExtractedDocument:
    ext = _extension_of(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(f"Unsupported file extension: {ext!r}")

    if ext in (".txt", ".md"):
        text = raw_bytes.decode("utf-8", errors="replace")
        return ExtractedDocument(text=text, pages=None)

    raise UnsupportedFormatError(f"Extension {ext!r} not yet implemented")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_extract.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/extract.py tests/userdocs/test_extract.py
git commit -m "feat(userdocs): extract text from .txt/.md documents"
```

---

### Task 3: `extract_text` for `.docx`

**Files:**
- Modify: `src/userdocs/extract.py`
- Modify: `tests/userdocs/test_extract.py`

**Interfaces:**
- Consumes: `python_docx.Document` (the `python-docx` package, already in `requirements.txt`).
- Produces: no new public names — extends `extract_text`'s dispatch.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/userdocs/test_extract.py
import io

from docx import Document as DocxDocument


def _make_docx_bytes(paragraphs: list[str]) -> bytes:
    doc = DocxDocument()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_extract_docx_joins_paragraphs_with_no_pages():
    raw = _make_docx_bytes(["Vacation policy", "Employees get 25 days per year."])
    result = extract_text("policy.docx", raw)
    assert result.text == "Vacation policy\nEmployees get 25 days per year."
    assert result.pages is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_extract.py -v`
Expected: FAIL — `test_extract_docx_joins_paragraphs_with_no_pages` raises `UnsupportedFormatError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/extract.py — add near the top
import io

from docx import Document as DocxDocument

from userdocs.errors import CorruptDocumentError, UnsupportedFormatError
```

```python
# src/userdocs/extract.py — inside extract_text, replace the trailing
# "raise UnsupportedFormatError(...)" fallback with:

    if ext == ".docx":
        try:
            doc = DocxDocument(io.BytesIO(raw_bytes))
        except Exception as exc:
            # Deliberately broad: python-docx raises PackageNotFoundError for a
            # non-zip file, but a truncated or subtly-malformed .docx surfaces
            # as KeyError, BadZipFile, or an lxml parse error depending on which
            # part is damaged. The caller only needs "this file is broken" — so
            # every parse-time failure maps to one exception type rather than
            # enumerating a list that a new python-docx version can invalidate.
            raise CorruptDocumentError(f"{filename} could not be parsed as .docx") from exc
        text = "\n".join(p.text for p in doc.paragraphs)
        return ExtractedDocument(text=text, pages=None)

    raise UnsupportedFormatError(f"Extension {ext!r} not yet implemented")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_extract.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/extract.py tests/userdocs/test_extract.py
git commit -m "feat(userdocs): extract text from .docx documents"
```

---

### Task 4: `extract_text` for `.pdf` with per-page text

**Files:**
- Modify: `src/userdocs/extract.py`
- Modify: `tests/userdocs/test_extract.py`

**Interfaces:**
- Consumes: `pypdf.PdfReader` (already in `requirements.txt`).
- Produces: for PDFs, `ExtractedDocument.pages` is a non-`None` `list[str]`, one entry per page.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/userdocs/test_extract.py
from pathlib import Path

from userdocs.errors import CorruptDocumentError

_FIXTURES = Path(__file__).parent / "fixtures"


def test_extract_pdf_returns_per_page_text():
    # A real fixture with actually-extractable text. pypdf can write blank
    # pages but can't draw text onto them, so the fixture is generated once
    # with reportlab and committed — small, deterministic, and no extra
    # package or network access needed at test time.
    raw = (_FIXTURES / "two_page_sample.pdf").read_bytes()
    result = extract_text("sample.pdf", raw)
    assert result.pages is not None
    assert len(result.pages) == 2
    assert "first page" in result.pages[0].lower()
    assert "second page" in result.pages[1].lower()
    assert result.text == "\n".join(result.pages)


def test_extract_pdf_corrupt_file_raises_corrupt_document_error():
    raw = b"%PDF-1.4 not actually a valid pdf body"
    try:
        extract_text("broken.pdf", raw)
        assert False, "expected CorruptDocumentError"
    except CorruptDocumentError:
        pass
```

Add a small fixture generator script (run once, output committed):

```python
# tests/userdocs/fixtures/generate_two_page_sample.py — one-off generator,
# committed as documentation of how the fixture was made. Not collected by
# pytest (no test_ prefix) and not imported by any test.
"""Regenerate tests/userdocs/fixtures/two_page_sample.pdf.

    pip install reportlab
    python tests/userdocs/fixtures/generate_two_page_sample.py

reportlab is NOT a runtime or test dependency — do not add it to
requirements.txt. The generated PDF is committed so the suite needs neither the
package nor a network connection.
"""

from pathlib import Path

from reportlab.pdfgen import canvas

OUTPUT = Path(__file__).parent / "two_page_sample.pdf"

pdf = canvas.Canvas(str(OUTPUT))
pdf.drawString(100, 700, "This is the first page of the sample document.")
pdf.showPage()
pdf.drawString(100, 700, "This is the second page of the sample document.")
pdf.showPage()
pdf.save()
print(f"Wrote {OUTPUT}")
```

Create the fixtures directory and generate the file before running the test:

```bash
mkdir -p tests/userdocs/fixtures
pip install reportlab
python tests/userdocs/fixtures/generate_two_page_sample.py
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_extract.py -v`
Expected: FAIL — `test_extract_pdf_returns_per_page_text` raises `UnsupportedFormatError`; `test_extract_pdf_corrupt_file_raises_corrupt_document_error` fails because no `CorruptDocumentError` is raised yet for `.pdf`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/extract.py — add to imports
from pypdf import PdfReader
```

```python
# src/userdocs/extract.py — replace the trailing fallback again:

    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            # `reader.pages` is lazy — page text is only extracted (and only
            # fails) when iterated, so the iteration has to be INSIDE the try.
            pages = [(page.extract_text() or "") for page in reader.pages]
        except Exception as exc:
            # Broad for the same reason as .docx above: pypdf raises
            # PdfReadError for a bad trailer but other exception types for
            # damaged xref tables, bad encodings, or encrypted files.
            raise CorruptDocumentError(f"{filename} could not be parsed as .pdf") from exc
        return ExtractedDocument(text="\n".join(pages), pages=pages)

    raise UnsupportedFormatError(f"Extension {ext!r} not yet implemented")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_extract.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/extract.py tests/userdocs/test_extract.py \
        tests/userdocs/fixtures/two_page_sample.pdf \
        tests/userdocs/fixtures/generate_two_page_sample.py
git commit -m "feat(userdocs): extract per-page text from .pdf documents"
```

---

### Task 5: `extract_text` empty-document handling

**Files:**
- Modify: `src/userdocs/extract.py`
- Modify: `tests/userdocs/test_extract.py`

**Interfaces:**
- Produces: `extract_text` now raises `EmptyDocumentError` for any format whose extracted text is empty after `.strip()`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/userdocs/test_extract.py
from userdocs.errors import EmptyDocumentError


def test_extract_txt_empty_file_raises_empty_document_error():
    try:
        extract_text("empty.txt", b"   \n\n  ")
        assert False, "expected EmptyDocumentError"
    except EmptyDocumentError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_extract.py -v`
Expected: FAIL — no `EmptyDocumentError` is raised, the whitespace-only text is returned as-is

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/extract.py — add to imports
from userdocs.errors import EmptyDocumentError
```

```python
# src/userdocs/extract.py — wrap the return of extract_text's dispatch:
# change each "return ExtractedDocument(...)" call site to instead build the
# result, then check-and-return once at the end. Restructure the function
# body to a single exit point:

def extract_text(filename: str, raw_bytes: bytes) -> ExtractedDocument:
    ext = _extension_of(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(f"Unsupported file extension: {ext!r}")

    if ext in (".txt", ".md"):
        text = raw_bytes.decode("utf-8", errors="replace")
        result = ExtractedDocument(text=text, pages=None)
    elif ext == ".docx":
        try:
            doc = DocxDocument(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise CorruptDocumentError(f"{filename} could not be parsed as .docx") from exc
        text = "\n".join(p.text for p in doc.paragraphs)
        result = ExtractedDocument(text=text, pages=None)
    else:  # ".pdf"
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages = [(page.extract_text() or "") for page in reader.pages]
        except Exception as exc:
            raise CorruptDocumentError(f"{filename} could not be parsed as .pdf") from exc
        result = ExtractedDocument(text="\n".join(pages), pages=pages)

    if not result.text.strip():
        raise EmptyDocumentError(f"{filename} contains no extractable text")
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_extract.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/extract.py tests/userdocs/test_extract.py
git commit -m "feat(userdocs): raise EmptyDocumentError for documents with no extractable text"
```

---

### Task 6: `chunk_text` — sliding token windows

**Files:**
- Create: `src/userdocs/chunk.py`
- Create: `tests/userdocs/test_chunk.py`

**Interfaces:**
- Consumes: `userdocs.config.{CHUNK_SIZE, CHUNK_OVERLAP}` (Task 1); `tiktoken` (already in `requirements.txt`).
- Produces: `userdocs.chunk.chunk_text(text: str, pages: list[str] | None) -> list[dict]`, each dict `{"text": str, "chunk_index": int, "page": int | None}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/userdocs/test_chunk.py
from userdocs.chunk import chunk_text


def test_chunk_text_short_input_returns_one_chunk():
    chunks = chunk_text("This is a short sentence.", pages=None)
    assert len(chunks) == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["text"] == "This is a short sentence."
    assert chunks[0]["page"] is None


def test_chunk_text_long_input_produces_overlapping_windows():
    # 1500 repeated words tokenize to well over CHUNK_SIZE=500 tokens,
    # guaranteeing at least 2 windows with CHUNK_OVERLAP=75 overlap.
    text = " ".join(["word"] * 1500)
    chunks = chunk_text(text, pages=None)
    assert len(chunks) >= 2
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    # every chunk_index is sequential starting at 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_chunk.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'userdocs.chunk'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/chunk.py
"""Token-based chunking. See spec/v3/SPEC.md §5.2.

Mirrors src/rag/chunk.py's tiktoken-based approach for consistency with the
rest of this repo, but uses userdocs.config's own CHUNK_SIZE/CHUNK_OVERLAP
(smaller than src/config.py's, tuned for shorter policy-document sections).
"""

import tiktoken

from userdocs.config import CHUNK_OVERLAP, CHUNK_SIZE

_ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, pages: list[str] | None) -> list[dict]:
    tokens = _ENCODING.encode(text)
    step = CHUNK_SIZE - CHUNK_OVERLAP

    chunks = []
    start = 0
    chunk_index = 0
    while start < len(tokens):
        window = tokens[start : start + CHUNK_SIZE]
        chunk_str = _ENCODING.decode(window)
        chunks.append({"text": chunk_str, "chunk_index": chunk_index, "page": None})
        chunk_index += 1
        start += step

    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_chunk.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/chunk.py tests/userdocs/test_chunk.py
git commit -m "feat(userdocs): chunk text into overlapping token windows"
```

---

### Task 7: `chunk_text` page attribution (bonus: PDF page numbers)

**Files:**
- Modify: `src/userdocs/chunk.py`
- Modify: `tests/userdocs/test_chunk.py`

**Interfaces:**
- Produces: when `pages is not None`, each chunk dict's `"page"` is the 1-based page number containing the chunk's first token (spec §5.2).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/userdocs/test_chunk.py
def test_chunk_text_attributes_page_number_when_pages_given():
    page1 = " ".join(["alpha"] * 400)   # well under CHUNK_SIZE tokens on its own
    page2 = " ".join(["beta"] * 400)
    pages = [page1, page2]
    full_text = "\n".join(pages)

    chunks = chunk_text(full_text, pages=pages)

    assert chunks[0]["page"] == 1
    # a chunk starting well into page 1's token range is still page 1
    assert all(c["page"] in (1, 2) for c in chunks)
    # the last chunk (starting after all of page 1's ~400 tokens plus some of
    # page 2) is attributed to page 2
    assert chunks[-1]["page"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_chunk.py -v`
Expected: FAIL — `chunks[-1]["page"]` is `None`, not `2`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/chunk.py — replace the module body's chunk_text with:

def _page_boundaries(pages: list[str]) -> list[int]:
    """Cumulative token count at the END of each page, e.g. for 3 pages of
    (100, 50, 200) tokens: [100, 150, 350]."""
    boundaries = []
    total = 0
    for page_text in pages:
        total += len(_ENCODING.encode(page_text))
        boundaries.append(total)
    return boundaries


def _page_for_token(token_index: int, boundaries: list[int]) -> int:
    """1-based page number containing token_index, given cumulative boundaries."""
    for page_num, boundary in enumerate(boundaries, start=1):
        if token_index < boundary:
            return page_num
    return len(boundaries)  # past the last boundary: attribute to the last page


def chunk_text(text: str, pages: list[str] | None) -> list[dict]:
    tokens = _ENCODING.encode(text)
    step = CHUNK_SIZE - CHUNK_OVERLAP
    boundaries = _page_boundaries(pages) if pages is not None else None

    chunks = []
    start = 0
    chunk_index = 0
    while start < len(tokens):
        window = tokens[start : start + CHUNK_SIZE]
        chunk_str = _ENCODING.decode(window)
        page = _page_for_token(start, boundaries) if boundaries is not None else None
        chunks.append({"text": chunk_str, "chunk_index": chunk_index, "page": page})
        chunk_index += 1
        start += step

    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_chunk.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/chunk.py tests/userdocs/test_chunk.py
git commit -m "feat(userdocs): attribute chunks to their source PDF page number"
```

---

### Task 8: `embed_chunks` — normalized embeddings

**Files:**
- Create: `src/userdocs/embed.py`
- Create: `tests/userdocs/test_embed.py`

**Interfaces:**
- Consumes: `userdocs.config.{EMBEDDING_MODEL, EMBEDDING_DIM}` (Task 1); `sentence_transformers.SentenceTransformer` (already in `requirements.txt`); `userdocs.errors.EmbeddingError` (Task 1).
- Produces: `userdocs.embed.embed_chunks(chunk_texts: list[str]) -> numpy.ndarray`, shape `(len(chunk_texts), EMBEDDING_DIM)`, `float32`, each row L2-normalized.

- [ ] **Step 1: Write the failing test**

```python
# tests/userdocs/test_embed.py
import numpy as np

from userdocs.config import EMBEDDING_DIM
from userdocs.embed import embed_chunks
from userdocs.errors import EmbeddingError


def test_embed_chunks_returns_correct_shape_and_dtype():
    vectors = embed_chunks(["25 vacation days per year.", "Notice period is 2 weeks."])
    assert vectors.shape == (2, EMBEDDING_DIM)
    assert vectors.dtype == np.float32


def test_embed_chunks_rows_are_l2_normalized():
    vectors = embed_chunks(["some text to embed"])
    norm = np.linalg.norm(vectors[0])
    assert abs(norm - 1.0) < 1e-4


def test_embed_chunks_wraps_encoder_failure(monkeypatch):
    import userdocs.embed as embed_module

    class BoomEncoder:
        def encode(self, *args, **kwargs):
            raise RuntimeError("model crashed")

    monkeypatch.setattr(embed_module, "_get_model", lambda: BoomEncoder())
    try:
        embed_chunks(["text"])
        assert False, "expected EmbeddingError"
    except EmbeddingError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_embed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'userdocs.embed'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/embed.py
"""Embedding generation. See spec/v3/SPEC.md §5.3.

Model is lazy-loaded (not at import time) so importing this module — e.g.
from a test that only exercises extract.py/chunk.py — doesn't pay the model
load cost. Reuses the same all-MiniLM-L6-v2 model already used by
src/rag/embed.py, loaded into a separate instance (no shared state between
the two subsystems, per spec/v3/SPEC.md §1's "additive, not a rewrite" scope).
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from userdocs.config import EMBEDDING_MODEL
from userdocs.errors import EmbeddingError

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_chunks(chunk_texts: list[str]) -> np.ndarray:
    try:
        model = _get_model()
        vectors = model.encode(
            chunk_texts, normalize_embeddings=True, convert_to_numpy=True
        )
    except Exception as exc:
        raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc
    return vectors.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_embed.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/embed.py tests/userdocs/test_embed.py
git commit -m "feat(userdocs): generate L2-normalized embeddings for chunks"
```

---

### Task 9: `UserDocsStore` — schema creation and `insert_document`

**Files:**
- Create: `src/userdocs/store.py`
- Create: `tests/userdocs/test_store.py`

**Interfaces:**
- Consumes: `userdocs.config.{USERDOCS_DB_PATH, EMBEDDING_DIM}` (Task 1); `sqlite_vec` (new dependency, Task 1); `userdocs.errors.SQLiteStoreError` (Task 1).
- Produces: `userdocs.store.UserDocsStore(db_path: str)` — `__init__` opens the connection, loads the `sqlite-vec` extension, runs `PRAGMA foreign_keys = ON`, and creates the schema if missing; `.insert_document(user_id: int, filename: str, file_type: str) -> int`; `.close() -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/userdocs/test_store.py
from userdocs.store import UserDocsStore


def test_insert_document_returns_new_id(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    doc_id = store.insert_document(user_id=1, filename="policy.pdf", file_type=".pdf")
    assert isinstance(doc_id, int)
    assert doc_id > 0
    store.close()


def test_insert_document_persists_across_reopen(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = UserDocsStore(db_path)
    doc_id = store.insert_document(user_id=1, filename="policy.pdf", file_type=".pdf")
    store.close()

    store2 = UserDocsStore(db_path)
    row = store2._conn.execute(
        "SELECT id, user_id, filename, file_type FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    assert row == (doc_id, 1, "policy.pdf", ".pdf")
    store2.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'userdocs.store'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/store.py
"""SQLite + sqlite-vec storage layer. See spec/v3/SPEC.md §7.

This is the ONLY module that runs SQL against userdocs.db. Every method that
reads or writes documents/chunks/chunk_vectors takes an explicit user_id and
narrows to it before touching chunks/chunk_vectors — see search_vectors and
delete_document below for the two-step "documents WHERE user_id, then chunks
WHERE document_id IN (...)" pattern this hard-requires (spec §10).
"""

import datetime
import sqlite3

import sqlite_vec

from userdocs.config import EMBEDDING_DIM
from userdocs.errors import SQLiteStoreError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    filename    TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    text          TEXT NOT NULL,
    page          INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
"""


class UserDocsStore:
    def __init__(self, db_path: str):
        try:
            self._conn = sqlite3.connect(db_path)
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING "
                f"vec0(embedding FLOAT[{EMBEDDING_DIM}])"
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise SQLiteStoreError(f"Failed to initialize userdocs.db: {exc}") from exc

    def insert_document(self, user_id: int, filename: str, file_type: str) -> int:
        try:
            created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            cursor = self._conn.execute(
                "INSERT INTO documents (user_id, filename, file_type, created_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, filename, file_type, created_at),
            )
            self._conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as exc:
            raise SQLiteStoreError(f"Failed to insert document: {exc}") from exc

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_store.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/store.py tests/userdocs/test_store.py
git commit -m "feat(userdocs): add UserDocsStore with sqlite-vec schema and insert_document"
```

---

### Task 10: `insert_chunks_with_vectors`

**Files:**
- Modify: `src/userdocs/store.py`
- Modify: `tests/userdocs/test_store.py`

**Interfaces:**
- Consumes: `numpy.ndarray` (from `embed_chunks`, Task 8) — this task doesn't import `embed.py`, it just accepts the array shape it produces.
- Produces: `.insert_chunks_with_vectors(document_id: int, chunks: list[dict], vectors: numpy.ndarray) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/userdocs/test_store.py
import numpy as np


def test_insert_chunks_with_vectors_stores_chunks_and_vectors(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    doc_id = store.insert_document(user_id=1, filename="policy.pdf", file_type=".pdf")

    chunks = [
        {"text": "chunk one", "chunk_index": 0, "page": 1},
        {"text": "chunk two", "chunk_index": 1, "page": 1},
    ]
    vectors = np.random.default_rng(0).random((2, 384)).astype(np.float32)

    store.insert_chunks_with_vectors(doc_id, chunks, vectors)

    rows = store._conn.execute(
        "SELECT chunk_index, text, page FROM chunks WHERE document_id = ? ORDER BY chunk_index",
        (doc_id,),
    ).fetchall()
    assert rows == [(0, "chunk one", 1), (1, "chunk two", 1)]

    vector_count = store._conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
    assert vector_count == 2
    store.close()


def test_insert_chunks_with_vectors_raises_on_length_mismatch(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    doc_id = store.insert_document(user_id=1, filename="policy.pdf", file_type=".pdf")
    chunks = [{"text": "only one chunk", "chunk_index": 0, "page": None}]
    vectors = np.zeros((2, 384), dtype=np.float32)  # mismatched length
    try:
        store.insert_chunks_with_vectors(doc_id, chunks, vectors)
        assert False, "expected ValueError"
    except ValueError:
        pass
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_store.py -v`
Expected: FAIL — `AttributeError: 'UserDocsStore' object has no attribute 'insert_chunks_with_vectors'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/store.py — add method to UserDocsStore

    def insert_chunks_with_vectors(self, document_id: int, chunks: list, vectors) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunks length ({len(chunks)}) != vectors row count ({vectors.shape[0]})"
            )
        try:
            for chunk, vector in zip(chunks, vectors):
                cursor = self._conn.execute(
                    "INSERT INTO chunks (document_id, chunk_index, text, page) "
                    "VALUES (?, ?, ?, ?)",
                    (document_id, chunk["chunk_index"], chunk["text"], chunk["page"]),
                )
                chunk_id = cursor.lastrowid
                self._conn.execute(
                    "INSERT INTO chunk_vectors (rowid, embedding) VALUES (?, ?)",
                    (chunk_id, sqlite_vec.serialize_float32(vector.tolist())),
                )
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to insert chunks/vectors: {exc}") from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_store.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/store.py tests/userdocs/test_store.py
git commit -m "feat(userdocs): insert chunks with their embeddings in one transaction"
```

---

### Task 11: `search_vectors` with per-user filtering (hard isolation requirement)

**Files:**
- Modify: `src/userdocs/store.py`
- Modify: `tests/userdocs/test_store.py`

**Interfaces:**
- Produces: `.search_vectors(user_id: int, query_vector: numpy.ndarray, k: int) -> list[tuple[int, float]]`, each tuple `(chunk_id, cosine_similarity)`, best first.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/userdocs/test_store.py
def _ingest_one_doc(store, user_id, filename, chunk_texts, rng):
    doc_id = store.insert_document(user_id, filename, ".txt")
    chunks = [{"text": t, "chunk_index": i, "page": None} for i, t in enumerate(chunk_texts)]
    vectors = rng.random((len(chunk_texts), 384)).astype(np.float32)
    # normalize rows so cosine-similarity math is meaningful in the test too
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    store.insert_chunks_with_vectors(doc_id, chunks, vectors)
    return doc_id, vectors


def test_search_vectors_returns_best_match_first(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    rng = np.random.default_rng(42)
    doc_id, vectors = _ingest_one_doc(store, 1, "doc.txt", ["a", "b", "c"], rng)

    # query with the exact vector for chunk index 1 -> should be the top hit
    query_vector = vectors[1]
    results = store.search_vectors(user_id=1, query_vector=query_vector, k=2)

    assert len(results) <= 2
    assert results[0][1] >= results[-1][1]  # descending similarity
    # the closest chunk_id should correspond to chunk_index 1 for this document
    top_chunk_id = results[0][0]
    row = store._conn.execute(
        "SELECT chunk_index FROM chunks WHERE id = ?", (top_chunk_id,)
    ).fetchone()
    assert row[0] == 1
    store.close()


def test_search_vectors_returns_empty_list_for_user_with_no_documents(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    query_vector = np.zeros(384, dtype=np.float32)
    results = store.search_vectors(user_id=999, query_vector=query_vector, k=5)
    assert results == []
    store.close()


def test_search_vectors_never_returns_another_users_chunks(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    rng = np.random.default_rng(7)
    _, vectors_a = _ingest_one_doc(store, user_id=1, filename="secret.pdf",
                                    chunk_texts=["s1", "s2"], rng=rng)
    _ingest_one_doc(store, user_id=2, filename="public.pdf",
                     chunk_texts=["p1", "p2"], rng=rng)

    # Search as user 2, using a query vector identical to one of user 1's chunks.
    results = store.search_vectors(user_id=2, query_vector=vectors_a[0], k=10)

    result_chunk_ids = {chunk_id for chunk_id, _ in results}
    user1_chunk_ids = {
        row[0] for row in store._conn.execute(
            "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.user_id = 1"
        ).fetchall()
    }
    assert result_chunk_ids.isdisjoint(user1_chunk_ids)
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_store.py -v`
Expected: FAIL — `AttributeError: 'UserDocsStore' object has no attribute 'search_vectors'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/store.py — add method to UserDocsStore

    def search_vectors(self, user_id: int, query_vector, k: int) -> list:
        try:
            document_ids = [
                row[0]
                for row in self._conn.execute(
                    "SELECT id FROM documents WHERE user_id = ?", (user_id,)
                ).fetchall()
            ]
            if not document_ids:
                return []

            placeholders = ",".join("?" for _ in document_ids)
            chunk_ids = {
                row[0]
                for row in self._conn.execute(
                    f"SELECT id FROM chunks WHERE document_id IN ({placeholders})",
                    document_ids,
                ).fetchall()
            }
            if not chunk_ids:
                return []

            over_fetch_k = min(max(k, len(chunk_ids)), 500)
            rows = self._conn.execute(
                "SELECT rowid, distance FROM chunk_vectors "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (sqlite_vec.serialize_float32(list(query_vector)), over_fetch_k),
            ).fetchall()

            results = []
            for rowid, distance in rows:
                if rowid in chunk_ids:
                    cosine_similarity = 1.0 - (distance ** 2) / 2.0
                    results.append((rowid, cosine_similarity))
                if len(results) >= k:
                    break
            return results
        except sqlite3.Error as exc:
            raise SQLiteStoreError(f"Vector search failed: {exc}") from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_store.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/store.py tests/userdocs/test_store.py
git commit -m "feat(userdocs): per-user-filtered vector search with cosine similarity"
```

---

### Task 12: `get_chunk` and `list_documents`

**Files:**
- Modify: `src/userdocs/store.py`
- Modify: `tests/userdocs/test_store.py`

**Interfaces:**
- Produces: `.get_chunk(chunk_id: int) -> dict` (`{"text", "chunk_index", "page", "filename"}`); `.list_documents(user_id: int) -> list[dict]` (`{"id", "filename", "file_type", "created_at"}`, ordered by `created_at ASC`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/userdocs/test_store.py
def test_get_chunk_returns_text_and_source_filename(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    doc_id = store.insert_document(1, "policy.pdf", ".pdf")
    store.insert_chunks_with_vectors(
        doc_id,
        [{"text": "vacation info", "chunk_index": 0, "page": 3}],
        np.zeros((1, 384), dtype=np.float32),
    )
    chunk_id = store._conn.execute("SELECT id FROM chunks").fetchone()[0]

    chunk = store.get_chunk(chunk_id)
    assert chunk == {
        "text": "vacation info",
        "chunk_index": 0,
        "page": 3,
        "filename": "policy.pdf",
    }
    store.close()


def test_list_documents_returns_only_this_users_documents_ordered(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    store.insert_document(1, "a.txt", ".txt")
    store.insert_document(1, "b.txt", ".txt")
    store.insert_document(2, "other_user.txt", ".txt")

    docs = store.list_documents(user_id=1)
    assert [d["filename"] for d in docs] == ["a.txt", "b.txt"]
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_store.py -v`
Expected: FAIL — `AttributeError: 'UserDocsStore' object has no attribute 'get_chunk'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/store.py — add methods to UserDocsStore

    def get_chunk(self, chunk_id: int) -> dict:
        try:
            row = self._conn.execute(
                "SELECT c.text, c.chunk_index, c.page, d.filename "
                "FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE c.id = ?",
                (chunk_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise SQLiteStoreError(f"Failed to fetch chunk {chunk_id}: {exc}") from exc
        if row is None:
            raise SQLiteStoreError(f"No chunk with id {chunk_id}")
        text, chunk_index, page, filename = row
        return {"text": text, "chunk_index": chunk_index, "page": page, "filename": filename}

    def list_documents(self, user_id: int) -> list:
        try:
            rows = self._conn.execute(
                "SELECT id, filename, file_type, created_at FROM documents "
                "WHERE user_id = ? ORDER BY created_at ASC",
                (user_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise SQLiteStoreError(f"Failed to list documents for user {user_id}: {exc}") from exc
        return [
            {"id": r[0], "filename": r[1], "file_type": r[2], "created_at": r[3]}
            for r in rows
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_store.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/store.py tests/userdocs/test_store.py
git commit -m "feat(userdocs): add get_chunk and list_documents"
```

---

### Task 13: `delete_document` — cascades chunks and vectors, isolation-safe

**Files:**
- Modify: `src/userdocs/store.py`
- Modify: `tests/userdocs/test_store.py`

**Interfaces:**
- Produces: `.delete_document(user_id: int, filename: str) -> bool` — `True` if a document was deleted, `False` if no matching document existed for this `user_id`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/userdocs/test_store.py
def test_delete_document_removes_document_chunks_and_vectors(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    doc_id = store.insert_document(1, "old.txt", ".txt")
    store.insert_chunks_with_vectors(
        doc_id, [{"text": "x", "chunk_index": 0, "page": None}],
        np.zeros((1, 384), dtype=np.float32),
    )

    deleted = store.delete_document(user_id=1, filename="old.txt")

    assert deleted is True
    assert store._conn.execute("SELECT COUNT(*) FROM documents WHERE id = ?", (doc_id,)).fetchone()[0] == 0
    assert store._conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,)).fetchone()[0] == 0
    assert store._conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0] == 0
    store.close()


def test_delete_document_returns_false_when_not_found(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    deleted = store.delete_document(user_id=1, filename="does_not_exist.txt")
    assert deleted is False
    store.close()


def test_delete_document_cannot_delete_another_users_file(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    doc_id = store.insert_document(user_id=1, filename="shared_name.txt", file_type=".txt")
    store.insert_chunks_with_vectors(
        doc_id, [{"text": "user1 data", "chunk_index": 0, "page": None}],
        np.zeros((1, 384), dtype=np.float32),
    )

    # user 2 tries to delete a file with the same name they never uploaded
    deleted = store.delete_document(user_id=2, filename="shared_name.txt")

    assert deleted is False
    assert store._conn.execute("SELECT COUNT(*) FROM documents WHERE id = ?", (doc_id,)).fetchone()[0] == 1
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_store.py -v`
Expected: FAIL — `AttributeError: 'UserDocsStore' object has no attribute 'delete_document'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/store.py — add method to UserDocsStore

    def delete_document(self, user_id: int, filename: str) -> bool:
        try:
            row = self._conn.execute(
                "SELECT id FROM documents WHERE user_id = ? AND filename = ?",
                (user_id, filename),
            ).fetchone()
            if row is None:
                return False
            document_id = row[0]

            chunk_ids = [
                r[0]
                for r in self._conn.execute(
                    "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
                ).fetchall()
            ]
            if chunk_ids:
                placeholders = ",".join("?" for _ in chunk_ids)
                self._conn.execute(
                    f"DELETE FROM chunk_vectors WHERE rowid IN ({placeholders})", chunk_ids
                )

            self._conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            self._conn.commit()
            return True
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to delete document {filename!r}: {exc}") from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_store.py -v`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/store.py tests/userdocs/test_store.py
git commit -m "feat(userdocs): delete_document removes chunks/vectors and is user-scoped"
```

---

### Task 14: `ingest_document` — the pipeline entrypoint

**Files:**
- Create: `src/userdocs/pipeline.py`
- Create: `tests/userdocs/test_pipeline.py`

**Interfaces:**
- Consumes: `userdocs.extract.extract_text` (Task 2–5), `userdocs.chunk.chunk_text` (Task 6–7), `userdocs.embed.embed_chunks` (Task 8), `userdocs.store.UserDocsStore` (Task 9–13), `userdocs.config.MAX_DOCUMENT_BYTES` (Task 1), `userdocs.errors.DocumentTooLargeError` (Task 1).
- Produces: `userdocs.pipeline.IngestResult` (dataclass: `document_id: int`, `filename: str`, `chunk_count: int`); `userdocs.pipeline.ingest_document(store, user_id: int, filename: str, raw_bytes: bytes, on_progress=None) -> IngestResult` (sync, `on_progress` is a plain callable); `async userdocs.pipeline.ingest_document_async(store, user_id: int, filename: str, raw_bytes: bytes, on_progress=None) -> IngestResult` (`on_progress` is an async callable — this is the one the Telegram handler uses).

- [ ] **Step 1: Write the failing test**

```python
# tests/userdocs/test_pipeline.py
import asyncio

from userdocs.errors import DocumentTooLargeError
from userdocs.pipeline import ingest_document, ingest_document_async
from userdocs.store import UserDocsStore


def test_ingest_document_stores_chunks_and_returns_result(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    raw = "Employees receive 25 paid vacation days per year.".encode("utf-8")

    result = ingest_document(store, user_id=1, filename="policy.txt", raw_bytes=raw)

    assert result.filename == "policy.txt"
    assert result.chunk_count >= 1
    stored_docs = store.list_documents(user_id=1)
    assert len(stored_docs) == 1
    assert stored_docs[0]["id"] == result.document_id
    store.close()


def test_ingest_document_rejects_oversized_upload(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    raw = b"x" * (20 * 1024 * 1024 + 1)  # one byte over MAX_DOCUMENT_BYTES
    try:
        ingest_document(store, user_id=1, filename="huge.txt", raw_bytes=raw)
        assert False, "expected DocumentTooLargeError"
    except DocumentTooLargeError:
        pass
    assert store.list_documents(user_id=1) == []
    store.close()


def test_ingest_document_async_awaits_on_progress_for_each_stage(tmp_path):
    store = UserDocsStore(str(tmp_path / "test.db"))
    raw = "Some short document text.".encode("utf-8")
    seen = []

    async def on_progress(message: str) -> None:
        seen.append(message)

    result = asyncio.run(
        ingest_document_async(store, 1, "doc.txt", raw, on_progress)
    )

    assert result.chunk_count >= 1
    assert any("Extracting" in m for m in seen)
    assert any("chunks created" in m for m in seen)
    assert any("Embeddings generated" in m for m in seen)
    store.close()


def test_ingest_document_works_with_on_progress_omitted(tmp_path):
    """Every on_progress call must be skipped entirely when it's None, so the
    pipeline has no Telegram dependency of its own."""
    store = UserDocsStore(str(tmp_path / "test.db"))
    result = ingest_document(store, 1, "doc.txt", b"Some text.", on_progress=None)
    assert result.chunk_count >= 1
    store.close()
```

> Note for the implementer: `ingest_document` itself is synchronous (it does
> no I/O that needs to be async — SQLite and the embedding model are both
> blocking calls); `on_progress` is the one *async* piece since it will
> eventually call Telegram's async API (Task in the Telegram-integration
> plan). To keep `ingest_document` callable from both sync tests and an
> async handler, this task adds a thin async wrapper,
> `ingest_document_async`, that awaits each `on_progress` call around the
> otherwise-synchronous `ingest_document`. See the implementation below.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/userdocs/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'userdocs.pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/userdocs/pipeline.py
"""Upload pipeline entrypoint. See spec/v3/SPEC.md §5.4."""

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

from userdocs.chunk import chunk_text
from userdocs.config import MAX_DOCUMENT_BYTES
from userdocs.embed import embed_chunks
from userdocs.errors import DocumentTooLargeError
from userdocs.extract import extract_text
from userdocs.store import UserDocsStore

OnProgress = Optional[Callable[[str], Awaitable[None]]]


@dataclass
class IngestResult:
    document_id: int
    filename: str
    chunk_count: int


def _ingest_sync_steps(store: UserDocsStore, user_id: int, filename: str, raw_bytes: bytes):
    """Runs every step except the on_progress calls; yields
    (stage_message, None) before each step and (None, final_result) at the end,
    so both the sync and async entrypoints can drive the same logic."""
    if len(raw_bytes) > MAX_DOCUMENT_BYTES:
        raise DocumentTooLargeError(
            f"{filename} is {len(raw_bytes)} bytes, exceeds the {MAX_DOCUMENT_BYTES} byte limit"
        )

    yield "⏳ Extracting text...", None
    extracted = extract_text(filename, raw_bytes)
    yield "✅ Text extracted", None

    yield "⏳ Creating chunks...", None
    chunks = chunk_text(extracted.text, extracted.pages)
    yield f"✅ {len(chunks)} chunks created", None

    yield "⏳ Generating embeddings...", None
    vectors = embed_chunks([c["text"] for c in chunks])
    yield "✅ Embeddings generated", None

    document_id = store.insert_document(user_id, filename, file_type=Path(filename).suffix)
    store.insert_chunks_with_vectors(document_id, chunks, vectors)

    yield None, IngestResult(document_id=document_id, filename=filename, chunk_count=len(chunks))


def ingest_document(
    store: UserDocsStore, user_id: int, filename: str, raw_bytes: bytes, on_progress=None
) -> IngestResult:
    """Synchronous entrypoint. `on_progress`, if given, must be a plain
    (non-async) callable — use ingest_document_async for an async caller."""
    result = None
    for message, final in _ingest_sync_steps(store, user_id, filename, raw_bytes):
        if message is not None and on_progress is not None:
            on_progress(message)
        if final is not None:
            result = final
    return result


async def ingest_document_async(
    store: UserDocsStore, user_id: int, filename: str, raw_bytes: bytes, on_progress: OnProgress = None
) -> IngestResult:
    """Async entrypoint for callers (e.g. the Telegram handler) whose
    on_progress needs to await a bot API call."""
    result = None
    for message, final in _ingest_sync_steps(store, user_id, filename, raw_bytes):
        if message is not None and on_progress is not None:
            await on_progress(message)
        if final is not None:
            result = final
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/userdocs/test_pipeline.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/userdocs/pipeline.py tests/userdocs/test_pipeline.py
git commit -m "feat(userdocs): wire extract/chunk/embed/store into ingest_document pipeline"
```

---

## Summary

At the end of this plan: `src/userdocs/{config,errors,extract,chunk,embed,store,pipeline}.py` exist, fully unit-tested (`tests/userdocs/test_{config,errors,extract,chunk,embed,store,pipeline}.py`), covering assignment §§1/3/4/5/6/9/10 error categories #1–7, and bonus +1 (PDF page numbers, Task 4/7) and the groundwork for bonus +1 (progress messages, Task 14's `on_progress`/`ingest_document_async`, wired into the actual Telegram handler in the Telegram-integration plan). `search_vectors`/`delete_document`'s per-`user_id` filtering (Tasks 11, 13) is the concrete answer to the hard per-user-isolation requirement, each with a dedicated negative test proving one user's data is unreachable from another user's call.
