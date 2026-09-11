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
