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
