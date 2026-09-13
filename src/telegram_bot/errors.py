"""Exception → user-facing message mapping. See spec/v3/SPEC.md §9's table.

This is the single place a raised exception becomes text a user reads.
Assignment §12's requirement is that a user never sees a stack trace, so no
message here interpolates str(exc) — the exception detail goes to the log, the
fixed friendly string goes to the chat.

Category #10 (Telegram API error) has no entry: if the Telegram call itself
failed there is no channel left to send a message on. It's logged by the
handler's outermost except and nothing is sent.
"""

from telegram_bot.llm_errors import LLMError, LLMTimeoutError
from userdocs.errors import (
    CorruptDocumentError,
    DocumentTooLargeError,
    EmbeddingError,
    EmptyDocumentError,
    SQLiteStoreError,
    UnsupportedFormatError,
)

GENERIC_MESSAGE = "❌ Something went wrong. Please try again."

ERROR_MESSAGES = {
    # 1
    UnsupportedFormatError: (
        "❌ Unsupported file format.\n\nPlease send a .txt, .md, .docx, or .pdf file."
    ),
    # 2 and 3 (corrupted PDF / corrupted DOCX share one type and one message)
    CorruptDocumentError: (
        "❌ Could not process the document.\n\n"
        "Please make sure the file isn't corrupted."
    ),
    # 4
    EmptyDocumentError: (
        "❌ This document appears to be empty — there's nothing to index."
    ),
    # 5
    DocumentTooLargeError: (
        "❌ This file is too large (max 20 MB).\n\nPlease split it or send a smaller file."
    ),
    # 6
    EmbeddingError: (
        "❌ Something went wrong while indexing your document. Please try again."
    ),
    # 7
    SQLiteStoreError: (
        "❌ Something went wrong while saving your document. Please try again."
    ),
    # 8
    LLMError: "❌ The assistant is temporarily unavailable. Please try again shortly.",
    # 9
    LLMTimeoutError: "⏳ That took too long — please try again.",
}

INGEST_ERRORS = (
    UnsupportedFormatError,
    CorruptDocumentError,
    EmptyDocumentError,
    DocumentTooLargeError,
    EmbeddingError,
    SQLiteStoreError,
)


def message_for(exc: Exception) -> str:
    """Most-specific-type-wins lookup: walk the exception's MRO so a subclass
    (LLMTimeoutError) gets its own message before falling back to its base
    (LLMError), and an entirely unmapped exception gets a generic message
    rather than leaking its text."""
    for candidate in type(exc).__mro__:
        if candidate in ERROR_MESSAGES:
            return ERROR_MESSAGES[candidate]
    return GENERIC_MESSAGE
