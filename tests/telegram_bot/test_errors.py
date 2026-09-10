from telegram_bot.errors import ERROR_MESSAGES, INGEST_ERRORS, message_for
from telegram_bot.llm_errors import LLMError, LLMTimeoutError
from userdocs.errors import (
    CorruptDocumentError,
    DocumentTooLargeError,
    EmbeddingError,
    EmptyDocumentError,
    SQLiteStoreError,
    UnsupportedFormatError,
)

_ALL_MAPPED_TYPES = [
    UnsupportedFormatError,
    CorruptDocumentError,
    EmptyDocumentError,
    DocumentTooLargeError,
    EmbeddingError,
    SQLiteStoreError,
    LLMError,
    LLMTimeoutError,
]


def test_every_error_category_has_a_message():
    for error_type in _ALL_MAPPED_TYPES:
        assert error_type in ERROR_MESSAGES
        assert ERROR_MESSAGES[error_type].strip()


def test_no_message_leaks_a_stack_trace_or_exception_class_name():
    for error_type, text in ERROR_MESSAGES.items():
        assert "Traceback" not in text
        assert error_type.__name__ not in text


def test_message_for_uses_the_exact_type_when_mapped():
    assert message_for(UnsupportedFormatError("x")) == ERROR_MESSAGES[UnsupportedFormatError]


def test_message_for_prefers_the_most_specific_type():
    """LLMTimeoutError subclasses LLMError — it must get its own timeout
    message, not the generic LLM one."""
    timeout_message = message_for(LLMTimeoutError("slow"))
    assert timeout_message == ERROR_MESSAGES[LLMTimeoutError]
    assert timeout_message != ERROR_MESSAGES[LLMError]


def test_message_for_falls_back_to_a_generic_message_for_unmapped_errors():
    text = message_for(RuntimeError("something nobody anticipated"))
    assert text.strip()
    assert "something nobody anticipated" not in text  # never echo raw exception text


def test_ingest_errors_covers_every_ingestion_side_category():
    assert set(INGEST_ERRORS) == {
        UnsupportedFormatError,
        CorruptDocumentError,
        EmptyDocumentError,
        DocumentTooLargeError,
        EmbeddingError,
        SQLiteStoreError,
    }
