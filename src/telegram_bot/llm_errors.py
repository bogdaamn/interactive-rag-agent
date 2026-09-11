"""LLM transport exceptions — error categories #8 and #9 in spec/v3/SPEC.md §9."""


class LLMError(Exception):
    """Category #8: the LLM call failed (connection, HTTP status, bad body)."""


class LLMTimeoutError(LLMError):
    """Category #9: the LLM call timed out. A subclass of LLMError so a caller
    that only cares about "the LLM failed" can catch one type, while the
    handler can still give a timeout-specific message."""
