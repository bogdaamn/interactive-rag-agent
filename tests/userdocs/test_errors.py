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
