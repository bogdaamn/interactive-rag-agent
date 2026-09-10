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
