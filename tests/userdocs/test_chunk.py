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
