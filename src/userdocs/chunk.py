"""Token-based chunking. See spec/v3/SPEC.md §5.2.

Mirrors src/rag/chunk.py's tiktoken-based approach for consistency with the
rest of this repo, but uses userdocs.config's own CHUNK_SIZE/CHUNK_OVERLAP
(smaller than src/config.py's, tuned for shorter policy-document sections).
"""

import tiktoken

from userdocs.config import CHUNK_OVERLAP, CHUNK_SIZE

_ENCODING = tiktoken.get_encoding("cl100k_base")


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
