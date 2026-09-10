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
