"""Embedding generation. See spec/v3/SPEC.md §5.3.

Model is lazy-loaded (not at import time) so importing this module — e.g.
from a test that only exercises extract.py/chunk.py — doesn't pay the model
load cost. Reuses the same all-MiniLM-L6-v2 model already used by
src/rag/embed.py, loaded into a separate instance (no shared state between
the two subsystems, per spec/v3/SPEC.md §1's "additive, not a rewrite" scope).
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from userdocs.config import EMBEDDING_MODEL
from userdocs.errors import EmbeddingError

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_chunks(chunk_texts: list[str]) -> np.ndarray:
    try:
        model = _get_model()
        vectors = model.encode(
            chunk_texts, normalize_embeddings=True, convert_to_numpy=True
        )
    except Exception as exc:
        raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc
    return vectors.astype(np.float32)
