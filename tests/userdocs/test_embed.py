import numpy as np

from userdocs.config import EMBEDDING_DIM
from userdocs.embed import embed_chunks
from userdocs.errors import EmbeddingError


def test_embed_chunks_returns_correct_shape_and_dtype():
    vectors = embed_chunks(["25 vacation days per year.", "Notice period is 2 weeks."])
    assert vectors.shape == (2, EMBEDDING_DIM)
    assert vectors.dtype == np.float32


def test_embed_chunks_rows_are_l2_normalized():
    vectors = embed_chunks(["some text to embed"])
    norm = np.linalg.norm(vectors[0])
    assert abs(norm - 1.0) < 1e-4


def test_embed_chunks_wraps_encoder_failure(monkeypatch):
    import userdocs.embed as embed_module

    class BoomEncoder:
        def encode(self, *args, **kwargs):
            raise RuntimeError("model crashed")

    monkeypatch.setattr(embed_module, "_get_model", lambda: BoomEncoder())
    try:
        embed_chunks(["text"])
        assert False, "expected EmbeddingError"
    except EmbeddingError:
        pass
