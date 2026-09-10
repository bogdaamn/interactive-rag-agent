# tests/rag/test_build_index.py
import pickle

import numpy as np
import faiss

import rag.build_index as build_index_module
from rag.fts_index import search_fts


def test_build_index_writes_faiss_idmap_chunks_and_fts_index(tmp_path, monkeypatch):
    fake_docs = [{"path": "doc.txt", "text": "irrelevant, chunking is faked"}]
    fake_chunks = [
        {"id": 0, "text": "invoice processing steps", "source": "doc.txt", "chunk_id": 0},
        {"id": 1, "text": "vacation policy details", "source": "doc.txt", "chunk_id": 1},
    ]
    fake_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")

    monkeypatch.setattr(build_index_module, "ingest_documents", lambda: fake_docs)
    monkeypatch.setattr(build_index_module, "chunk_documents", lambda docs: fake_chunks)
    monkeypatch.setattr(build_index_module, "embed_chunks", lambda chunks: fake_embeddings)

    index_path = tmp_path / "index.faiss"
    chunks_path = tmp_path / "chunks.pkl"
    fts_db_path = tmp_path / "fts.db"
    # FAISS_INDEX_PATH/CHUNKS_PATH are imported at build_index.py's module top,
    # so patch the copies bound there. FTS_DB_PATH is imported *inside*
    # build_index() (a local import each call), so patch it at the source.
    monkeypatch.setattr(build_index_module, "FAISS_INDEX_PATH", str(index_path))
    monkeypatch.setattr(build_index_module, "CHUNKS_PATH", str(chunks_path))
    monkeypatch.setattr("config.FTS_DB_PATH", str(fts_db_path))

    build_index_module.build_index()

    loaded_index = faiss.read_index(str(index_path))
    query_emb = np.array([[1.0, 0.0]], dtype="float32")
    faiss.normalize_L2(query_emb)
    _, ids = loaded_index.search(query_emb, 1)
    assert ids[0][0] == 0  # IndexIDMap preserves chunk id 0, not just position

    with open(chunks_path, "rb") as f:
        loaded_chunks = pickle.load(f)
    assert loaded_chunks == fake_chunks

    assert search_fts(fts_db_path, "invoice", top_k=10) == [0]
