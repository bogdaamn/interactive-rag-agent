import faiss
import numpy as np
import pickle
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.ingest import ingest_documents
from rag.chunk import chunk_documents
from rag.embed import embed_chunks
from config import FAISS_INDEX_PATH, CHUNKS_PATH


def build_index():
    """Build FAISS index from documents."""
    # Resolve paths relative to src directory
    src_dir = Path(__file__).parent.parent
    index_path = src_dir / FAISS_INDEX_PATH
    chunks_path = src_dir / CHUNKS_PATH

    print("📥 Loading documents...")
    documents = ingest_documents()

    if not documents:
        print("❌ No documents found. Please add documents to the docs directory.")
        return

    print("✂️ Chunking...")
    chunks = chunk_documents(documents)

    print("🧠 Generating embeddings...")
    embeddings = embed_chunks(chunks)

    print("📦 Creating FAISS index...")
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    ids = np.array([c["id"] for c in chunks], dtype=np.int64)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    index.add_with_ids(embeddings, ids)

    print("🔎 Creating FTS5 index...")
    from rag.fts_index import build_fts_index
    from config import FTS_DB_PATH
    fts_db_path = src_dir / FTS_DB_PATH
    build_fts_index(chunks, fts_db_path)

    print("💾 Saving...")
    faiss.write_index(index, str(index_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Indexing complete: {len(chunks)} chunks indexed")
    print(f"   Index saved to: {index_path}")
    print(f"   Chunks saved to: {chunks_path}")
    print(f"   FTS index saved to: {fts_db_path}")


if __name__ == "__main__":
    build_index()
