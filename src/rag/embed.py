import numpy as np
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(chunks):
    """Generate embeddings for all chunks."""
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)


if __name__ == "__main__":
    # Test embedding
    test_chunks = [
        {"text": "This is a test chunk.", "source": "test.txt", "chunk_id": 0}
    ]
    embeddings = embed_chunks(test_chunks)
    print(f"Embedding shape: {embeddings.shape}")
