import faiss
import pickle
import requests
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    OLLAMA_URL,
    OLLAMA_MODEL,
    TOP_K,
    FTS_DB_PATH,
    RRF_K,
    NUM_QUERY_EXPANSIONS,
    CANDIDATE_K,
)
from rag.fusion import rrf_fuse
from rag.fts_index import search_fts
from rag.expand import generate_keywords
from concurrent.futures import ThreadPoolExecutor

model = SentenceTransformer(EMBEDDING_MODEL)

src_dir = Path(__file__).parent.parent
fts_db_path = src_dir / FTS_DB_PATH

# Global variables for index and chunks
index = None
chunks = []
chunks_by_id = {}


def _ensure_index_exists():
    """Ensure FAISS index exists, build it if it doesn't."""
    global index, chunks, chunks_by_id

    # Resolve paths relative to src directory
    src_dir = Path(__file__).parent.parent
    index_path = src_dir / FAISS_INDEX_PATH
    chunks_path = src_dir / CHUNKS_PATH

    # Check if index exists
    if index_path.exists() and chunks_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            chunks_by_id = {c["id"]: c for c in chunks}
            return True
        except Exception as e:
            print(f"⚠️  Warning: Error loading existing index: {e}")
            print("Rebuilding index...")

    # Index doesn't exist or failed to load, build it
    print("📦 Index not found. Building index from documents...")
    try:
        from rag.build_index import build_index
        build_index()

        # Load the newly created index
        if index_path.exists() and chunks_path.exists():
            index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            chunks_by_id = {c["id"]: c for c in chunks}
            print("✅ Index built and loaded successfully")
            return True
        else:
            print("❌ Failed to build index. No documents found or error occurred.")
            from config import DOCUMENTS_DIR
            docs_path = src_dir / DOCUMENTS_DIR
            print(f"   Check that documents exist in: {docs_path}")
            return False
    except Exception as e:
        print(f"❌ Error building index: {e}")
        import traceback
        traceback.print_exc()
        return False


# Initialize index on module load
_ensure_index_exists()


def _vector_search(query, top_k):
    """Encode `query`, normalize, search the FAISS index, return up to
    top_k chunk ids best-first. [] if index/chunks aren't loaded."""
    if index is None or len(chunks) == 0:
        return []

    q_emb = model.encode([query])
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, top_k)
    return [int(i) for i in ids[0] if i != -1]


def _hybrid_search_task(search_fn, variants, top_k, rrf_k):
    """Run search_fn(variant, top_k) for every variant, collect each result
    as one ranked list, and rrf_fuse() them into a single ranking."""
    per_variant_rankings = [search_fn(variant, top_k) for variant in variants]
    return rrf_fuse(per_variant_rankings, k=rrf_k)


def retrieve(query):
    """Hybrid retrieval: query expansion -> parallel vector+FTS -> RRF fusion."""
    if index is None or len(chunks) == 0:
        if not _ensure_index_exists():
            return []

    if index is None or len(chunks) == 0:
        return []

    keywords = generate_keywords(query)
    variants = [query] + keywords[:NUM_QUERY_EXPANSIONS]

    def fts_search_fn(variant, top_k):
        return search_fts(fts_db_path, variant, top_k)

    with ThreadPoolExecutor(max_workers=2) as executor:
        vector_future = executor.submit(
            _hybrid_search_task, _vector_search, variants, CANDIDATE_K, RRF_K
        )
        fts_future = executor.submit(
            _hybrid_search_task, fts_search_fn, variants, CANDIDATE_K, RRF_K
        )
        vector_fused = vector_future.result()
        fts_fused = fts_future.result()

    final_ids = rrf_fuse([vector_fused, fts_fused], k=RRF_K)[:TOP_K]
    return [chunks_by_id[i] for i in final_ids if i in chunks_by_id]


def build_prompt(query, contexts):
    """Build prompt with retrieved context."""
    if not contexts:
        return f"""
<role>You are a helpful assistant that answers questions about company information.</role>
<instructions>Answer the question based on your general knowledge. If you don't know, say so.</instructions>

<query>
{query}
</query>

<assistant>
"""

    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in contexts
    )

    return f"""
<role>You are a helpful assistant that answers questions about company information.</role>
<instructions>Answer the question ONLY based on the context provided below. If the answer is not in the context, say "I don't have that information in the knowledge base."</instructions>

<context>
{context_text}
</context>

<query>
{query}
</query>

<assistant>
"""


def ask_llm(prompt, temperature=None, timeout=None):
    """Query Ollama LLM.

    temperature: if not None, sent as {"options": {"temperature": temperature}}.
    timeout: if not None, passed as requests.post(..., timeout=timeout).
             If None, blocks indefinitely (unchanged existing behavior).
    """
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    if temperature is not None:
        payload["options"] = {"temperature": temperature}
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    return response.json()["response"]


def ask(query: str):
    """Answer a question using RAG."""
    contexts = retrieve(query)
    prompt = build_prompt(query, contexts)
    return ask_llm(prompt), contexts


if __name__ == "__main__":
    while True:
        q = input("\n❓ Question: ")
        if q.lower() in {"exit", "quit"}:
            break
        print("\n🤖 Answer:\n")
        answer, sources = ask(q)
        print(answer)
        if sources:
            print("\n📚 Sources:")
            for src in sources:
                print(f"  - {src['source']}")
