"""
Compares vector-only retrieval (the pre-hybrid baseline) against the new
hybrid retrieve() on a small fixed set of sample queries, using whatever
documents currently exist in DOCUMENTS_DIR (build the index first via
`python main.py build-index` if needed — this script does not build it).

Usage: python src/benchmark.py
"""
from config import TOP_K
import rag.query as rag_query

# Each entry: a query designed to be hard for embeddings (rare terms, exact
# file/command names, abbreviations), and the substring expected to appear
# in a hit's source path/text. src/docs/ now has 4 CC BY-SA Wikipedia
# extracts (bm25_ranking.txt, full_text_search.txt, sentence_embeddings.txt,
# sqlite.txt) — rebuild the index first via `python main.py build-index`.
SAMPLE_QUERIES = [
    {"query": "avgdl", "expected_substring": "bm25_ranking.txt"},
    {"query": "ROWID", "expected_substring": "sqlite.txt"},
    {"query": "FTS5", "expected_substring": "sqlite.txt"},
    {"query": "WAL", "expected_substring": "sqlite.txt"},
    {"query": "Robertson", "expected_substring": "bm25_ranking.txt"},
]


def vector_only_retrieve(query):
    """Reimplements the pre-hybrid retrieve(): single vector search, no
    expansion, no FTS, no fusion."""
    ids = rag_query._vector_search(query, TOP_K)
    return [rag_query.chunks_by_id[i] for i in ids if i in rag_query.chunks_by_id]


def run_benchmark():
    """For each sample query: run vector_only_retrieve() and retrieve(),
    print both result sets' sources, and report hit/miss on
    expected_substring for each (qualitative demo, not a formal metric)."""
    if not SAMPLE_QUERIES:
        print(
            "No SAMPLE_QUERIES configured yet — add entries once documents "
            "exist in DOCUMENTS_DIR (see spec/v1/SPEC.md §9)."
        )
        return

    for case in SAMPLE_QUERIES:
        query_text = case["query"]
        expected = case["expected_substring"]

        vector_results = vector_only_retrieve(query_text)
        hybrid_results = rag_query.retrieve(query_text)

        vector_hit = any(expected in c["source"] or expected in c["text"] for c in vector_results)
        hybrid_hit = any(expected in c["source"] or expected in c["text"] for c in hybrid_results)

        print(f"\nQuery: {query_text!r}  (expecting {expected!r})")
        print(f"  vector-only: {'HIT' if vector_hit else 'MISS'} - {[c['source'] for c in vector_results]}")
        print(f"  hybrid:      {'HIT' if hybrid_hit else 'MISS'} - {[c['source'] for c in hybrid_results]}")


if __name__ == "__main__":
    run_benchmark()
