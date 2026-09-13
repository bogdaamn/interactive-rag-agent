"""Config constants for the user-documents RAG subsystem.

Deliberately separate from src/config.py (the existing company-KB pipeline's
config) so the two subsystems' constants can diverge (see spec/v3/SPEC.md §5.2)
without either importing the other.
"""

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

TOP_K = 5
CANDIDATE_K = TOP_K * 3
RRF_K = 60
RELEVANCE_THRESHOLD = 0.30
HYBRID_SEARCH_ENABLED = True
RERANK_ENABLED = True

MAX_DOCUMENT_BYTES = 20 * 1024 * 1024

USERDOCS_DB_PATH = "userdocs.db"

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Cross-encoder logits aren't calibrated against RELEVANCE_THRESHOLD's cosine
# scale, so this is a separate cutoff on the reranker's own scale — 0 is the
# model's relevant/irrelevant boundary. Verified against this repo's fixture
# corpus: an irrelevant same-topic-area chunk that clears the 0.30 cosine
# threshold anyway (e.g. vacation_policy.pdf on a notice-period question,
# +0.44 cosine) reliably scores well below 0 here (-5.53), while the actually
# relevant chunk scores well above (+7.16) — see rerank.py.
RERANK_THRESHOLD = 0.0
CONVERSATION_HISTORY_TURNS = 3
AGENT_MAX_STEPS = 4
