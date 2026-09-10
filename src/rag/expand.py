import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NUM_QUERY_EXPANSIONS

PROMPT_TEMPLATE = """Extract up to 3 short search keywords or phrases from this question.
Reply with ONLY a comma-separated list, nothing else.

Question: {query}

Keywords:
"""


def _normalize(text):
    return re.sub(r"[^\w\s]", "", text).strip().lower()


def generate_keywords(query, ask_llm_fn=None):
    """Ask the LLM for up to NUM_QUERY_EXPANSIONS alternative search phrases
    for `query`. Returns [] on ANY failure (network error, timeout,
    unparseable/empty response) — never raises. `ask_llm_fn` defaults to
    `rag.query.ask_llm`, imported lazily to avoid a circular import with
    query.py (which imports generate_keywords at module level)."""
    if ask_llm_fn is None:
        from rag.query import ask_llm as ask_llm_fn

    prompt = PROMPT_TEMPLATE.format(query=query)
    try:
        response = ask_llm_fn(prompt, temperature=0.1, timeout=8.0)
    except Exception:
        return []

    normalized_query = _normalize(query)
    keywords = []
    for piece in response.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if _normalize(piece) == normalized_query:
            continue
        keywords.append(piece)

    return keywords[:NUM_QUERY_EXPANSIONS]
