"""The search_documents tool the agent calls. See spec/v3/SPEC.md §11.1, §13.

user_id is captured in the closure below, not declared as a tool parameter, so
the LLM has no way to search another user's documents even if it tried — the
binding happens in trusted code (spec §10's hard isolation requirement, agent
layer).

Source attribution (spec §13) has two layers now. The bracketed hint below is
formatted into the tool's return text because that's the only thing the LLM
ever sees about a retrieved chunk. But the LLM composing its own citation from
that hint is exactly how it can invent a wrong filename under pressure (it did,
in testing) — so `format_citation_footer` builds the guaranteed-correct,
code-authored citation line that the caller appends to the final answer,
independent of whatever the model says in its own prose.
"""

from userdocs.retrieve import retrieve
from userdocs.tool_registry import Tool

NO_RESULTS_MESSAGE = "No relevant information found in the user's documents."

_DESCRIPTION = (
    "Search the user's uploaded documents for information relevant to a "
    "question. Use this whenever the user asks something that might be "
    "answered by a document they uploaded."
)

_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query, in the same language as the user's question.",
        }
    },
    "required": ["query"],
}


def _format_source(chunk) -> str:
    if chunk.page is not None:
        return f"[Source: {chunk.filename}, page {chunk.page}]"
    return f"[Source: {chunk.filename}]"


def _format_citation(chunk) -> str:
    """Preferred over _format_source's bracket hint for the *final* citation:
    always includes a page (PDF) or, failing that, a chunk number, so a
    non-PDF citation is still precise (assignment §13's "or chunk" option).
    1-indexed for readability — "chunk #1", not "chunk #0"."""
    if chunk.page is not None:
        return f"Source: {chunk.filename}, page {chunk.page}"
    return f"Source: {chunk.filename}, chunk #{chunk.chunk_index + 1}"


def format_citation_footer(chunks) -> str:
    """One citation per line, deduped in first-seen order, for the caller to
    append to the final answer text — this is what makes source attribution
    correct regardless of what the LLM says in its own prose."""
    lines = []
    for chunk in chunks:
        line = _format_citation(chunk)
        if line not in lines:
            lines.append(line)
    return "\n".join(lines)


def build_search_documents_tool(store, user_id: int, on_sources=None) -> Tool:
    async def _handler(query: str) -> str:
        chunks = retrieve(store, user_id, query)
        if not chunks:
            return NO_RESULTS_MESSAGE
        if on_sources is not None:
            on_sources(chunks)
        return "\n\n".join(f"{_format_source(c)}\n{c.text}" for c in chunks)

    return Tool(
        name="search_documents",
        description=_DESCRIPTION,
        parameters=_PARAMETERS,
        handler=_handler,
    )
