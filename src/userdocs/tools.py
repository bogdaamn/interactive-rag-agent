"""The search_documents tool the agent calls. See spec/v3/SPEC.md §11.1, §13.

user_id is captured in the closure below, not declared as a tool parameter, so
the LLM has no way to search another user's documents even if it tried — the
binding happens in trusted code (spec §10's hard isolation requirement, agent
layer).

Source attribution (spec §13) is formatted here because this string is the only
thing the LLM ever sees about a retrieved chunk; the system prompt in agent.py
tells it to cite what appears in these brackets.
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


def build_search_documents_tool(store, user_id: int) -> Tool:
    async def _handler(query: str) -> str:
        chunks = retrieve(store, user_id, query)
        if not chunks:
            return NO_RESULTS_MESSAGE
        return "\n\n".join(f"{_format_source(c)}\n{c.text}" for c in chunks)

    return Tool(
        name="search_documents",
        description=_DESCRIPTION,
        parameters=_PARAMETERS,
        handler=_handler,
    )
