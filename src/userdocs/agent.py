"""Multi-step tool-use loop. See spec/v3/SPEC.md §11.2.

Adapted from ../telegram-bot/agent.py (spec §2's reuse table) — the loop shape
is the same; the system prompt is userdocs-specific and carries two hard
behavioral requirements from the assignment: always cite the source (§13), and
never substitute general knowledge for a document lookup (§14).

This module deliberately depends only on duck-typed llm/registry/session
objects, so it can be tested with fakes and reused by any caller (the Telegram
handler is the only real one today).
"""

from userdocs.config import AGENT_MAX_STEPS

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about documents the "
    "user has uploaded. You have access to a search_documents tool — use it "
    "whenever the question might be answered by their documents. If "
    "search_documents returns 'No relevant information found in the user's "
    "documents.', or the retrieved text doesn't actually answer the question, "
    "tell the user you did not find that information in their documents — "
    "never answer from your own general knowledge instead. Always cite the "
    "source filename (and page, if given) from the tool result in your final "
    "answer.\n\n"
    "If the user asks what commands are available or how to use the bot, "
    "answer directly (do not call search_documents for this) by listing: "
    "send a .txt/.md/.docx/.pdf file to index it, /documents to list your "
    "uploaded documents, /delete <filename> to remove one, and /stats to see "
    "their usage report — tokens spent, documents indexed, and errors "
    "encountered."
)

_STEP_BUDGET_EXHAUSTED = (
    "I wasn't able to finish processing that — please try rephrasing your question."
)


async def run(
    llm, registry, session, user_text: str, max_steps: int = AGENT_MAX_STEPS, on_llm_usage=None
) -> str:
    for _ in range(max_steps):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + session.messages()
        assistant_message = await llm.chat(messages, tools=registry.schemas())

        if on_llm_usage is not None:
            on_llm_usage(
                assistant_message.get("prompt_tokens", 0),
                assistant_message.get("completion_tokens", 0),
            )

        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            return assistant_message.get("content", "")

        for call in tool_calls:
            result = await registry.invoke(call)
            session.append_tool_result(call, result)

    return _STEP_BUDGET_EXHAUSTED
