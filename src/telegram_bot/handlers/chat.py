"""Plain-text message handler. See spec/v3/SPEC.md §10.3, §13.

The tool registry is built fresh per message, with the sender's user_id baked
into the search_documents closure — the LLM never sees a user_id parameter it
could change (spec §11.1).

Source attribution is appended here, not left to the LLM's own prose: every
chunk search_documents actually retrieved this turn is collected via
on_sources, and a deduped citation footer is appended to the final answer —
guaranteed correct regardless of what the model says (spec §13).

search_documents is also run once, deterministically, before the model's
first turn (see _prefetch_search) — small local models observably skip
calling the tool even for clearly document-answerable questions, so the
*whether to search* decision is no longer left entirely to the model. The
model still decides how to answer, and remains free to call the tool again
with a different query if it wants to.
"""

import logging

from telegram_bot.errors import message_for
from telegram_bot.llm_errors import LLMError
from userdocs.agent import run as agent_run
from userdocs.config import AGENT_MAX_STEPS
from userdocs.errors import SQLiteStoreError
from userdocs.tool_registry import ToolRegistry
from userdocs.tools import build_search_documents_tool, format_citation_footer

logger = logging.getLogger(__name__)


async def _prefetch_search(registry: ToolRegistry, session, user_text: str) -> None:
    """Run search_documents once, unconditionally, and feed the result into
    the session as if the model had already called it this turn.

    This is the deterministic backstop for the agent-loop nudge in
    userdocs/agent.py: that nudge reduces but doesn't eliminate the model
    skipping the tool call on hard phrasings (verified against the live
    qwen2.5:7b/Ollama stack — advice-phrased questions like "how long should
    employee work before taking any vacations?" still slipped through the
    nudge in roughly 2/5 tries). Running the search here removes the model's
    discretion over *whether* to search at all, while leaving *how to
    answer* — and whether to search again with a refined query — up to it.
    """
    call = {"function": {"name": "search_documents", "arguments": {"query": user_text}}}
    result = await registry.invoke(call)
    session.append_tool_result(call, result)


async def handle_text(message, store, llm, sessions, stats) -> None:
    user_id = message.from_user.id
    session = sessions.get_or_create(user_id)

    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    sources = []

    def _accumulate_usage(prompt_tokens: int, completion_tokens: int) -> None:
        usage["prompt_tokens"] += prompt_tokens
        usage["completion_tokens"] += completion_tokens

    try:
        session.append_user_message(message.text)

        registry = ToolRegistry(
            [build_search_documents_tool(store, user_id, on_sources=sources.extend)]
        )

        await _prefetch_search(registry, session, message.text)

        answer = await agent_run(
            llm,
            registry,
            session,
            message.text,
            max_steps=AGENT_MAX_STEPS,
            on_llm_usage=_accumulate_usage,
        )

        if sources:
            answer = f"{answer}\n\n{format_citation_footer(sources)}"

        session.append_assistant_message(answer)
        stats.record_turn(user_id, usage["prompt_tokens"], usage["completion_tokens"])
    except (LLMError, SQLiteStoreError) as exc:
        logger.warning("Agent run failed for user %s: %s", user_id, exc)
        stats.record_error(user_id, type(exc).__name__)
        await message.answer(message_for(exc))
        return

    await message.answer(answer)
