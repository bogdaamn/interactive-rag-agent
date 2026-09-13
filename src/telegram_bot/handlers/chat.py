"""Plain-text message handler. See spec/v3/SPEC.md §10.3.

The tool registry is built fresh per message, with the sender's user_id baked
into the search_documents closure — the LLM never sees a user_id parameter it
could change (spec §11.1).
"""

import logging

from telegram_bot.errors import message_for
from telegram_bot.llm_errors import LLMError
from userdocs.agent import run as agent_run
from userdocs.config import AGENT_MAX_STEPS
from userdocs.tool_registry import ToolRegistry
from userdocs.tools import build_search_documents_tool

logger = logging.getLogger(__name__)


async def handle_text(message, store, llm, sessions, stats) -> None:
    user_id = message.from_user.id
    session = sessions.get_or_create(user_id)
    session.append_user_message(message.text)

    registry = ToolRegistry([build_search_documents_tool(store, user_id)])

    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def _accumulate_usage(prompt_tokens: int, completion_tokens: int) -> None:
        usage["prompt_tokens"] += prompt_tokens
        usage["completion_tokens"] += completion_tokens

    try:
        answer = await agent_run(
            llm,
            registry,
            session,
            message.text,
            max_steps=AGENT_MAX_STEPS,
            on_llm_usage=_accumulate_usage,
        )
    except LLMError as exc:
        logger.warning("Agent run failed for user %s: %s", user_id, exc)
        stats.record_error(user_id, type(exc).__name__)
        await message.answer(message_for(exc))
        return

    stats.record_turn(user_id, usage["prompt_tokens"], usage["completion_tokens"])
    session.append_assistant_message(answer)
    await message.answer(answer)
