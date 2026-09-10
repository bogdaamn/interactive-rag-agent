"""Bot bootstrap and long-polling entrypoint. See spec/v3/SPEC.md §10.

Handler registration order matters: aiogram dispatches to the first matching
handler, so the command routes are registered before the catch-all text route.

Dependencies (store, llm, sessions) are closed over by the route functions
rather than pulled from module globals, which is what lets build_dispatcher be
tested with None placeholders and no bot token.
"""

import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command

from telegram_bot.config import load_config
from telegram_bot.handlers.chat import handle_text
from telegram_bot.handlers.commands import handle_delete_command, handle_documents_command
from telegram_bot.handlers.documents import handle_document
from telegram_bot.llm_client import OllamaClient
from telegram_bot.middleware import AuthMiddleware
from telegram_bot.session import SessionStore
from userdocs.store import UserDocsStore

logger = logging.getLogger(__name__)


def build_dispatcher(store, llm, sessions, allowed_user_ids) -> Dispatcher:
    dispatcher = Dispatcher()
    dispatcher.message.middleware(AuthMiddleware(allowed_user_ids))

    @dispatcher.message(Command("documents"))
    async def documents_route(message):
        await handle_documents_command(message, store)

    @dispatcher.message(Command("delete"))
    async def delete_route(message):
        await handle_delete_command(message, store)

    @dispatcher.message(F.document)
    async def document_route(message):
        await handle_document(message, store)

    @dispatcher.message(F.text)
    async def text_route(message):
        await handle_text(message, store, llm, sessions)

    return dispatcher


async def run_bot() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()

    store = UserDocsStore(config.userdocs_db_path)
    sessions = SessionStore(config.userdocs_db_path)
    llm = OllamaClient(
        base_url=config.ollama_base_url,
        model=config.ollama_model,
        timeout_seconds=config.llm_timeout_seconds,
    )
    bot = Bot(token=config.bot_token)
    dispatcher = build_dispatcher(store, llm, sessions, config.allowed_user_ids)

    try:
        await dispatcher.start_polling(bot)
    finally:
        await llm.close()
        store.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(run_bot())
