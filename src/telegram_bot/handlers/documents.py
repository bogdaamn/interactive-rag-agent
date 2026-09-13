"""Document upload handler. See spec/v3/SPEC.md §10.1.

The two scripted messages below follow the assignment's §1 user scenario
(translated to English for consistency with the rest of the bot's copy).
"""

import logging

from telegram_bot.errors import INGEST_ERRORS, message_for
from userdocs.pipeline import ingest_document_async

logger = logging.getLogger(__name__)

RECEIVED_MESSAGE = "📄 Document received.\n\nStarting processing..."
READY_MESSAGE = "✅ Document is ready.\n\nYou can now ask questions about the document."


async def handle_document(message, store, stats) -> None:
    await message.answer(RECEIVED_MESSAGE)

    file = await message.bot.get_file(message.document.file_id)
    downloaded = await message.bot.download_file(file.file_path)
    raw_bytes = downloaded.read()

    progress_message = await message.answer("⏳ Starting...")

    async def on_progress(text: str) -> None:
        try:
            await progress_message.edit_text(text)
        except Exception as exc:
            # Telegram rejects an unchanged-text edit and rate-limits rapid
            # edits (error category #10). A progress update is cosmetic — never
            # let it abort the actual indexing work.
            logger.warning("Could not update progress message: %s", exc)

    try:
        result = await ingest_document_async(
            store,
            message.from_user.id,
            message.document.file_name,
            raw_bytes,
            on_progress,
        )
    except INGEST_ERRORS as exc:
        logger.warning(
            "Ingestion failed for user %s, file %s: %s",
            message.from_user.id,
            message.document.file_name,
            exc,
        )
        stats.record_error(message.from_user.id, type(exc).__name__)
        await message.answer(message_for(exc))
        return

    logger.info(
        "Indexed %s for user %s (%s chunks)",
        result.filename,
        message.from_user.id,
        result.chunk_count,
    )
    stats.record_ingestion(message.from_user.id, result.chunk_count)
    await message.answer(READY_MESSAGE)
