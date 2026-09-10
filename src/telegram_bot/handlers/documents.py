"""Document upload handler. See spec/v3/SPEC.md §10.1.

The two scripted messages below are verbatim from the assignment's §1 user
scenario and must not be reworded.
"""

import logging

from userdocs.pipeline import ingest_document_async

logger = logging.getLogger(__name__)

RECEIVED_MESSAGE = "📄 Документ получен.\n\nНачинаю обработку..."
READY_MESSAGE = "✅ Документ готов.\n\nТеперь вы можете задавать вопросы по документу."


async def handle_document(message, store) -> None:
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

    result = await ingest_document_async(
        store,
        message.from_user.id,
        message.document.file_name,
        raw_bytes,
        on_progress,
    )
    logger.info(
        "Indexed %s for user %s (%s chunks)",
        result.filename,
        message.from_user.id,
        result.chunk_count,
    )
    await message.answer(READY_MESSAGE)
