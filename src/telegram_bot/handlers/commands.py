"""/documents and /delete command handlers. See spec/v3/SPEC.md §10.2.

Both handlers scope every store call to message.from_user.id, so a user can
neither list nor delete another user's documents even if they know the exact
filename (assignment §10 — the store enforces this too, but keeping the
handler explicit means the isolation isn't reliant on one layer alone).
"""

import logging

from telegram_bot.errors import message_for
from userdocs.errors import SQLiteStoreError

logger = logging.getLogger(__name__)

NO_DOCUMENTS_MESSAGE = "📚 You haven't uploaded any documents yet."
DELETE_USAGE_MESSAGE = "Usage: /delete <filename>\n\nExample: /delete vacation_policy.pdf"


async def handle_documents_command(message, store) -> None:
    try:
        documents = store.list_documents(message.from_user.id)
    except SQLiteStoreError as exc:
        logger.warning("Could not list documents for user %s: %s", message.from_user.id, exc)
        await message.answer(message_for(exc))
        return

    if not documents:
        await message.answer(NO_DOCUMENTS_MESSAGE)
        return

    lines = ["📚 Your documents:", ""]
    for index, document in enumerate(documents, start=1):
        lines.append(f"{index}. {document['filename']}")
    await message.answer("\n".join(lines))


async def handle_delete_command(message, store) -> None:
    filename = (message.text or "").removeprefix("/delete").strip()
    if not filename:
        await message.answer(DELETE_USAGE_MESSAGE)
        return

    try:
        deleted = store.delete_document(message.from_user.id, filename)
    except SQLiteStoreError as exc:
        logger.warning("Could not delete %s for user %s: %s", filename, message.from_user.id, exc)
        await message.answer(message_for(exc))
        return

    if deleted:
        await message.answer(f"✅ Deleted {filename}.")
    else:
        await message.answer(f"❌ No document named {filename} found.")
