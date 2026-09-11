import pytest

from telegram_bot.handlers.commands import (
    DELETE_USAGE_MESSAGE,
    NO_DOCUMENTS_MESSAGE,
    handle_delete_command,
    handle_documents_command,
)
from userdocs.errors import SQLiteStoreError


class FakeMessage:
    def __init__(self, text="", user_id=1):
        self.text = text
        self.from_user = type("FakeUser", (), {"id": user_id})()
        self.sent = []

    async def answer(self, text):
        self.sent.append(text)


class FakeStore:
    def __init__(self, documents=None, delete_result=True, delete_error=None):
        self._documents = documents or {}
        self._delete_result = delete_result
        self._delete_error = delete_error
        self.delete_calls = []

    def list_documents(self, user_id):
        return self._documents.get(user_id, [])

    def delete_document(self, user_id, filename):
        self.delete_calls.append((user_id, filename))
        if self._delete_error is not None:
            raise self._delete_error
        return self._delete_result


@pytest.mark.asyncio
async def test_documents_command_lists_numbered_filenames():
    store = FakeStore(documents={1: [
        {"id": 1, "filename": "employee_handbook.pdf", "file_type": ".pdf", "created_at": "t1"},
        {"id": 2, "filename": "vacation_policy.pdf", "file_type": ".pdf", "created_at": "t2"},
        {"id": 3, "filename": "benefits.md", "file_type": ".md", "created_at": "t3"},
    ]})
    message = FakeMessage(user_id=1)

    await handle_documents_command(message, store)

    reply = message.sent[0]
    assert "1. employee_handbook.pdf" in reply
    assert "2. vacation_policy.pdf" in reply
    assert "3. benefits.md" in reply


@pytest.mark.asyncio
async def test_documents_command_reports_empty_list():
    message = FakeMessage(user_id=1)
    await handle_documents_command(message, FakeStore())
    assert message.sent == [NO_DOCUMENTS_MESSAGE]


@pytest.mark.asyncio
async def test_documents_command_only_lists_the_senders_documents():
    store = FakeStore(documents={
        1: [{"id": 1, "filename": "mine.txt", "file_type": ".txt", "created_at": "t"}],
        2: [{"id": 2, "filename": "theirs.txt", "file_type": ".txt", "created_at": "t"}],
    })
    message = FakeMessage(user_id=1)

    await handle_documents_command(message, store)

    assert "mine.txt" in message.sent[0]
    assert "theirs.txt" not in message.sent[0]


@pytest.mark.asyncio
async def test_delete_command_confirms_a_successful_delete():
    store = FakeStore(delete_result=True)
    message = FakeMessage(text="/delete vacation_policy.pdf", user_id=7)

    await handle_delete_command(message, store)

    assert store.delete_calls == [(7, "vacation_policy.pdf")]
    assert "vacation_policy.pdf" in message.sent[0]
    assert message.sent[0].startswith("✅")


@pytest.mark.asyncio
async def test_delete_command_reports_a_missing_document():
    store = FakeStore(delete_result=False)
    message = FakeMessage(text="/delete nope.pdf", user_id=7)

    await handle_delete_command(message, store)

    assert message.sent[0].startswith("❌")
    assert "nope.pdf" in message.sent[0]


@pytest.mark.asyncio
async def test_delete_command_without_a_filename_shows_usage():
    store = FakeStore()
    message = FakeMessage(text="/delete", user_id=7)

    await handle_delete_command(message, store)

    assert message.sent == [DELETE_USAGE_MESSAGE]
    assert store.delete_calls == []


@pytest.mark.asyncio
async def test_delete_command_handles_a_storage_error_gracefully():
    store = FakeStore(delete_error=SQLiteStoreError("db is locked"))
    message = FakeMessage(text="/delete policy.pdf", user_id=7)

    await handle_delete_command(message, store)

    assert not any("db is locked" in text for text in message.sent)
    assert message.sent[0].startswith("❌")


@pytest.mark.asyncio
async def test_documents_command_handles_a_storage_error_gracefully():
    class BoomStore:
        def list_documents(self, user_id):
            raise SQLiteStoreError("db is locked")

    message = FakeMessage(user_id=1)
    await handle_documents_command(message, BoomStore())

    assert not any("db is locked" in text for text in message.sent)
    assert message.sent[0].startswith("❌")
