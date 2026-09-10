import pytest

from telegram_bot.handlers.documents import (
    READY_MESSAGE,
    RECEIVED_MESSAGE,
    handle_document,
)


class FakeSentMessage:
    """Stands in for the Message object message.answer() returns, so
    edit_text() calls can be observed."""

    def __init__(self, text, log):
        self.text = text
        self._log = log

    async def edit_text(self, text):
        self.text = text
        self._log.append(("edit", text))


class FakeBot:
    def __init__(self, file_bytes=b"document contents"):
        self._file_bytes = file_bytes

    async def get_file(self, file_id):
        return type("FakeFile", (), {"file_path": f"remote/{file_id}"})()

    async def download_file(self, file_path):
        import io

        return io.BytesIO(self._file_bytes)


class FakeDocument:
    def __init__(self, file_name="policy.txt", file_id="fid-1"):
        self.file_name = file_name
        self.file_id = file_id


class FakeMessage:
    def __init__(self, user_id=1, file_name="policy.txt", file_bytes=b"document contents"):
        self.from_user = type("FakeUser", (), {"id": user_id})()
        self.document = FakeDocument(file_name=file_name)
        self.bot = FakeBot(file_bytes=file_bytes)
        self.sent = []          # list of ("answer", text) / ("edit", text)

    async def answer(self, text):
        self.sent.append(("answer", text))
        return FakeSentMessage(text, self.sent)


@pytest.mark.asyncio
async def test_handle_document_sends_scripted_received_and_ready_messages(monkeypatch, tmp_path):
    import telegram_bot.handlers.documents as documents_module

    async def fake_ingest(store, user_id, filename, raw_bytes, on_progress=None):
        if on_progress is not None:
            await on_progress("⏳ Extracting text...")
        return type("R", (), {"document_id": 1, "filename": filename, "chunk_count": 3})()

    monkeypatch.setattr(documents_module, "ingest_document_async", fake_ingest)

    message = FakeMessage()
    await handle_document(message, store=None)

    answers = [text for kind, text in message.sent if kind == "answer"]
    assert answers[0] == RECEIVED_MESSAGE
    assert READY_MESSAGE in answers


@pytest.mark.asyncio
async def test_handle_document_passes_the_senders_user_id_to_the_pipeline(monkeypatch):
    import telegram_bot.handlers.documents as documents_module

    seen = {}

    async def fake_ingest(store, user_id, filename, raw_bytes, on_progress=None):
        seen["user_id"] = user_id
        seen["filename"] = filename
        seen["raw_bytes"] = raw_bytes
        return type("R", (), {"document_id": 1, "filename": filename, "chunk_count": 1})()

    monkeypatch.setattr(documents_module, "ingest_document_async", fake_ingest)

    message = FakeMessage(user_id=555, file_name="handbook.pdf", file_bytes=b"pdf bytes")
    await handle_document(message, store=None)

    assert seen["user_id"] == 555
    assert seen["filename"] == "handbook.pdf"
    assert seen["raw_bytes"] == b"pdf bytes"
