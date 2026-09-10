import pytest

from telegram_bot.middleware import AuthMiddleware


class FakeEvent:
    def __init__(self, user_id=1):
        self.from_user = type("FakeUser", (), {"id": user_id})()


@pytest.mark.asyncio
async def test_middleware_calls_handler_for_an_allowed_user():
    called = []

    async def handler(event, data):
        called.append(event)
        return "handled"

    middleware = AuthMiddleware(frozenset({1, 2}))
    result = await middleware(handler, FakeEvent(user_id=1), {})

    assert result == "handled"
    assert len(called) == 1


@pytest.mark.asyncio
async def test_middleware_blocks_a_disallowed_user():
    called = []

    async def handler(event, data):
        called.append(event)

    middleware = AuthMiddleware(frozenset({1, 2}))
    result = await middleware(handler, FakeEvent(user_id=99), {})

    assert result is None
    assert called == []


@pytest.mark.asyncio
async def test_middleware_allows_everyone_when_no_allowlist_configured():
    called = []

    async def handler(event, data):
        called.append(event)
        return "handled"

    middleware = AuthMiddleware(frozenset())
    result = await middleware(handler, FakeEvent(user_id=12345), {})

    assert result == "handled"
    assert len(called) == 1


@pytest.mark.asyncio
async def test_middleware_blocks_an_event_with_no_sender_when_an_allowlist_exists():
    called = []

    async def handler(event, data):
        called.append(event)

    class SenderlessEvent:
        from_user = None

    middleware = AuthMiddleware(frozenset({1}))
    result = await middleware(handler, SenderlessEvent(), {})

    assert result is None
    assert called == []
