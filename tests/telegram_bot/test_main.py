from telegram_bot.main import build_dispatcher


def _handler_names(dispatcher):
    return [handler.callback.__name__ for handler in dispatcher.message.handlers]


def test_build_dispatcher_registers_all_four_message_routes():
    dispatcher = build_dispatcher(store=None, llm=None, sessions=None, allowed_user_ids=frozenset())
    names = _handler_names(dispatcher)

    assert "documents_route" in names
    assert "delete_route" in names
    assert "document_route" in names
    assert "text_route" in names


def test_command_routes_are_registered_before_the_catch_all_text_route():
    """aiogram checks handlers in registration order and dispatches to the
    first whose filters match. F.text matches "/documents" too, so if the
    catch-all text route were registered first it would swallow both commands
    and they would silently never run."""
    dispatcher = build_dispatcher(store=None, llm=None, sessions=None, allowed_user_ids=frozenset())
    names = _handler_names(dispatcher)

    assert names.index("documents_route") < names.index("text_route")
    assert names.index("delete_route") < names.index("text_route")


def test_build_dispatcher_registers_an_auth_middleware():
    from telegram_bot.middleware import AuthMiddleware

    dispatcher = build_dispatcher(store=None, llm=None, sessions=None, allowed_user_ids=frozenset({1}))
    registered = list(dispatcher.message.middleware)

    assert any(isinstance(middleware, AuthMiddleware) for middleware in registered)
