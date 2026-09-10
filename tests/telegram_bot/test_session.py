from telegram_bot.session import SessionStore


def test_append_and_read_back_a_single_turn(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    session = store.get_or_create(user_id=1)

    session.append_user_message("How many vacation days?")
    session.append_assistant_message("25 days.")

    assert session.messages() == [
        {"role": "user", "content": "How many vacation days?"},
        {"role": "assistant", "content": "25 days."},
    ]
    store.close()


def test_messages_persist_across_store_reopen(tmp_path):
    db_path = str(tmp_path / "test.db")

    store = SessionStore(db_path)
    session = store.get_or_create(user_id=1)
    session.append_user_message("How many vacation days?")
    session.append_assistant_message("25 days.")
    store.close()

    store2 = SessionStore(db_path)
    session2 = store2.get_or_create(user_id=1)
    assert session2.messages() == [
        {"role": "user", "content": "How many vacation days?"},
        {"role": "assistant", "content": "25 days."},
    ]
    store2.close()


def test_get_or_create_returns_an_empty_session_for_a_new_user(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    assert store.get_or_create(user_id=99).messages() == []
    store.close()


def test_get_or_create_returns_the_same_session_object_for_the_same_user(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    assert store.get_or_create(user_id=1) is store.get_or_create(user_id=1)
    store.close()
