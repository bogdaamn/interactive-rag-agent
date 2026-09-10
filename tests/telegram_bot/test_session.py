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


def test_one_users_history_never_appears_in_anothers(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))

    session_a = store.get_or_create(user_id=1)
    session_a.append_user_message("What is my salary?")
    session_a.append_assistant_message("Your salary is confidential information X.")

    session_b = store.get_or_create(user_id=2)
    session_b.append_user_message("What are the office hours?")

    b_contents = [m["content"] for m in session_b.messages()]
    assert b_contents == ["What are the office hours?"]
    assert not any("confidential information X" in c for c in b_contents)
    store.close()


def test_a_new_users_session_is_empty_even_when_others_have_history(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))

    session_a = store.get_or_create(user_id=1)
    session_a.append_user_message("first user's question")
    session_a.append_assistant_message("first user's answer")

    assert store.get_or_create(user_id=2).messages() == []
    store.close()
