from telegram_bot.session import SessionStore
from userdocs.config import CONVERSATION_HISTORY_TURNS


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


def test_messages_returns_at_most_the_configured_window(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    session = store.get_or_create(user_id=1)

    for turn in range(CONVERSATION_HISTORY_TURNS + 4):
        session.append_user_message(f"question {turn}")
        session.append_assistant_message(f"answer {turn}")

    messages = session.messages()
    assert len(messages) == CONVERSATION_HISTORY_TURNS * 2
    store.close()


def test_messages_window_keeps_the_most_recent_turns_in_order(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    session = store.get_or_create(user_id=1)

    for turn in range(CONVERSATION_HISTORY_TURNS + 4):
        session.append_user_message(f"question {turn}")
        session.append_assistant_message(f"answer {turn}")

    messages = session.messages()
    last_turn = CONVERSATION_HISTORY_TURNS + 3

    # oldest turns are dropped, newest are kept, chronological order preserved
    assert messages[-1] == {"role": "assistant", "content": f"answer {last_turn}"}
    assert messages[-2] == {"role": "user", "content": f"question {last_turn}"}
    assert not any(m["content"] == "question 0" for m in messages)
    store.close()


def test_messages_window_does_not_truncate_a_short_history(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    session = store.get_or_create(user_id=1)
    session.append_user_message("only question")
    session.append_assistant_message("only answer")

    assert len(session.messages()) == 2
    store.close()


_CALL = {"function": {"name": "search_documents", "arguments": {"query": "vacation"}}}


def test_append_tool_result_makes_the_result_visible_to_the_next_llm_call(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    session = store.get_or_create(user_id=1)

    session.append_user_message("How many vacation days?")
    session.append_tool_result(_CALL, "[Source: policy.pdf, page 3]\n25 days")

    contents = [m["content"] for m in session.messages()]
    assert any("25 days" in c for c in contents)
    store.close()


def test_append_tool_result_pairs_each_tool_message_with_an_assistant_tool_call(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    session = store.get_or_create(user_id=1)

    session.append_user_message("How many vacation days?")
    session.append_tool_result(_CALL, "25 days")

    messages = session.messages()
    tool_index = next(i for i, m in enumerate(messages) if m["role"] == "tool")
    preceding = messages[tool_index - 1]
    assert preceding["role"] == "assistant"
    assert preceding["tool_calls"] == [_CALL]
    store.close()


def test_tool_messages_are_not_persisted(tmp_path):
    db_path = str(tmp_path / "test.db")

    store = SessionStore(db_path)
    session = store.get_or_create(user_id=1)
    session.append_user_message("How many vacation days?")
    session.append_tool_result(_CALL, "25 days")
    session.append_assistant_message("You get 25 days. Source: policy.pdf")
    store.close()

    store2 = SessionStore(db_path)
    roles = [m["role"] for m in store2.get_or_create(user_id=1).messages()]
    assert roles == ["user", "assistant"]
    assert "tool" not in roles
    store2.close()


def test_append_user_message_clears_the_previous_turns_tool_messages(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    session = store.get_or_create(user_id=1)

    session.append_user_message("How many vacation days?")
    session.append_tool_result(_CALL, "stale tool output from turn one")
    session.append_assistant_message("25 days.")

    session.append_user_message("Can I carry them over?")

    contents = [m["content"] for m in session.messages()]
    assert not any("stale tool output" in c for c in contents)
    # ...but the conversational turn itself is still there, which is the whole
    # point of conversation-aware RAG
    assert any("25 days." in c for c in contents)
    store.close()
