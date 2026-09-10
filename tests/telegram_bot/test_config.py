import pytest

from telegram_bot.config import ConfigError, load_config

_MINIMAL_ENV = {"TELEGRAM_BOT_TOKEN": "123:abc"}


def test_load_config_reads_bot_token():
    config = load_config(_MINIMAL_ENV)
    assert config.bot_token == "123:abc"


def test_load_config_applies_documented_defaults():
    config = load_config(_MINIMAL_ENV)
    assert config.ollama_base_url == "http://localhost:11434"
    assert config.ollama_model == "qwen2.5:7b"
    assert config.llm_timeout_seconds == 120.0
    assert config.userdocs_db_path == "userdocs.db"
    assert config.allowed_user_ids == frozenset()


def test_load_config_parses_allowed_user_ids():
    config = load_config({**_MINIMAL_ENV, "ALLOWED_USER_IDS": "111, 222,333"})
    assert config.allowed_user_ids == frozenset({111, 222, 333})


def test_load_config_raises_when_bot_token_missing():
    with pytest.raises(ConfigError, match="TELEGRAM_BOT_TOKEN"):
        load_config({})


def test_load_config_raises_on_non_numeric_allowed_user_id():
    with pytest.raises(ConfigError, match="ALLOWED_USER_IDS"):
        load_config({**_MINIMAL_ENV, "ALLOWED_USER_IDS": "111,not-a-number"})


def test_load_config_raises_on_non_positive_timeout():
    with pytest.raises(ConfigError, match="LLM_TIMEOUT_SECONDS"):
        load_config({**_MINIMAL_ENV, "LLM_TIMEOUT_SECONDS": "0"})
