"""Environment-driven bot config. See spec/v3/SPEC.md §11.2's model decision.

Follows ../telegram-bot/config.py's approach: a frozen dataclass built once at
startup, with a hard ConfigError on anything invalid rather than a silent
default or clamp — a bot that starts with a subtly wrong config is worse than
one that refuses to start.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    bot_token: str
    allowed_user_ids: frozenset
    ollama_base_url: str
    ollama_model: str
    llm_timeout_seconds: float
    userdocs_db_path: str


def _parse_allowed_user_ids(raw: str) -> frozenset:
    if not raw.strip():
        return frozenset()
    ids = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            ids.add(int(token))
        except ValueError as exc:
            raise ConfigError(
                f"ALLOWED_USER_IDS contains a non-numeric entry: {token!r}"
            ) from exc
    return frozenset(ids)


def load_config(env: dict | None = None) -> Config:
    if env is None:
        load_dotenv()
        env = dict(os.environ)

    bot_token = env.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not bot_token:
        raise ConfigError("TELEGRAM_BOT_TOKEN is required but was not set")

    raw_timeout = env.get("LLM_TIMEOUT_SECONDS", "120")
    try:
        llm_timeout_seconds = float(raw_timeout)
    except ValueError as exc:
        raise ConfigError(f"LLM_TIMEOUT_SECONDS is not a number: {raw_timeout!r}") from exc
    if llm_timeout_seconds <= 0:
        raise ConfigError(f"LLM_TIMEOUT_SECONDS must be positive, got {llm_timeout_seconds}")

    return Config(
        bot_token=bot_token,
        allowed_user_ids=_parse_allowed_user_ids(env.get("ALLOWED_USER_IDS", "")),
        ollama_base_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=env.get("OLLAMA_MODEL", "qwen2.5:7b"),
        llm_timeout_seconds=llm_timeout_seconds,
        userdocs_db_path=env.get("USERDOCS_DB_PATH", "userdocs.db"),
    )
