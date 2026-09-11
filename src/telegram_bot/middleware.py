"""Allowlist auth middleware, adapted from ../telegram-bot/main.py.

An empty allowlist means no allowlist is configured, so everyone is allowed —
this bot's isolation guarantee is per-user data separation (spec §10), not
access control, so refusing every user by default would be wrong. When an
allowlist IS configured, an event with no identifiable sender is blocked:
there's no user_id to check it against.
"""

import logging

logger = logging.getLogger(__name__)


class AuthMiddleware:
    def __init__(self, allowed_user_ids: frozenset):
        self._allowed_user_ids = allowed_user_ids

    async def __call__(self, handler, event, data):
        if not self._allowed_user_ids:
            return await handler(event, data)

        from_user = getattr(event, "from_user", None)
        user_id = getattr(from_user, "id", None) if from_user is not None else None

        if user_id is None or user_id not in self._allowed_user_ids:
            logger.info("Blocked message from unauthorized user_id=%s", user_id)
            return None

        return await handler(event, data)
