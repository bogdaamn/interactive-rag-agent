"""In-memory per-user usage counters, surfaced via the /stats command. See
spec/v3/SPEC.md §20 — not persisted, resets to zero on every bot restart.
"""

from dataclasses import dataclass, field


@dataclass
class UserStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    turns: int = 0
    documents_indexed: int = 0
    chunks_indexed: int = 0
    errors: dict = field(default_factory=dict)


class UsageStats:
    def __init__(self):
        self._by_user: dict[int, UserStats] = {}

    def _get_or_create(self, user_id: int) -> UserStats:
        return self._by_user.setdefault(user_id, UserStats())

    def record_turn(self, user_id: int, prompt_tokens: int, completion_tokens: int) -> None:
        stats = self._get_or_create(user_id)
        stats.prompt_tokens += prompt_tokens
        stats.completion_tokens += completion_tokens
        stats.turns += 1

    def record_ingestion(self, user_id: int, chunk_count: int) -> None:
        stats = self._get_or_create(user_id)
        stats.documents_indexed += 1
        stats.chunks_indexed += chunk_count

    def record_error(self, user_id: int, category: str) -> None:
        stats = self._get_or_create(user_id)
        stats.errors[category] = stats.errors.get(category, 0) + 1

    def snapshot(self, user_id: int) -> UserStats:
        stats = self._by_user.get(user_id)
        if stats is None:
            return UserStats()
        return UserStats(
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            turns=stats.turns,
            documents_indexed=stats.documents_indexed,
            chunks_indexed=stats.chunks_indexed,
            errors=dict(stats.errors),
        )
