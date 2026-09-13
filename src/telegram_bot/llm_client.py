"""Ollama chat client with native tool-calling. See spec/v3/SPEC.md §11.2.

Adapted from ../telegram-bot/llm_client.py. Uses /api/chat with a `tools` array
(the model's own tool-calling format) rather than /api/generate with a
hand-written JSON convention — assignment §8 wants the *model* deciding when to
search, and this is the interface that lets it.

Every httpx failure mode is collapsed into LLMError (or LLMTimeoutError for the
two timeout variants) so callers never need to know httpx exists.
"""

import json

import httpx

from telegram_bot.llm_errors import LLMError, LLMTimeoutError


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: float):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._http = httpx.AsyncClient(timeout=timeout_seconds)

    @staticmethod
    def _normalize_tool_calls(tool_calls: list) -> list:
        """Ollama returns tool-call arguments as either a decoded object or a
        JSON string depending on the model. Normalize to a dict so downstream
        code has one shape to handle."""
        normalized = []
        for call in tool_calls:
            function = dict(call.get("function", {}))
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    function["arguments"] = {}
            normalized.append({**call, "function": function})
        return normalized

    async def chat(self, messages: list, tools: list) -> dict:
        payload = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "stream": False,
        }
        try:
            response = await self._http.post(f"{self._base_url}/api/chat", json=payload)
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            raise LLMTimeoutError(f"LLM request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise LLMError(f"Could not connect to the LLM: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        if response.status_code >= 400:
            raise LLMError(f"LLM returned HTTP {response.status_code}")

        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(f"LLM returned an unparseable body: {exc}") from exc

        message = body.get("message")
        if not isinstance(message, dict):
            raise LLMError("LLM response contained no message object")

        return {
            "content": message.get("content", ""),
            "tool_calls": self._normalize_tool_calls(message.get("tool_calls") or []),
            "prompt_tokens": body.get("prompt_eval_count", 0),
            "completion_tokens": body.get("eval_count", 0),
        }

    async def close(self) -> None:
        await self._http.aclose()
