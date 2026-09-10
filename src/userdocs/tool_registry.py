"""Tool definition and dispatch for the agent loop.

Adapted from ../telegram-bot/tools/__init__.py (see spec/v3/SPEC.md §2's reuse
table). Handler exceptions become "ERROR: ..." strings the LLM can read and
react to, rather than exceptions that abort the whole turn.
"""

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Optional[Callable[..., Awaitable[str]]]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list):
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list:
        return [tool.schema() for tool in self._tools.values()]

    async def invoke(self, call: dict) -> str:
        function = call.get("function", {})
        name = function.get("name")
        tool = self._tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool {name!r}"

        arguments: Any = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return f"ERROR: could not parse arguments for {name!r}: {exc}"

        try:
            return await tool.handler(**arguments)
        except Exception as exc:
            return f"ERROR: {exc}"
