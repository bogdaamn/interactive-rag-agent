import asyncio

from userdocs.tool_registry import Tool, ToolRegistry


def _make_tool(handler):
    return Tool(
        name="echo",
        description="Echoes its input.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        handler=handler,
    )


def test_tool_schema_matches_ollama_function_format():
    tool = _make_tool(handler=None)
    assert tool.schema() == {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echoes its input.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }


def test_registry_invoke_calls_the_matching_handler():
    async def handler(text: str) -> str:
        return f"echoed: {text}"

    registry = ToolRegistry([_make_tool(handler)])
    result = asyncio.run(registry.invoke({"function": {"name": "echo", "arguments": {"text": "hi"}}}))
    assert result == "echoed: hi"


def test_registry_invoke_returns_error_string_for_unknown_tool():
    registry = ToolRegistry([])
    result = asyncio.run(registry.invoke({"function": {"name": "nope", "arguments": {}}}))
    assert result.startswith("ERROR:")
    assert "nope" in result


def test_registry_invoke_returns_error_string_instead_of_raising():
    async def handler(text: str) -> str:
        raise ValueError("handler blew up")

    registry = ToolRegistry([_make_tool(handler)])
    result = asyncio.run(registry.invoke({"function": {"name": "echo", "arguments": {"text": "hi"}}}))
    assert result.startswith("ERROR:")
    assert "handler blew up" in result


def test_registry_invoke_parses_json_string_arguments():
    """Ollama sometimes returns tool-call arguments as a JSON string rather
    than a decoded object — both forms must work."""

    async def handler(text: str) -> str:
        return f"echoed: {text}"

    registry = ToolRegistry([_make_tool(handler)])
    result = asyncio.run(
        registry.invoke({"function": {"name": "echo", "arguments": '{"text": "hi"}'}})
    )
    assert result == "echoed: hi"
