import httpx
import pytest

from telegram_bot.llm_client import OllamaClient
from telegram_bot.llm_errors import LLMError, LLMTimeoutError


def _client(transport):
    client = OllamaClient(base_url="http://ollama.test", model="test-model", timeout_seconds=5.0)
    client._http = httpx.AsyncClient(transport=transport, timeout=5.0)
    return client


@pytest.mark.asyncio
async def test_chat_returns_content_and_empty_tool_calls():
    def handler(request):
        return httpx.Response(200, json={"message": {"content": "Hello!"}})

    client = _client(httpx.MockTransport(handler))
    result = await client.chat([{"role": "user", "content": "hi"}], tools=[])

    assert result == {"content": "Hello!", "tool_calls": []}


@pytest.mark.asyncio
async def test_chat_sends_model_messages_and_tools_and_disables_streaming():
    captured = {}

    def handler(request):
        import json as _json

        captured.update(_json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    client = _client(httpx.MockTransport(handler))
    tools = [{"type": "function", "function": {"name": "t"}}]
    await client.chat([{"role": "user", "content": "hi"}], tools=tools)

    assert captured["model"] == "test-model"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["tools"] == tools
    assert captured["stream"] is False


@pytest.mark.asyncio
async def test_chat_normalizes_tool_call_arguments_given_as_a_json_string():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "search_documents", "arguments": '{"query": "vacation"}'}}
                    ],
                }
            },
        )

    client = _client(httpx.MockTransport(handler))
    result = await client.chat([], tools=[])

    assert result["tool_calls"][0]["function"]["arguments"] == {"query": "vacation"}


@pytest.mark.asyncio
async def test_chat_raises_llm_timeout_error_on_read_timeout():
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(LLMTimeoutError):
        await client.chat([], tools=[])


@pytest.mark.asyncio
async def test_chat_raises_llm_error_on_connect_failure():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(LLMError):
        await client.chat([], tools=[])


@pytest.mark.asyncio
async def test_chat_raises_llm_error_on_http_500():
    def handler(request):
        return httpx.Response(500, text="internal error")

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(LLMError):
        await client.chat([], tools=[])


@pytest.mark.asyncio
async def test_chat_raises_llm_error_on_unparseable_body():
    def handler(request):
        return httpx.Response(200, text="not json at all")

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(LLMError):
        await client.chat([], tools=[])


@pytest.mark.asyncio
async def test_llm_timeout_error_is_an_llm_error():
    assert issubclass(LLMTimeoutError, LLMError)
