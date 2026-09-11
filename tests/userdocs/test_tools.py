import asyncio

from userdocs.retrieve import RetrievedChunk
from userdocs.tools import NO_RESULTS_MESSAGE, build_search_documents_tool


def test_tool_schema_has_no_user_id_parameter():
    """user_id is bound in trusted code, never exposed to the LLM — otherwise a
    model could request another user's documents (spec §10, §11.1)."""
    tool = build_search_documents_tool(store=None, user_id=42)
    properties = tool.schema()["function"]["parameters"]["properties"]
    assert "query" in properties
    assert "user_id" not in properties


def test_tool_handler_formats_pdf_source_with_page(monkeypatch):
    import userdocs.tools as tools_module

    monkeypatch.setattr(
        tools_module,
        "retrieve",
        lambda store, user_id, query: [
            RetrievedChunk(
                text="Employees receive 25 vacation days.",
                filename="vacation_policy.pdf",
                page=12,
                chunk_index=37,
                score=0.88,
            )
        ],
    )

    tool = build_search_documents_tool(store=None, user_id=1)
    result = asyncio.run(tool.handler(query="how many vacation days?"))

    assert "[Source: vacation_policy.pdf, page 12]" in result
    assert "Employees receive 25 vacation days." in result


def test_tool_handler_formats_non_pdf_source_without_page(monkeypatch):
    import userdocs.tools as tools_module

    monkeypatch.setattr(
        tools_module,
        "retrieve",
        lambda store, user_id, query: [
            RetrievedChunk(
                text="Wellness stipend is $50/month.",
                filename="benefits.md",
                page=None,
                chunk_index=2,
                score=0.75,
            )
        ],
    )

    tool = build_search_documents_tool(store=None, user_id=1)
    result = asyncio.run(tool.handler(query="wellness stipend?"))

    assert "[Source: benefits.md]" in result
    assert "page" not in result


def test_tool_handler_returns_no_results_message_when_nothing_relevant(monkeypatch):
    import userdocs.tools as tools_module

    monkeypatch.setattr(tools_module, "retrieve", lambda store, user_id, query: [])

    tool = build_search_documents_tool(store=None, user_id=1)
    result = asyncio.run(tool.handler(query="parental leave?"))

    assert result == NO_RESULTS_MESSAGE


def test_tool_handler_passes_the_bound_user_id_to_retrieve(monkeypatch):
    import userdocs.tools as tools_module

    seen = {}

    def fake_retrieve(store, user_id, query):
        seen["user_id"] = user_id
        return []

    monkeypatch.setattr(tools_module, "retrieve", fake_retrieve)

    tool = build_search_documents_tool(store=None, user_id=777)
    asyncio.run(tool.handler(query="anything"))

    assert seen["user_id"] == 777
