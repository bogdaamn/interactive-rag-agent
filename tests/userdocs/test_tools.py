import asyncio

from userdocs.retrieve import RetrievedChunk
from userdocs.tools import (
    NO_RESULTS_MESSAGE,
    build_search_documents_tool,
    format_citation_footer,
)


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


def test_tool_handler_reports_retrieved_chunks_via_on_sources(monkeypatch):
    import userdocs.tools as tools_module

    chunk = RetrievedChunk(
        text="Employees receive 25 vacation days.",
        filename="vacation_policy.pdf",
        page=12,
        chunk_index=37,
        score=0.88,
    )
    monkeypatch.setattr(tools_module, "retrieve", lambda store, user_id, query: [chunk])

    seen = []
    tool = build_search_documents_tool(store=None, user_id=1, on_sources=seen.append)
    asyncio.run(tool.handler(query="how many vacation days?"))

    assert seen == [[chunk]]


def test_tool_handler_does_not_call_on_sources_when_nothing_relevant(monkeypatch):
    import userdocs.tools as tools_module

    monkeypatch.setattr(tools_module, "retrieve", lambda store, user_id, query: [])

    seen = []
    tool = build_search_documents_tool(store=None, user_id=1, on_sources=seen.append)
    asyncio.run(tool.handler(query="parental leave?"))

    assert seen == []


def test_format_citation_footer_uses_page_for_a_pdf_chunk():
    chunk = RetrievedChunk(
        text="irrelevant", filename="vacation_policy.pdf", page=12, chunk_index=37, score=0.9
    )

    assert format_citation_footer([chunk]) == "Source: vacation_policy.pdf, page 12"


def test_format_citation_footer_uses_a_1_indexed_chunk_number_without_a_page():
    chunk = RetrievedChunk(
        text="irrelevant", filename="benefits.md", page=None, chunk_index=2, score=0.9
    )

    assert format_citation_footer([chunk]) == "Source: benefits.md, chunk #3"


def test_format_citation_footer_dedupes_and_preserves_first_seen_order():
    a = RetrievedChunk(text="x", filename="a.pdf", page=1, chunk_index=0, score=0.9)
    b = RetrievedChunk(text="y", filename="b.md", page=None, chunk_index=0, score=0.9)
    a_again = RetrievedChunk(text="z", filename="a.pdf", page=1, chunk_index=0, score=0.5)

    footer = format_citation_footer([a, b, a_again])

    assert footer == "Source: a.pdf, page 1\nSource: b.md, chunk #1"
