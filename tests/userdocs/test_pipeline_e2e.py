"""End-to-end test (assignment §15's fifth level):

    document -> index -> question -> retrieval -> LLM -> answer

Real extraction, real chunking, real embeddings, real SQLite + sqlite-vec. Only
the LLM is faked, so the suite needs no running Ollama — the same allowance the
assignment makes for the Telegram API, extended to the LLM.
"""

import pytest

from tests.fixtures import CORPUS_DIR
from userdocs.agent import run as agent_run
from userdocs.pipeline import ingest_document
from userdocs.store import UserDocsStore
from userdocs.tool_registry import ToolRegistry
from userdocs.tools import NO_RESULTS_MESSAGE, build_search_documents_tool

pytestmark = pytest.mark.slow


@pytest.fixture
def ingested_store(tmp_path):
    """A real store with the full fixture corpus ingested for user 1."""
    store = UserDocsStore(str(tmp_path / "e2e.db"))
    for path in sorted(CORPUS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".docx", ".pdf"}:
            ingest_document(store, user_id=1, filename=path.name, raw_bytes=path.read_bytes())
    yield store
    store.close()


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.seen_messages = []

    async def chat(self, messages, tools):
        self.seen_messages.append([dict(m) for m in messages])
        return self._responses.pop(0)


def test_ingesting_the_corpus_creates_documents_and_chunks(ingested_store):
    documents = ingested_store.list_documents(user_id=1)
    assert len(documents) == 4

    total_chunks = ingested_store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert total_chunks >= 4


@pytest.mark.asyncio
async def test_question_retrieves_the_correct_document_and_cites_it(ingested_store):
    tool = build_search_documents_tool(ingested_store, user_id=1)
    result = await tool.handler(query="How many paid vacation days do employees get?")

    assert "vacation_policy.pdf" in result
    assert "25" in result
    assert "[Source: vacation_policy.pdf" in result


@pytest.mark.asyncio
async def test_pdf_source_attribution_includes_a_page_number(ingested_store):
    tool = build_search_documents_tool(ingested_store, user_id=1)
    result = await tool.handler(query="Can unused vacation days be carried over?")

    assert "vacation_policy.pdf" in result
    assert "page" in result


@pytest.mark.asyncio
async def test_non_pdf_source_attribution_has_no_page_number(ingested_store):
    tool = build_search_documents_tool(ingested_store, user_id=1)
    result = await tool.handler(query="What is the monthly wellness stipend?")

    assert "[Source: benefits.md]" in result


@pytest.mark.asyncio
async def test_unanswerable_question_returns_the_no_results_message(ingested_store):
    tool = build_search_documents_tool(ingested_store, user_id=1)
    result = await tool.handler(
        query="What is the airspeed velocity of an unladen swallow in kilometres per hour?"
    )

    assert result == NO_RESULTS_MESSAGE


@pytest.mark.asyncio
async def test_full_agent_turn_produces_an_answer_grounded_in_the_tool_result(ingested_store):
    """The complete chain: the agent decides to call search_documents, the real
    retrieval runs against the real index, and the tool output is fed back for
    the final answer."""

    class FakeSession:
        def __init__(self):
            self._messages = []

        def messages(self):
            return list(self._messages)

        def append_user_message(self, text):
            self._messages.append({"role": "user", "content": text})

        def append_assistant_message(self, text):
            self._messages.append({"role": "assistant", "content": text})

        def append_tool_result(self, call, result):
            self._messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
            self._messages.append({"role": "tool", "content": result})

    question = "How many paid vacation days do employees get?"
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [
            {"function": {"name": "search_documents", "arguments": {"query": question}}}
        ]},
        {"content": "Employees get 25 days. Source: vacation_policy.pdf", "tool_calls": []},
    ])

    session = FakeSession()
    session.append_user_message(question)
    registry = ToolRegistry([build_search_documents_tool(ingested_store, user_id=1)])

    answer = await agent_run(llm, registry, session, question)

    assert answer == "Employees get 25 days. Source: vacation_policy.pdf"
    # the real retrieval output reached the LLM before it composed that answer
    second_prompt_contents = [m.get("content", "") for m in llm.seen_messages[1]]
    assert any("vacation_policy.pdf" in c for c in second_prompt_contents)
    assert any("25" in c for c in second_prompt_contents)


@pytest.mark.asyncio
async def test_user_isolation_holds_through_the_real_pipeline(tmp_path):
    """Assignment §15's fourth level, exercised end-to-end rather than at the
    store layer: user 2 uploads nothing, so their search finds nothing — even
    though user 1's indexed documents answer the question perfectly."""
    store = UserDocsStore(str(tmp_path / "isolation.db"))
    ingest_document(
        store,
        user_id=1,
        filename="vacation_policy.pdf",
        raw_bytes=(CORPUS_DIR / "vacation_policy.pdf").read_bytes(),
    )

    tool_for_user_2 = build_search_documents_tool(store, user_id=2)
    result = await tool_for_user_2.handler(query="How many paid vacation days do employees get?")

    assert result == NO_RESULTS_MESSAGE
    assert "vacation_policy.pdf" not in result
    store.close()


@pytest.mark.asyncio
async def test_deleted_document_no_longer_participates_in_search(ingested_store):
    """Assignment §11: after /delete, the document must be gone from RAG search."""
    tool = build_search_documents_tool(ingested_store, user_id=1)

    before = await tool.handler(query="How many paid vacation days do employees get?")
    assert "vacation_policy.pdf" in before

    assert ingested_store.delete_document(user_id=1, filename="vacation_policy.pdf") is True

    after = await tool.handler(query="How many paid vacation days do employees get?")
    assert "vacation_policy.pdf" not in after
