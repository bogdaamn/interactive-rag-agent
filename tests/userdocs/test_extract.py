import io
from pathlib import Path

from docx import Document as DocxDocument

from userdocs.errors import CorruptDocumentError, EmptyDocumentError
from userdocs.extract import extract_text

_FIXTURES = Path(__file__).parent / "fixtures"


def test_extract_txt_returns_decoded_text_with_no_pages():
    raw = "Employees get 25 vacation days per year.".encode("utf-8")
    result = extract_text("policy.txt", raw)
    assert result.text == "Employees get 25 vacation days per year."
    assert result.pages is None


def test_extract_md_returns_decoded_text_with_no_pages():
    raw = "# Benefits\n\nWellness stipend: $50/month.".encode("utf-8")
    result = extract_text("benefits.md", raw)
    assert result.text == "# Benefits\n\nWellness stipend: $50/month."
    assert result.pages is None


def _make_docx_bytes(paragraphs: list[str]) -> bytes:
    doc = DocxDocument()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_extract_docx_joins_paragraphs_with_no_pages():
    raw = _make_docx_bytes(["Vacation policy", "Employees get 25 days per year."])
    result = extract_text("policy.docx", raw)
    assert result.text == "Vacation policy\nEmployees get 25 days per year."
    assert result.pages is None


def test_extract_pdf_returns_per_page_text():
    # A real fixture with actually-extractable text. pypdf can write blank
    # pages but can't draw text onto them, so the fixture is generated once
    # with reportlab and committed — small, deterministic, and no extra
    # package or network access needed at test time.
    raw = (_FIXTURES / "two_page_sample.pdf").read_bytes()
    result = extract_text("sample.pdf", raw)
    assert result.pages is not None
    assert len(result.pages) == 2
    assert "first page" in result.pages[0].lower()
    assert "second page" in result.pages[1].lower()
    assert result.text == "\n".join(result.pages)


def test_extract_pdf_corrupt_file_raises_corrupt_document_error():
    raw = b"%PDF-1.4 not actually a valid pdf body"
    try:
        extract_text("broken.pdf", raw)
        assert False, "expected CorruptDocumentError"
    except CorruptDocumentError:
        pass


def test_extract_txt_empty_file_raises_empty_document_error():
    try:
        extract_text("empty.txt", b"   \n\n  ")
        assert False, "expected EmptyDocumentError"
    except EmptyDocumentError:
        pass
