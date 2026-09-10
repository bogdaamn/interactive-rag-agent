import io

from docx import Document as DocxDocument

from userdocs.extract import extract_text


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
