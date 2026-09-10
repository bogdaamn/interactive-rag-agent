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
