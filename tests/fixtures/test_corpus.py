"""Guards the fixture corpus: every format is present, every file is
extractable, and each file actually contains the text the eval dataset expects
to find in it. A silently-empty or unparseable fixture would make the
end-to-end test and the eval both pass vacuously."""

import pytest

from tests.fixtures import CORPUS_DIR
from userdocs.extract import extract_text

_EXPECTED_CONTENT = {
    "vacation_policy.pdf": ["25", "vacation"],
    "employee_handbook.docx": ["notice", "resignation"],
    "benefits.md": ["wellness", "stipend"],
    "office_info.txt": ["office", "hours"],
}


def test_corpus_covers_all_four_supported_formats():
    suffixes = {path.suffix.lower() for path in CORPUS_DIR.iterdir() if path.is_file()}
    assert {".txt", ".md", ".docx", ".pdf"} <= suffixes


@pytest.mark.parametrize("filename", sorted(_EXPECTED_CONTENT))
def test_each_fixture_extracts_to_text_containing_its_expected_terms(filename):
    path = CORPUS_DIR / filename
    assert path.exists(), f"missing fixture: {filename}"

    extracted = extract_text(filename, path.read_bytes())
    lowered = extracted.text.lower()

    for term in _EXPECTED_CONTENT[filename]:
        assert term.lower() in lowered, f"{filename} does not mention {term!r}"


def test_pdf_fixture_has_more_than_one_page_so_page_attribution_is_testable():
    extracted = extract_text("vacation_policy.pdf", (CORPUS_DIR / "vacation_policy.pdf").read_bytes())
    assert extracted.pages is not None
    assert len(extracted.pages) >= 2
