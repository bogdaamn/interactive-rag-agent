"""One-off generator for the fixture corpus. Run once, commit the output.

Requires two packages that are NOT runtime dependencies and must not be added
to requirements.txt:
    pip install reportlab python-docx

Usage:
    python tests/fixtures/generate_corpus.py
"""

from pathlib import Path

from docx import Document as DocxDocument
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

CORPUS_DIR = Path(__file__).parent / "corpus"

VACATION_PAGE_1 = [
    "Vacation Policy",
    "",
    "All full-time employees receive 25 days of paid vacation per year.",
    "Vacation accrues monthly and becomes available after the first 90 days",
    "of employment.",
]

VACATION_PAGE_2 = [
    "Carrying Over Unused Vacation",
    "",
    "Up to 5 unused vacation days may be carried over into the following",
    "calendar year. Any balance above 5 days is forfeited on December 31.",
    "Carried-over days must be used by March 31.",
]

HANDBOOK_PARAGRAPHS = [
    "Employee Handbook",
    "",
    "Resignation and notice period.",
    "Employees resigning from the company must give a notice period of "
    "four weeks in writing to their direct manager.",
    "",
    "Sick leave.",
    "The company covers 10 paid sick days per calendar year. A doctor's note "
    "is required for any absence longer than three consecutive days.",
]

BENEFITS_MARKDOWN = """# Benefits

## Wellness

Every employee receives a wellness stipend of $50 per month, which can be
spent on gym memberships, fitness classes, or mental-health services.

Remote employees are fully eligible for the wellness stipend on the same terms
as office-based employees.

## Equipment

New hires receive a laptop and a one-time $300 home-office budget.
"""

OFFICE_INFO_TEXT = """Office Information

The office is open from 9:00 to 18:00, Monday through Friday.
Badge access is required outside those office hours.
The nearest parking garage is on Second Street.
"""


def _write_pdf() -> None:
    pdf_canvas = canvas.Canvas(str(CORPUS_DIR / "vacation_policy.pdf"), pagesize=letter)
    for page_lines in (VACATION_PAGE_1, VACATION_PAGE_2):
        y = 720
        for line in page_lines:
            pdf_canvas.drawString(72, y, line)
            y -= 18
        pdf_canvas.showPage()
    pdf_canvas.save()


def _write_docx() -> None:
    document = DocxDocument()
    for paragraph in HANDBOOK_PARAGRAPHS:
        document.add_paragraph(paragraph)
    document.save(str(CORPUS_DIR / "employee_handbook.docx"))


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    _write_pdf()
    _write_docx()
    (CORPUS_DIR / "benefits.md").write_text(BENEFITS_MARKDOWN, encoding="utf-8")
    (CORPUS_DIR / "office_info.txt").write_text(OFFICE_INFO_TEXT, encoding="utf-8")
    print(f"Wrote fixture corpus to {CORPUS_DIR}")


if __name__ == "__main__":
    main()
