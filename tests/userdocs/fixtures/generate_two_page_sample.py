# tests/userdocs/fixtures/generate_two_page_sample.py — one-off generator,
# committed as documentation of how the fixture was made. Not collected by
# pytest (no test_ prefix) and not imported by any test.
"""Regenerate tests/userdocs/fixtures/two_page_sample.pdf.

    pip install reportlab
    python tests/userdocs/fixtures/generate_two_page_sample.py

reportlab is NOT a runtime or test dependency — do not add it to
requirements.txt. The generated PDF is committed so the suite needs neither the
package nor a network connection.
"""

from pathlib import Path

from reportlab.pdfgen import canvas

OUTPUT = Path(__file__).parent / "two_page_sample.pdf"

pdf = canvas.Canvas(str(OUTPUT))
pdf.drawString(100, 700, "This is the first page of the sample document.")
pdf.showPage()
pdf.drawString(100, 700, "This is the second page of the sample document.")
pdf.showPage()
pdf.save()
print(f"Wrote {OUTPUT}")
