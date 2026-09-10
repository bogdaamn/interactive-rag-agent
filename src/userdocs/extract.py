"""Text extraction for uploaded documents. See spec/v3/SPEC.md §5.1."""

from dataclasses import dataclass

from userdocs.errors import UnsupportedFormatError

SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}


@dataclass
class ExtractedDocument:
    text: str
    pages: list[str] | None


def _extension_of(filename: str) -> str:
    idx = filename.rfind(".")
    return filename[idx:].lower() if idx != -1 else ""


def extract_text(filename: str, raw_bytes: bytes) -> ExtractedDocument:
    ext = _extension_of(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(f"Unsupported file extension: {ext!r}")

    if ext in (".txt", ".md"):
        text = raw_bytes.decode("utf-8", errors="replace")
        return ExtractedDocument(text=text, pages=None)

    raise UnsupportedFormatError(f"Extension {ext!r} not yet implemented")
