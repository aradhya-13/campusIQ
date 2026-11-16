import io
from pathlib import Path

from pypdf import PdfReader


def load_pages(filename: str, data: bytes) -> list[tuple[int, str]]:
    """Return a list of (page_number, text). Non-PDF files are a single page."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((i, text))
        return pages

    if ext in {".txt", ".md"}:
        text = data.decode("utf-8", errors="replace").strip()
        return [(1, text)] if text else []

    raise ValueError(f"Unsupported file type: {ext}")
