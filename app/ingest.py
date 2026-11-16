from pathlib import Path

from . import store
from .chunking import split_text
from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .loaders import load_pages


def ingest_file(filename: str, data: bytes) -> int:
    """Parse, chunk, embed and store a file. Returns the number of chunks."""
    name = Path(filename).name
    pages = load_pages(name, data)

    records: list[tuple[int, str]] = []
    for page_number, text in pages:
        for chunk in split_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            records.append((page_number, chunk))

    if not records:
        raise ValueError(
            "No extractable text found (scanned PDFs need OCR first)."
        )

    store.replace_document(name, records)
    return len(records)
