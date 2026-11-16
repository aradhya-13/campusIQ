import re


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into chunks of roughly `size` characters.

    Paragraphs are kept together where possible. Paragraphs longer than `size`
    are cut at sentence boundaries. Each chunk after the first starts with the
    tail of the previous one (`overlap` characters) so context isn't lost at
    the boundaries.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        while len(para) > size:
            cut = para.rfind(". ", 0, size)
            cut = size if cut < size // 2 else cut + 1
            piece, para = para[:cut].strip(), para[cut:].strip()
            if current:
                chunks.append(current)
                current = ""
            if piece:
                chunks.append(piece)
        if not para:
            continue
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    if overlap > 0 and len(chunks) > 1:
        with_overlap = [chunks[0]]
        for prev, chunk in zip(chunks, chunks[1:]):
            tail = prev[-overlap:]
            space = tail.find(" ")
            if space != -1:
                tail = tail[space + 1 :]  # start on a word boundary
            with_overlap.append(f"{tail.strip()} {chunk}".strip())
        chunks = with_overlap

    return chunks
