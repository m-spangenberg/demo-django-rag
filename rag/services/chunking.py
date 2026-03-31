from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PageText:
    page_number: int
    text: str


@dataclass(slots=True)
class Chunk:
    page_number: int
    text: str


def build_chunks(
    pages: list[PageText],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        text = " ".join(page.text.split())
        if not text:
            continue
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(page_number=page.page_number, text=chunk_text))
            if end >= text_length:
                break
            start = max(end - chunk_overlap, start + 1)
    return chunks
