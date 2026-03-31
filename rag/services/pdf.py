from __future__ import annotations

from pypdf import PdfReader

from .chunking import PageText


def extract_pdf_pages(file_path: str) -> list[PageText]:
    reader = PdfReader(file_path)
    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(PageText(page_number=index, text=text))
    return pages
