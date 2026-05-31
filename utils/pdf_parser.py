from pypdf import PdfReader
from typing import List


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file page by page."""
    reader = PdfReader(file_path)
    full_text = ""
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n[Page {page_num + 1}]\n{text}"
    return full_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping word-based chunks."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1

        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            overlap_words = current_chunk[-10:]
            current_chunk = overlap_words
            current_length = sum(len(w) + 1 for w in overlap_words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
