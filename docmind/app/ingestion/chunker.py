"""Sentence-aware chunking with overlap. Chunks never cross page boundaries,
so every chunk maps to exactly one page for citations."""
import re
from app.config import CHUNK_WORDS, CHUNK_OVERLAP_WORDS

_SENT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def chunk_pages(pages: list[tuple[int, str]]) -> list[dict]:
    chunks = []
    for page_no, text in pages:
        text = re.sub(r"[ \t]+", " ", text)
        sentences = [s.strip() for s in _SENT.split(text) if s.strip()]
        current: list[str] = []
        count = 0
        for sent in sentences:
            words = len(sent.split())
            if count + words > CHUNK_WORDS and current:
                chunks.append({"page": page_no, "text": " ".join(current)})
                overlap, n = [], 0
                for s in reversed(current):
                    n += len(s.split())
                    if n > CHUNK_OVERLAP_WORDS:
                        break
                    overlap.insert(0, s)
                current, count = overlap, sum(len(s.split()) for s in overlap)
            current.append(sent)
            count += words
        if current:
            chunks.append({"page": page_no, "text": " ".join(current)})
    return [c for c in chunks if len(c["text"].split()) >= 5]
