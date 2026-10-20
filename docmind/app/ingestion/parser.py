"""Turn an uploaded file into a list of (page_number, text)."""
from pathlib import Path


def parse_file(path: Path) -> list[tuple[int, str]]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        import fitz  # PyMuPDF
        pages = []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append((i, text))
        return pages
    if ext == ".docx":
        import docx
        text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs if p.text.strip())
        return [(1, text)] if text else []
    if ext == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        lines, pages = text.splitlines(), []
        for i in range(0, len(lines), 400):  # virtual pages of 400 lines
            block = "\n".join(lines[i:i + 400]).strip()
            if block:
                pages.append((i // 400 + 1, block))
        return pages
    raise ValueError(f"Unsupported file type: {ext}")
