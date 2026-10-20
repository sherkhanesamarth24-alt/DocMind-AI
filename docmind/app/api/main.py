import json
import uuid
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.config import ALLOWED_EXT, BASE_DIR, MAX_UPLOAD_MB, UPLOAD_DIR
from app.generation.llm import generate_answer, rewrite_query
from app.ingestion.chunker import chunk_pages
from app.ingestion.parser import parse_file
from app.retrieval import store
from app.retrieval.pipeline import retrieve

app = FastAPI(title="DocMind API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []
    doc_ids: list[str] | None = None
    hybrid: bool = True
    rerank: bool = True
    rewrite: bool = True
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def documents():
    return store.list_documents()


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            results.append({"filename": f.filename, "error": f"Unsupported type {ext}. Use PDF, DOCX or TXT."})
            continue
        data = await f.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            results.append({"filename": f.filename, "error": f"File exceeds {MAX_UPLOAD_MB} MB."})
            continue
        doc_id = uuid.uuid4().hex[:10]
        path = UPLOAD_DIR / f"{doc_id}{ext}"
        path.write_bytes(data)
        try:
            chunks = chunk_pages(parse_file(path))
            if not chunks:
                raise ValueError("No readable text found (scanned PDFs need OCR).")
            n = store.add_chunks(doc_id, f.filename, chunks)
            results.append({"filename": f.filename, "doc_id": doc_id, "chunks": n})
        except Exception as e:
            path.unlink(missing_ok=True)
            results.append({"filename": f.filename, "error": str(e)})
    return results


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    store.delete_document(doc_id)
    for p in UPLOAD_DIR.glob(f"{doc_id}.*"):
        p.unlink(missing_ok=True)
    return {"deleted": doc_id}


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question is empty.")
    try:
        query = rewrite_query(req.question, req.history) if req.rewrite else req.question
        chunks = retrieve(query, k=req.top_k, hybrid=req.hybrid, use_rerank=req.rerank, doc_ids=req.doc_ids)
        out = generate_answer(req.question, chunks, req.history)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    return {**out, "rewritten_query": query}


@app.get("/eval/results")
def eval_results():
    p = BASE_DIR / "evaluation" / "results.json"
    return json.loads(p.read_text()) if p.exists() else []
