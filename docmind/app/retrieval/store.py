"""Chroma vector store + embedding model (lazy-loaded singletons)."""
from functools import lru_cache
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import CHROMA_DIR, EMBED_MODEL

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
version = 0  # bumped on every write so the BM25 cache knows when to rebuild


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection("docmind", metadata={"hnsw:space": "cosine"})


def embed(texts: list[str]) -> list[list[float]]:
    return _embedder().encode(texts, normalize_embeddings=True, batch_size=32).tolist()


def add_chunks(doc_id: str, filename: str, chunks: list[dict]) -> int:
    global version
    col = _collection()
    for i in range(0, len(chunks), 256):
        batch = chunks[i:i + 256]
        col.add(
            ids=[f"{doc_id}-{i + j}" for j in range(len(batch))],
            documents=[c["text"] for c in batch],
            embeddings=embed([c["text"] for c in batch]),
            metadatas=[{"doc_id": doc_id, "filename": filename, "page": c["page"]} for c in batch],
        )
    version += 1
    return len(chunks)


def delete_document(doc_id: str) -> None:
    global version
    _collection().delete(where={"doc_id": doc_id})
    version += 1


def get_all() -> list[dict]:
    res = _collection().get(include=["documents", "metadatas"])
    return [{"id": i, "text": d, **m} for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]


def list_documents() -> list[dict]:
    docs: dict[str, dict] = {}
    for c in get_all():
        d = docs.setdefault(c["doc_id"], {"doc_id": c["doc_id"], "filename": c["filename"], "chunks": 0})
        d["chunks"] += 1
    return sorted(docs.values(), key=lambda d: d["filename"])


def dense_search(query: str, k: int, doc_ids: list[str] | None = None) -> list[dict]:
    col = _collection()
    total = col.count()
    if total == 0:
        return []
    where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
    res = col.query(
        query_embeddings=embed([_QUERY_PREFIX + query]),
        n_results=min(k, total),
        where=where,
        include=["documents", "metadatas"],
    )
    return [{"id": i, "text": d, **m} for i, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])]
