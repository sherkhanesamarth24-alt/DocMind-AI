"""Hybrid retrieval: dense + BM25 merged with Reciprocal Rank Fusion,
then optional cross-encoder re-ranking."""
import re
from functools import lru_cache
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from app.config import RERANK_MODEL
from app.retrieval import store

_tok = lambda s: re.findall(r"\w+", s.lower())
_bm25_cache = {"version": -1, "index": None, "chunks": []}


def _bm25_index():
    if _bm25_cache["version"] != store.version:
        chunks = store.get_all()
        _bm25_cache.update(
            version=store.version,
            chunks=chunks,
            index=BM25Okapi([_tok(c["text"]) for c in chunks]) if chunks else None,
        )
    return _bm25_cache["index"], _bm25_cache["chunks"]


def bm25_search(query: str, k: int, doc_ids: list[str] | None = None) -> list[dict]:
    index, chunks = _bm25_index()
    if index is None:
        return []
    scores = index.get_scores(_tok(query))
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    out = [c for s, c in ranked if s > 0 and (not doc_ids or c["doc_id"] in doc_ids)]
    return out[:k]


def rrf(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores, by_id = {}, {}
    for results in result_lists:
        for rank, c in enumerate(results):
            scores[c["id"]] = scores.get(c["id"], 0) + 1 / (k + rank + 1)
            by_id[c["id"]] = c
    return [by_id[i] for i in sorted(scores, key=scores.get, reverse=True)]


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


def rerank(query: str, chunks: list[dict]) -> list[dict]:
    scores = _reranker().predict([(query, c["text"]) for c in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [{**c, "score": float(s)} for s, c in ranked]


def retrieve(query: str, k: int = 5, hybrid: bool = True, use_rerank: bool = True,
             doc_ids: list[str] | None = None, pool: int = 20) -> list[dict]:
    dense = store.dense_search(query, pool, doc_ids)
    candidates = rrf([dense, bm25_search(query, pool, doc_ids)])[:pool] if hybrid else dense
    if use_rerank and candidates:
        candidates = rerank(query, candidates)
    return candidates[:k]
