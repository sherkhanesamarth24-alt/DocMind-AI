"""Ablation study: vector only vs hybrid vs hybrid+reranker.
Upload your documents in the app first, then run:  python -m evaluation.eval_retrieval"""
import json
from pathlib import Path
from app.retrieval.pipeline import retrieve

HERE = Path(__file__).parent
SETUPS = {
    "Vector only": dict(hybrid=False, use_rerank=False),
    "Hybrid (BM25 + vector)": dict(hybrid=True, use_rerank=False),
    "Hybrid + re-ranker": dict(hybrid=True, use_rerank=True),
}


def run():
    tests = json.loads((HERE / "testset.json").read_text())
    rows = []
    for name, cfg in SETUPS.items():
        hits, rr = 0, 0.0
        for t in tests:
            results = retrieve(t["question"], k=5, **cfg)
            for rank, c in enumerate(results, 1):
                if c["filename"] == t["filename"] and c["page"] == t["page"]:
                    hits += 1
                    rr += 1 / rank
                    break
        rows.append({"setup": name, "hit@5": round(hits / len(tests), 3), "mrr": round(rr / len(tests), 3)})
        print(rows[-1])
    (HERE / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    run()
