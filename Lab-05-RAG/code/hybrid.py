"""
hybrid.py - Lab 5 advanced RAG: hybrid retrieval (BM25 + dense) with
Reciprocal Rank Fusion (RRF), compared against the dense-only baseline.
"""
import re
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from evaluation import evaluate, dense_retrieve, db

STOP = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "to", "and",
    "or", "for", "with", "which", "what", "who", "how", "does", "do", "did",
    "you", "your", "be", "by", "at", "from", "as", "it", "this", "that",
}


def tokenize(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOP]


# Build the BM25 keyword index from the chunks already stored in ChromaDB
_data = db.get()
CHUNKS = [
    Document(page_content=t, metadata=m)
    for t, m in zip(_data["documents"], _data["metadatas"])
]
_bm25 = BM25Okapi([tokenize(c.page_content) for c in CHUNKS])


def bm25_retrieve(question, k):
    scores = _bm25.get_scores(tokenize(question))
    order = sorted(range(len(CHUNKS)), key=lambda i: scores[i], reverse=True)
    return [CHUNKS[i] for i in order[:k]]


def _key(doc):
    return (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)


def hybrid_retrieve(question, k, pool=20, rrf_c=60):
    """Fuse dense and BM25 rankings with Reciprocal Rank Fusion."""
    fused, docs = {}, {}
    for results in (dense_retrieve(question, pool), bm25_retrieve(question, pool)):
        for rank, d in enumerate(results, start=1):
            key = _key(d)
            docs[key] = d
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_c + rank)
    best = sorted(fused, key=fused.get, reverse=True)[:k]
    return [docs[key] for key in best]


if __name__ == "__main__":
    systems = [
        ("Dense only (baseline)", dense_retrieve),
        ("BM25 only", bm25_retrieve),
        ("Hybrid BM25 + Dense (RRF)", hybrid_retrieve),
    ]
    results = {name: evaluate(fn, name) for name, fn in systems}

    lines = ["SUMMARY (12 questions, top-5)"]
    for name, (recall, mrr) in results.items():
        lines.append(f"  {name:<28} Recall@5 = {recall:.2f}   MRR = {mrr:.3f}")
    print("\n" + "\n".join(lines))
    with open("retrieval_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")