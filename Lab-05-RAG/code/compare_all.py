"""
compare_all.py - Lab 5 Group 1: Dense vs BM25 vs Hybrid vs Hybrid + Rerank.
Metrics: Recall@5, MRR and average retrieval latency per query.
"""
import time

from sentence_transformers import CrossEncoder

from evaluation import EVAL_SET, K, is_relevant, dense_retrieve
from hybrid import bm25_retrieve, hybrid_retrieve

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_retrieve(question, k, candidates=10):
    """Hybrid top-10, then re-score each (question, chunk) pair with a cross-encoder."""
    cands = hybrid_retrieve(question, candidates)
    scores = reranker.predict([(question, d.page_content) for d in cands])
    ranked = sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:k]]


SYSTEMS = [
    ("Dense only (baseline)", dense_retrieve),
    ("BM25 only", bm25_retrieve),
    ("Hybrid (BM25 + Dense, RRF)", hybrid_retrieve),
    ("Hybrid + Rerank", rerank_retrieve),
]


def run(fn):
    fn(EVAL_SET[0][0], K)  # warm-up call, not timed (loads models)
    hits, rr_sum, total = 0, 0.0, 0.0
    for q, file_name, phrase in EVAL_SET:
        t0 = time.perf_counter()
        docs = fn(q, K)
        total += time.perf_counter() - t0
        rank = next(
            (i + 1 for i, d in enumerate(docs) if is_relevant(d, file_name, phrase)),
            None,
        )
        if rank:
            hits += 1
            rr_sum += 1 / rank
    n = len(EVAL_SET)
    return hits / n, rr_sum / n, total / n * 1000


if __name__ == "__main__":
    lines = ["COMPARISON (12 questions, top-5)",
             f"  {'System':<28} {'Recall@5':>9} {'MRR':>7} {'Latency (ms/query)':>20}"]
    for name, fn in SYSTEMS:
        print(f"Running {name} ...")
        recall, mrr, ms = run(fn)
        lines.append(f"  {name:<28} {recall:>9.2f} {mrr:>7.3f} {ms:>20.0f}")
    print("\n" + "\n".join(lines))
    with open("retrieval_results_full.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")