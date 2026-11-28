"""
answer_eval.py - Lab 5 Group 1: end-to-end answer correctness and latency.
Compares the baseline (Dense) with the proposed system (Hybrid + Rerank).
"""
import time

from langchain_core.output_parsers import StrOutputParser

from compare_all import rerank_retrieve
from evaluation import EVAL_SET, dense_retrieve
from rag_pipeline import PROMPT, llm

TOP_K = 4

# Words that must appear in a correct answer (one list per question)
KEYWORDS = [
    ["1 hour", "python"],
    ["pandas", "nltk", "scikit-learn"],
    ["accuracy", "precision", "recall", "f1"],
    ["group 3"],
    ["3 minutes"],
    ["opinion mining"],
    ["dictionary"],
    ["semantic", "order"],
    ["sentence", "word", "character"],
    ["word2vec"],
    ["thilini"],
]
QUESTIONS = EVAL_SET[: len(KEYWORDS)]
chain = PROMPT | llm | StrOutputParser()

SYSTEMS = [
    ("Dense (baseline)", dense_retrieve),
    ("Hybrid + Rerank", rerank_retrieve),
]


def answer(question, retrieve_fn):
    t0 = time.perf_counter()
    docs = retrieve_fn(question, TOP_K)
    context = "\n\n".join(d.page_content for d in docs)
    text = chain.invoke({"context": context, "question": question}).strip()
    return text, time.perf_counter() - t0


if __name__ == "__main__":
    chain.invoke({"context": "Test.", "question": "Say OK."})  # warm-up (loads the LLM)
    log, summary = [], []
    for name, fn in SYSTEMS:
        correct = refused = 0
        total = 0.0
        for i, ((q, _, _), kws) in enumerate(zip(QUESTIONS, KEYWORDS), start=1):
            text, secs = answer(q, fn)
            total += secs
            low = text.lower()
            if all(k in low for k in kws):
                result = "CORRECT"
                correct += 1
            elif "could not find this in the documents" in low:
                result = "REFUSED"
                refused += 1
            else:
                result = "WRONG/INCOMPLETE"
            print(f"[{name}] Q{i}: {result} ({secs:.0f}s)")
            log.append(f"[{name}] Q{i}: {q}\n  ANSWER: {text}\n  RESULT: {result} | {secs:.1f}s\n")
        n = len(QUESTIONS)
        summary.append(
            f"  {name:<20} correct {correct}/{n} | refused {refused} | "
            f"avg end-to-end {total / n:.1f} s/query"
        )

    out = "ANSWER CORRECTNESS (11 questions, top-4 chunks)\n" + "\n".join(summary)
    print("\n" + out)
    with open("answer_results.txt", "w", encoding="utf-8") as f:
        f.write(out + "\n\n" + "\n".join(log))