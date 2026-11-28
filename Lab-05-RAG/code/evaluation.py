"""
evaluation.py - Lab 5: measure retrieval quality (Recall@5 and MRR).
A chunk counts as relevant if it comes from the expected PDF and contains
the expected phrase.
"""
import re
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DB_DIR = "chroma_db"
EMBED_MODEL = "qwen3-embedding:0.6b"
K = 5

# (question, expected PDF, phrase that must appear in a relevant chunk)
EVAL_SET = [
    ("How long is the Lab sheet 1 session and which programming language is used?",
     "Lab sheet 1.pdf", "Duration: 1 Hour"),
    ("Which Python libraries are used in Lab sheet 1?",
     "Lab sheet 1.pdf", "Pandas, NumPy, NLTK, Scikit-learn, Matplotlib, Seaborn"),
    ("Which metrics should be used to evaluate the model in Lab sheet 1?",
     "Lab sheet 1.pdf", "Accuracy, Precision, Recall, F1"),
    ("Which group in Lab sheet 2 is about reducing food waste?",
     "Lab sheet 2.pdf", "GROUP 3 - The Food Waster"),
    ("How long is each group's final presentation in Lab sheet 2?",
     "Lab sheet 2.pdf", "minutes per group"),
    ("What is sentiment analysis also called?",
     "Week 1.pdf", "opinion mining"),
    ("How is lemmatization different from stemming?",
     "Week 1.pdf", "Reduces words to their dictionary form"),
    ("What are the limitations of Bag of Words?",
     "Week 1.pdf", "No Semantics"),
    ("What are the types of tokenization?",
     "Week 1.pdf", "Character-level"),
    ("Which model did Google introduce in 2013 to learn word embeddings from context?",
     "Week 1.pdf", "Word2Vec (Google, 2013)"),
    ("Who is the lecturer of the Week 1 lesson?",
     "Week 1.pdf", "Thilini Bakmeedeniya"),
    ("Which exercise asks you to build sentiment analysis with neural networks?",
     "Week 1.pdf", "Implement a sentiment analysis model using Neural Networks"),
]


def norm(text):
    return re.sub(r"\s+", " ", text).lower()


def is_relevant(doc, file_name, phrase):
    src = doc.metadata.get("source", "").replace("\\", "/").split("/")[-1]
    return src == file_name and norm(phrase) in norm(doc.page_content)


def evaluate(retrieve_fn, name):
    hits, rr_sum = 0, 0.0
    print(f"\n=== {name} ===")
    for q, file_name, phrase in EVAL_SET:
        docs = retrieve_fn(q, K)
        rank = next(
            (i + 1 for i, d in enumerate(docs) if is_relevant(d, file_name, phrase)),
            None,
        )
        if rank:
            hits += 1
            rr_sum += 1 / rank
        print(f"  rank={str(rank or '-').rjust(2)} | {q[:75]}")
    n = len(EVAL_SET)
    print(f"\n{name}: Recall@{K} = {hits}/{n} = {hits / n:.2f} | MRR = {rr_sum / n:.3f}")
    return hits / n, rr_sum / n


db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
)


def dense_retrieve(question, k):
    return db.similarity_search(question, k=k)


if __name__ == "__main__":
    evaluate(dense_retrieve, "Dense retrieval (baseline)")