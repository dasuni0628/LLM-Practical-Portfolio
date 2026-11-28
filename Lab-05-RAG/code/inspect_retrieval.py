"""inspect_retrieval.py - show which chunks the retriever returns for a query."""
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=OllamaEmbeddings(model="qwen3-embedding:0.6b"),
)

QUERIES = [
    "How long is the Lab sheet 1 session and which programming language is used?",
    "Sentiment analysis duration 1 hour programming language Python",
]

for q in QUERIES:
    print("=" * 70)
    print("QUERY:", q)
    for doc, score in db.similarity_search_with_score(q, k=5):
        name = Path(doc.metadata.get("source", "")).name
        page = doc.metadata.get("page", 0) + 1
        text = doc.page_content[:110].replace("\n", " ")
        print(f"  distance={score:.3f} | {name} p{page} | {text}")