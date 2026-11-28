"""
rag_pipeline.py - Lab 5 RAG: answer questions using the PDF chunks.
Pipeline: question -> retriever (ChromaDB) -> prompt template -> LLM (Ollama)
"""
from pathlib import Path

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

DB_DIR = "chroma_db"
EMBED_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "llama3.2:3b"
TOP_K = 4  # number of chunks retrieved per question

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I could not find this in the documents."

Context:
{context}

Question: {question}

Answer:"""
)

embeddings = OllamaEmbeddings(model=EMBED_MODEL)
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": TOP_K})
llm = ChatOllama(model=LLM_MODEL, temperature=0)


def ask(question):
    """Return (answer, sources). sources = list of (file name, page number)."""
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    answer = (PROMPT | llm).invoke({"context": context, "question": question}).content
    sources = [
        (Path(d.metadata.get("source", "")).name, d.metadata.get("page", 0) + 1)
        for d in docs
    ]
    return answer, sources

if __name__ == "__main__":
    while True:
        q = input("\nQuestion (type exit to quit): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        answer, sources = ask(q)
        print("\nAnswer:", answer)
        print("Sources:", sources)