"""
ingest.py - Lab 5 RAG: build the vector database from PDF files.
Pipeline: load PDFs -> split into chunks -> embed (Ollama) -> store (ChromaDB)
"""
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = Path("documents")
DB_DIR = "chroma_db"
EMBED_MODEL = "qwen3-embedding:0.6b"
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks


def load_documents():
    docs = []
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):
        pages = PyMuPDFLoader(str(pdf)).load()
        print(f"Loaded {pdf.name}: {len(pages)} pages")
        docs.extend(pages)
    return docs


def main():
    docs = load_documents()
    if not docs:
        print("No PDFs found in the documents folder.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Total pages: {len(docs)}  ->  Total chunks: {len(chunks)}")

    # start from a clean database each time (avoids duplicate chunks)
    if Path(DB_DIR).exists():
        shutil.rmtree(DB_DIR)

    print(f"Embedding chunks with {EMBED_MODEL} (this can take a few minutes on CPU)...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
    print(f"Done. {db._collection.count()} chunks stored in '{DB_DIR}'.")

    print("\nExample chunk:")
    print(chunks[0].page_content[:300])
    print("Metadata:", chunks[0].metadata)


if __name__ == "__main__":
    main()