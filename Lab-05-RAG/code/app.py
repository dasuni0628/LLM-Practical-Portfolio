"""
app.py - Lab 5: Streamlit chat UI for the RAG pipeline.
Features: multiple PDFs, user PDF upload, hybrid vs dense retrieval, sources.
"""
import re
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

DB_DIR = "chroma_db"
DOCS_DIR = Path("documents")
EMBED_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "llama3.2:3b"
DOCS_DIR.mkdir(exist_ok=True)

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I could not find this in the documents."

Context:
{context}

Question: {question}

Answer:"""
)

STOP = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "to", "and",
    "or", "for", "with", "which", "what", "who", "how", "does", "do", "did",
    "you", "your", "be", "by", "at", "from", "as", "it", "this", "that",
}


def tokenize(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOP]


@st.cache_resource
def load_models():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    return db, llm


db, llm = load_models()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)


def rebuild_index():
    """Reload all chunks from ChromaDB and rebuild the BM25 keyword index."""
    data = db.get()
    chunks = [
        Document(page_content=t, metadata=m)
        for t, m in zip(data["documents"], data["metadatas"])
    ]
    st.session_state.chunks = chunks
    st.session_state.bm25 = (
        BM25Okapi([tokenize(c.page_content) for c in chunks]) if chunks else None
    )
    st.session_state.files = sorted(
        {Path(c.metadata.get("source", "")).name for c in chunks}
    )


def add_pdf(uploaded):
    path = DOCS_DIR / uploaded.name
    path.write_bytes(uploaded.getvalue())
    pages = PyMuPDFLoader(str(path)).load()
    chunks = splitter.split_documents(pages)
    db.add_documents(chunks)
    return len(pages), len(chunks)


def dense_retrieve(question, k):
    return db.similarity_search(question, k=k)


def bm25_retrieve(question, k):
    chunks = st.session_state.chunks
    scores = st.session_state.bm25.get_scores(tokenize(question))
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in order[:k]]


def hybrid_retrieve(question, k, pool=20, rrf_c=60):
    fused, docs = {}, {}
    for results in (dense_retrieve(question, pool), bm25_retrieve(question, pool)):
        for rank, d in enumerate(results, start=1):
            key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content)
            docs[key] = d
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_c + rank)
    best = sorted(fused, key=fused.get, reverse=True)[:k]
    return [docs[key] for key in best]


def show_sources(sources):
    with st.expander("Sources used"):
        for s in sources:
            st.markdown(f"**{s['file']} - page {s['page']}**")
            st.caption(s["text"])


if "chunks" not in st.session_state:
    rebuild_index()
    st.session_state.messages = []

# ---------------- Sidebar ----------------
st.sidebar.header("Knowledge base")
uploads = st.sidebar.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True
)
if uploads:
    new_files = [u for u in uploads if u.name not in st.session_state.files]
    if new_files:
        with st.spinner("Indexing uploaded PDFs (embedding on CPU)..."):
            for u in new_files:
                n_pages, n_chunks = add_pdf(u)
                st.sidebar.success(f"{u.name}: {n_pages} pages, {n_chunks} chunks added")
            rebuild_index()

st.sidebar.write(f"**Indexed files ({len(st.session_state.files)}):**")
for name in st.session_state.files:
    st.sidebar.write(f"- {name}")
st.sidebar.write(f"Total chunks: {len(st.session_state.chunks)}")

mode = st.sidebar.radio(
    "Retrieval mode", ["Hybrid (BM25 + Dense)", "Dense only (baseline)"]
)
k = st.sidebar.slider("Chunks to retrieve (top-k)", 2, 8, 4)

# ---------------- Main chat ----------------
st.title("Lab 5 - RAG Chat with your PDFs")
st.caption(f"Embeddings: {EMBED_MODEL} | LLM: {LLM_MODEL} (both local via Ollama)")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not st.session_state.chunks:
            st.info("The knowledge base is empty. Upload a PDF in the sidebar.")
        else:
            if mode.startswith("Hybrid"):
                docs = hybrid_retrieve(question, k)
            else:
                docs = dense_retrieve(question, k)
            context = "\n\n".join(d.page_content for d in docs)
            chain = PROMPT | llm | StrOutputParser()
            answer = st.write_stream(
                chain.stream({"context": context, "question": question})
            )
            sources = [
                {
                    "file": Path(d.metadata.get("source", "")).name,
                    "page": d.metadata.get("page", 0) + 1,
                    "text": d.page_content[:300],
                }
                for d in docs
            ]
            show_sources(sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )