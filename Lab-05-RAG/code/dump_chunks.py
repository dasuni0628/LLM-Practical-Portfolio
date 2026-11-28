from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=OllamaEmbeddings(model="qwen3-embedding:0.6b"),
)
data = db.get()
rows = []
for text, meta in zip(data["documents"], data["metadatas"]):
    rows.append((Path(meta.get("source", "")).name, meta.get("page", 0) + 1, text))
rows.sort(key=lambda r: (r[0], r[1]))

with open("chunks_dump.txt", "w", encoding="utf-8") as f:
    for i, (name, page, text) in enumerate(rows):
        f.write(f"--- CHUNK {i} | {name} | page {page} ---\n{text}\n\n")

print(f"Wrote {len(rows)} chunks to chunks_dump.txt")