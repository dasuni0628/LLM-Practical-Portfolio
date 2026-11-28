# Lab 05 – Retrieval-Augmented Generation (RAG) Architecture

## 1. Laboratory Session

| | |
|---|---|
| **Module** | PUSL 3205 – Large Language Models and Generative AI |
| **Lab sheet** | Lab Sheet 5 – RAG Architecture |
| **Student** | Rubasinghe D D (Student ID 33680) |
| **Completed** | 28 November 2026 |
| **Group activity** | Group 1 – Hybrid Retrieval and Reranking (completed individually) |
| **Environment** | Windows 11 laptop, Intel Iris Xe graphics (no NVIDIA GPU), 16 GB RAM. All models run locally on the CPU. |

---

## 2. Objective

The objective of this lab was to build a complete **Retrieval-Augmented Generation (RAG)** system that runs entirely on a local laptop and answers questions from my own PDF documents:

documents → text splitter → embeddings → vector database → retriever → prompt template → LLM → chat interface

The second objective (group activity) was to **improve the retrieval stage** with hybrid search (BM25 keyword search + vector search, rank fusion and reranking) and to **measure honestly** whether the improvement is real, using Recall@5, MRR, latency and answer correctness.

---

## 3. Tasks and Implementation

### 3.1 Environment setup
- Python 3.12.10 virtual environment (`rag_env`) on Windows 11.
- Libraries: LangChain 1.x, ChromaDB 1.5.9, Streamlit 1.67.0, sentence-transformers 6.3.0, PyMuPDF, rank-bm25 (full list in `code/requirements.txt`).
- Ollama 0.40.18 serving two local models:
  - LLM: `llama3.2:3b`
  - Embedding model: `qwen3-embedding:0.6b`
- Evidence: screenshots 01 to 08.

### 3.2 Document ingestion – `code/ingest.py`
1. Load every PDF in the `documents/` folder with `PyMuPDFLoader` (one document per page).
2. Split pages into chunks with `RecursiveCharacterTextSplitter` (**800 characters, 150 overlap**).
3. Embed each chunk with `qwen3-embedding:0.6b` through Ollama.
4. Store chunks and vectors in a persistent **ChromaDB** collection.

Documents used: Lab sheet 1, Lab sheet 2 and the Week 1 lecture notes → **42 pages → 48 chunks**. Later, Lab sheet 5 was uploaded through the app (6 pages, 10 chunks), giving **58 chunks**.

### 3.3 Baseline RAG pipeline – `code/rag_pipeline.py`
- Retriever: dense (vector) search, top-4 chunks.
- Prompt template: *"Answer the question using ONLY the context below. If the answer is not in the context, reply exactly: I could not find this in the documents."*
- LLM: `llama3.2:3b` with temperature 0.
- Returns the answer and the source file and page of every retrieved chunk.

### 3.4 Streamlit chat application – `code/app.py`
- Chat interface with streaming answers.
- Works with **multiple PDFs** and lets the user **upload new PDFs** while the app is running (they are chunked, embedded and added to ChromaDB immediately).
- Sidebar switch between **Hybrid (BM25 + Dense)** and **Dense only (baseline)**, and a top-k slider.
- Every answer has a "Sources used" panel showing file name, page and chunk text.
- **Note:** the app's *Hybrid* mode is BM25 + dense search + Reciprocal Rank Fusion. The cross-encoder **reranker** is used only in the evaluation scripts (`compare_all.py`, `answer_eval.py`), not inside the app.

### 3.5 Group activity (Group 1) – hybrid retrieval and reranking
| Script | Purpose |
|---|---|
| `code/evaluation.py` | 12-question evaluation set; Recall@5 and MRR for the dense baseline |
| `code/hybrid.py` | BM25 index + **Reciprocal Rank Fusion** (dense top-20 + BM25 top-20) |
| `code/compare_all.py` | Adds a **cross-encoder reranker** (`ms-marco-MiniLM-L-6-v2`) that re-scores the hybrid top-10 chunks, and measures Recall@5, MRR and latency for 4 systems |
| `code/answer_eval.py` | Asks the real LLM 11 questions with the baseline and the proposed system (top-4 chunks go to the LLM); counts correct answers and refusals; measures end-to-end time |
| `code/inspect_retrieval.py`, `code/dump_chunks.py` | Diagnostic helpers (see which chunks are retrieved, list all stored chunks) |

The architecture diagram and a short technical report are in [`group-activity/`](group-activity/).

### 3.6 Repository contents
```
Lab-05-RAG/
├── README.md
├── code/              Python scripts + requirements.txt
├── results/           retrieval and answer evaluation results (text files)
├── screenshots/       evidence screenshots 01 - 21
└── group-activity/    architecture diagram + technical report
```
The PDF documents, the ChromaDB folder and the virtual environment are **not** uploaded (the PDFs are course material and the other two can be regenerated).

### 3.7 How to reproduce
```powershell
py -3.12 -m venv rag_env
.\rag_env\Scripts\Activate.ps1
pip install -r code\requirements.txt
ollama pull llama3.2:3b
ollama pull qwen3-embedding:0.6b
# put your own PDFs in a folder called documents\
python code\ingest.py
streamlit run code\app.py
```
Run the scripts from the folder that contains `documents/` and `chroma_db/`.

### 3.8 Note on AI assistance
I used Claude as a guide while building and debugging the scripts. I ran every command and script myself on my laptop, took all screenshots from my own runs, and wrote the observations from my own results.

---

## 4. Results and Evidence

### 4.1 Environment setup (screenshots 01 to 08)
<details>
<summary>Show setup screenshots</summary>

**01 – Python versions installed (3.12 selected for the lab)**

<img src="screenshots/01_python_versions.png" width="700">

**02 – Virtual environment created and activated**

<img src="screenshots/02_venv_activated.png" width="700">

**03 – Libraries installed**

<img src="screenshots/03_libraries_installed.png" width="700">

**04 – ChromaDB check**

<img src="screenshots/04_chromadb_check.png" width="700">

**05 – Ollama installed**

<img src="screenshots/05_ollama_installed.png" width="700">

**06 – Models downloaded (`llama3.2:3b` and `qwen3-embedding:0.6b`)**

<img src="screenshots/06_models_downloaded.png" width="700">

**07 – Ollama service running**

<img src="screenshots/07_ollama_running.png" width="700">

</details>

### 4.2 The LLM on its own (no retrieval)
Asked directly *"Explain in one sentence what RAG means in AI."*, `llama3.2:3b` answered that RAG is "Randomized Augmented Gradient", which is **wrong** (RAG means Retrieval-Augmented Generation).

<img src="screenshots/08_llm_test.png" width="700">

### 4.3 Ingestion
Three PDFs were loaded (42 pages), split into 48 chunks, embedded and stored in ChromaDB.

<img src="screenshots/09_documents_folder.png" width="600">

<img src="screenshots/10_ingest_run.png" width="700">

### 4.4 First RAG test and retrieval inspection
The first baseline test answered *"I could not find this in the documents."* for two questions about Lab sheet 1. The inspection script shows why: for the first question all five retrieved chunks had distance above 1.0, and Lab sheet 1 was not among them.

<img src="screenshots/11_rag_test.png" width="700">

<img src="screenshots/12_retrieval_inspection.png" width="700">

### 4.5 Streamlit application
<img src="screenshots/15_streamlit_home.png" width="700">

**Hybrid mode answers correctly** ("1 hour, Python") with `Lab sheet 1.pdf – page 1` as the first source:

<img src="screenshots/16a_chat_answer.png" width="700">

<details>
<summary>Sources panel (hybrid mode)</summary>

<img src="screenshots/16b_chat_sources.png" width="700">

</details>

**Dense-only (baseline) mode on the same question refuses to answer:**

<img src="screenshots/17a_dense_answer.png" width="700">

**Uploading a new PDF while the app is running** (Lab sheet 5: 6 pages, 10 chunks added; 4 files, 58 chunks in total):

<img src="screenshots/18a_upload_indexed.png" width="700">

**A question about the uploaded PDF is answered correctly** (*"What does Group 5 – Multilingual University RAG evaluate?"* → English, Sinhala and Tamil questions against university documents):

<img src="screenshots/18d_upload_answer_ok.png" width="700">

<details>
<summary>First question about the uploaded PDF (false refusal) and extraction check</summary>

*"Which embedding model and which LLM are used in Lab sheet 5?"* was answered with "I could not find this in the documents", although the right chunk was retrieved (rank 1). Screenshot 19 shows the extracted page text does contain "Llama 3.2 3B" and "Qwen3 Embedding 0.6B", so this was a **generator-side** failure, not a retrieval failure.

<img src="screenshots/18b_upload_answer.png" width="700">

<img src="screenshots/18c_upload_answer.png" width="700">

<img src="screenshots/19_page2_extraction.png" width="700">

</details>

### 4.6 Retrieval evaluation (12 questions, top-5)
A chunk counts as relevant if it comes from the expected PDF and contains the expected phrase.

**Run 1 – 3 documents, 48 chunks** (`results/retrieval_results.txt`)

| System | Recall@5 | MRR |
|---|---|---|
| Dense only (baseline) | 0.92 | 0.618 |
| BM25 only | 1.00 | 0.847 |
| Hybrid BM25 + Dense (RRF) | 1.00 | 0.861 |

**Run 2 – 4 documents, 58 chunks, all four systems** (`results/retrieval_results_full.txt`)

| System | Recall@5 | MRR | Retrieval latency (ms/query) |
|---|---|---|---|
| Dense only (baseline) | 0.83 | 0.507 | 175 |
| BM25 only | 1.00 | 0.792 | ~0 |
| Hybrid BM25 + Dense (RRF) | 1.00 | 0.819 | 176 |
| **Hybrid + Rerank** | **1.00** | **0.958** | 589 |

<img src="screenshots/13_baseline_evaluation.png" width="700">

<img src="screenshots/20_full_comparison.png" width="700">

<details>
<summary>Run 1 detailed output</summary>

<img src="screenshots/14a_hybrid_evaluation.png" width="700">

<img src="screenshots/14b_hybrid_summary.png" width="700">

</details>

### 4.7 End-to-end answer evaluation (11 questions, top-4 chunks)
Full answers are in `results/answer_results.txt`. I graded the answers twice: an automatic keyword check, and my own reading of every answer.

| System | Keyword check | My manual reading | Refusals | Avg end-to-end time |
|---|---|---|---|---|
| Dense (baseline) | 8 / 11 | 6 correct, 2 partial, 3 failed (2 refusals + 1 answer from the wrong document) | 2 | 7.7 s/query |
| **Hybrid + Rerank** | **10 / 11** | **10 correct**, 1 refusal | 1 | 9.6 s/query |

<img src="screenshots/21_answer_evaluation.png" width="700">

Architecture diagram (details in the [technical report](group-activity/technical-report.md)):

<img src="group-activity/architecture-diagram.png" width="800">

---

## 5. Observations

**What worked**
- The local pipeline works end to end on a CPU-only laptop: answers take roughly 4 to 21 seconds per question and every answer lists its sources.
- Without retrieval the 3B model **hallucinated** the meaning of RAG. With retrieval and a "use only the context" prompt, it either answered from the documents or **refused** instead of inventing an answer.
- **Hybrid retrieval beat the dense baseline.** In run 1 MRR rose from 0.618 to 0.861 and the missing Lab sheet 1 question was found. BM25 catches exact words (such as "Duration" or "Programming Language") that the embedding model missed.
- **Reranking gave the biggest gain** in ranking quality (MRR 0.819 → 0.958 in run 2) for about 400 ms extra per query, which is small compared with the seconds the LLM needs.
- End to end, Hybrid + Rerank answered 10 of 11 questions correctly, against 6 fully correct for the baseline by my manual reading.

**What went wrong, and why**
- **The retriever matches meaning, not file names.** The question "…in Lab sheet 1" failed in dense mode because the words "Lab sheet 1" are not inside the chunk text (they are only in metadata).
- **Adding a document changed the results.** After uploading Lab sheet 5 (+10 chunks), dense Recall@5 fell from 0.92 to 0.83 and MRR from 0.618 to 0.507. The baseline even answered a Lab sheet 1 question with the pip-install commands from Lab sheet 5 (a *wrong-document* answer). This is the most likely cause, but I did not run a controlled test to prove it.
- **Failures happen in two places.** A retrieval failure (right chunk not found) and a generation failure (right chunk found, but the LLM refuses; see the Lab sheet 5 model-names question) need different fixes. The strict "reply exactly: I could not find this" instruction probably makes the small 3B model over-cautious. I did not test a softer prompt.
- **One regression:** for the "final presentation length" question the proposed system refused while the baseline answered correctly. I did not investigate the cause.
- **Low retrieval precision:** in the first hybrid test only 1 of the 4 retrieved chunks was relevant; the answer was correct only because that chunk was included.
- **PDF extraction limits:** two slides in Week 1 (pages 2 and 3, probably image-only) produced no text at all, and the command snippets in Lab sheet 5 did not appear in the extracted text. Lab sheet 2 also produced near-duplicate chunks because every group repeats the same "Your challenge…" paragraph.

**Limitations of my evaluation (please read the numbers with these in mind)**
- Only 12 retrieval questions and 11 answer questions, from a small corpus of 4 documents, one run each.
- I wrote the questions **after reading the chunks**, so many share exact words with the text. This favours keyword search (BM25). Heavily paraphrased questions would probably help dense retrieval more. I did not test this.
- The answer check is partly automatic (keywords) and partly my own reading, with no second grader.
- Everything ran on a CPU with a 3B model; a larger model might refuse less.

**Ideas for improvement (not tested)**
- Add the file name or section title to each chunk before embedding.
- Try the query-instruction format that Qwen3-Embedding is designed for.
- OCR for image-only slides; section-aware or parent/child chunking.
- A softer refusal prompt and a larger paraphrased evaluation set.

---

## 6. Learning Outcomes

- I can explain and build every stage of a RAG pipeline: loading, chunking, embedding, vector storage, retrieval, prompting and generation, and I understand why each stage exists.
- I understand why an LLM alone hallucinates and how grounding it in retrieved context, together with a "use only the context" instruction, reduces this.
- I learned the difference between dense retrieval (meaning) and lexical retrieval (BM25, exact words), how Reciprocal Rank Fusion combines them, and how a cross-encoder reranker improves the ranking.
- I can measure retrieval with Recall@5 and MRR, and measure the full system with answer correctness and latency, and I learned to report results and their limitations honestly.
- I learned to diagnose whether a bad answer comes from the **retriever** or the **generator** by inspecting the retrieved chunks and the extracted text.
- I gained practical experience running local models with Ollama, using ChromaDB and building a Streamlit chat application with PDF upload on a laptop without a GPU.
