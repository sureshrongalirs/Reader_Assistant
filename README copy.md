# StudyRAG — PDF Intelligence Platform

Analytical Q&A over up to 20 PDFs using **LlamaIndex + CrewAI + Groq**, served through a single **Streamlit** app. No backend server needed.

---

## Features

| Capability | Detail |
|---|---|
| **Pre-loaded PDFs** | Drop up to 20 PDFs in `data/preloaded_pdfs/` — indexed once, persisted to disk |
| **Live uploads** | Upload PDFs directly from the UI — added to the same persistent index |
| **Skip re-indexing** | Files are hash-checked — already-indexed files are never re-embedded |
| **4 query modes** | Q&A · Summarise · Compare · Extract Data — auto-detected or user-selected |
| **Source citations** | Every answer cites which PDF(s) it came from |
| **Confidence scores** | High / Medium / Low based on retrieval quality |
| **Persistent index** | ChromaDB on disk — survives app restarts, no re-embedding needed |

---

## Project Structure

```
studyrag/
├── app.py                        # Streamlit entry point
├── requirements.txt
├── .env.example                  # Copy to .env and fill in keys
│
├── data/
│   └── preloaded_pdfs/           # ← Place your 20 PDFs here
│
├── vector_store/                 # ← ChromaDB writes here (auto-created)
│
└── src/
    ├── config/
    │   └── settings.py           # Pydantic settings, reads from .env
    │
    ├── ingestion/
    │   └── ingestion.py          # PDF loading, chunking, embedding, dedup
    │
    └── agents/
        ├── rag.py                # LlamaIndex RAG query layer
        └── crew.py               # CrewAI agents (QA / Summarise / Compare / Extract)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — add your GROQ_API_KEY
```

### 3. Add your PDFs
```bash
# Copy up to 20 PDFs into:
data/preloaded_pdfs/
```

### 4. Run the app
```bash
streamlit run app.py
```

### 5. First-time indexing
- Open the app in your browser
- Enter your Groq API key in the sidebar (or set it in `.env`)
- Click **"🔄 Index Pre-loaded PDFs"** — this runs once; subsequent restarts skip already-indexed files
- Start asking questions!

---

## Query Modes

| Mode | Auto-triggered by | Example |
|---|---|---|
| **Q&A** | Default | *"What is the conclusion of report X?"* |
| **Summarise** | "summarise", "overview", "brief" | *"Summarise the key findings"* |
| **Compare** | "compare", "vs", "difference", "contrast" | *"Compare results in doc A vs doc B"* |
| **Extract Data** | "how many", "list all", "statistics", "%" | *"List all revenue figures mentioned"* |

---

## Configuration (`.env`)

```env
GROQ_API_KEY=your_key_here

PRELOADED_PDFS_DIR=data/preloaded_pdfs
VECTOR_STORE_DIR=vector_store
COLLECTION_NAME=studyrag_docs

MODEL_NAME=llama-3.3-70b-versatile
MODEL_TEMPERATURE=0.0

CHUNK_SIZE=1024
CHUNK_OVERLAP=100
```

---

## How It Works

```
PDFs on disk
    │
    ▼
SimpleDirectoryReader (LlamaIndex)
    │  reads text from PDF pages
    ▼
SimpleNodeParser
    │  splits into 1024-token chunks, 100-token overlap
    ▼
HuggingFaceEmbedding
    │  converts chunks to vectors (free, local)
    ▼
ChromaDB PersistentClient  ──── saved to vector_store/ ────
    │
    │  (on query)
    ▼
LlamaIndex VectorStoreIndex.as_query_engine(top_k=5)
    │  retrieves most relevant chunks
    ▼
CrewAI Agent  (QA / Summarise / Compare / Extract)
    │  structures, cites, and scores the answer
    ▼
Streamlit UI
```

---

## Tips for Best Results

- **Chunk size**: Default 1024 works well for narrative PDFs. For data-heavy PDFs (tables, financials), try `CHUNK_SIZE=512`.
- **Top-K**: Compare and Extract modes use `top_k=7` automatically for wider coverage.
- **Re-indexing**: If you update a PDF, delete the `vector_store/` directory and re-index.
- **Model**: `llama-3.3-70b-versatile` gives the best analytical answers; `llama3-8b-8192` is fastest.
