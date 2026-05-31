# 📄 Company Policy Q&A — RAG + FAISS

An intelligent document Q&A system built with **FastAPI + FAISS + Llama3 (Groq API)**. Upload any company policy PDF and ask natural language questions — answers are grounded strictly in the document, no hallucination. FAISS indexes persist to disk so embeddings survive server restarts.

---

## 🚀 What It Does

> Upload a PDF → Ask questions in natural language → Get accurate, document-grounded answers

**Example:**
- Upload: `HR_Policy_2024.pdf`
- Ask: *"How many days of annual leave am I entitled to?"*
- Answer: Pulled directly from the document with source chunks shown

---

## 🏗️ RAG Pipeline

```
PDF Upload
    │
    ▼
Text Extraction (pypdf) → Chunking (500 chars, 50 overlap)
    │
    ▼
Sentence Transformer Embeddings (all-MiniLM-L6-v2, 384-dim, normalized)
    │
    ▼
FAISS IndexFlatIP → Saved to disk (.faiss + .pkl)
    │
    ▼
User Question → Embed → FAISS Search (top-3 chunks)
    │
    ▼
Context + History → Groq API (Llama3) → Grounded Answer
```

1. **PDF Parsing** — Extracts text page by page, preserving page numbers
2. **Chunking** — Splits at sentence boundaries for cleaner context
3. **Embedding** — `all-MiniLM-L6-v2` converts chunks to normalized 384-dim float32 vectors
4. **FAISS Indexing** — `IndexFlatIP` (inner product = cosine sim for normalized vectors), saved to disk
5. **Persistence** — FAISS index auto-loads on server restart, no re-upload needed
6. **LLM Answer** — Top-3 chunks sent to Llama3 via Groq API with strict prompt to prevent hallucination
7. **Chat History** — Last 4 exchanges maintained per session, saved to disk as JSON

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python, Pydantic |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS (`IndexFlatIP`) |
| LLM | Llama3 via Groq API |
| PDF Parsing | pypdf |
| Frontend | HTML, CSS, JavaScript |

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/Sudharshanreddy136/policy-qa-rag.git
cd policy-qa-rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key
Get a free key from [console.groq.com](https://console.groq.com)
```bash
export GROQ_API_KEY="your_key_here"
```

### 4. Run the app
```bash
uvicorn main:app --reload
```

### 5. Open in browser
```
http://localhost:8000
```

---

## 📁 Project Structure

```
policy-qa-rag/
│
├── main.py                          # FastAPI app — all API routes (v2.0.0)
├── requirements.txt                 # All dependencies
├── README.md
│
├── static/
│   └── index.html                   # Frontend UI
│
├── faiss_indexes/                   # Persisted FAISS indexes (auto-created)
│   ├── <session_id>.faiss
│   └── <session_id>.pkl
│
├── chat_history/                    # Saved conversation history (auto-created)
│   └── <session_id>_history.json
│
└── utils/
    ├── __init__.py
    ├── faiss_store.py               # FAISS index build, search, save, load, delete
    ├── groq_client.py               # Groq API (Llama3) + prompt engineering
    └── pdf_parser.py                # PDF text extraction + chunking
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Frontend UI |
| `POST` | `/upload` | Upload PDF → build & save FAISS index |
| `POST` | `/ask` | Ask a question (auto-loads index if server restarted) |
| `POST` | `/load/{session_id}` | Load existing index from disk |
| `POST` | `/reset/{session_id}` | Clear conversation history |
| `DELETE` | `/session/{session_id}` | Delete session from memory and disk |
| `GET` | `/sessions` | List active in-memory sessions |
| `GET` | `/saved-indexes` | List all saved FAISS indexes on disk |

---

## 🧠 Key Design Decisions

- **FAISS IndexFlatIP** — Inner product search on normalized vectors equals cosine similarity; fast and accurate
- **Disk persistence** — FAISS index and chat history saved to disk; survives server restarts without re-uploading
- **Auto-load on `/ask`** — If session not in memory, automatically loads from disk; seamless UX
- **No hallucination** — Strict system prompt forces Llama3 to answer only from retrieved context
- **Normalized embeddings** — `normalize_embeddings=True` + `float32` cast for FAISS compatibility
- **Conversation history** — Last 4 exchanges sent as context for follow-up question support
- **Session management** — Multiple PDFs supported simultaneously via session IDs

---

## 📦 Requirements

```
fastapi
uvicorn
groq
sentence-transformers
faiss-cpu
pypdf
numpy
python-multipart
pydantic
```

---

## 👤 Author

**Sudharshan Reddy Dosti**  
Python Backend Developer | Gen AI & RAG Enthusiast

