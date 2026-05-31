import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from utils.pdf_parser import extract_text_from_pdf, chunk_text
from utils.faiss_store import FAISSStore
from utils.groq_client import ask_groq

app = FastAPI(title="Company Policy Q&A — FAISS Edition", version="2.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory session store
sessions = {}

HISTORY_DIR = "chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)


# ── History helpers ──────────────────────────────────────────

def history_path(session_id: str) -> str:
    return os.path.join(HISTORY_DIR, f"{session_id}_history.json")

def load_history(session_id: str) -> list:
    path = history_path(session_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def save_history(session_id: str, history: list) -> None:
    with open(history_path(session_id), "w") as f:
        json.dump(history, f, indent=2)


# ── Routes ───────────────────────────────────────────────────

@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload PDF → extract text → chunk → build FAISS index → save to disk
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()

    os.makedirs("uploads", exist_ok=True)
    temp_path = f"uploads/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)

    raw_text = extract_text_from_pdf(temp_path)
    os.remove(temp_path)

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    chunks = chunk_text(raw_text, chunk_size=500, overlap=50)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks created from PDF.")

    session_id = file.filename.replace(".pdf", "").replace(" ", "_")
    store = FAISSStore(session_id=session_id)
    store.build(chunks)
    store.save()

    history = load_history(session_id)

    sessions[session_id] = {
        "store": store,
        "history": history,
        "filename": file.filename,
    }

    return {
        "session_id": session_id,
        "filename": file.filename,
        "chunks_created": len(chunks),
        "vectors_in_faiss": store.index.ntotal,
        "message": "PDF processed! FAISS index built and saved to disk."
    }


@app.post("/load/{session_id}")
def load_existing_session(session_id: str):
    """Load a previously saved FAISS index from disk — no re-upload needed."""
    if session_id in sessions:
        return {"message": "Session already loaded.", "session_id": session_id}

    store = FAISSStore(session_id=session_id)
    if not store.load():
        raise HTTPException(status_code=404, detail="No saved index found for this session_id.")

    history = load_history(session_id)
    sessions[session_id] = {
        "store": store,
        "history": history,
        "filename": f"{session_id}.pdf",
    }

    return {
        "session_id": session_id,
        "vectors_loaded": store.index.ntotal,
        "history_turns": len(history),
        "message": "FAISS index loaded from disk successfully!"
    }


class QuestionRequest(BaseModel):
    session_id: str
    question: str


@app.post("/ask")
def ask_question(req: QuestionRequest):
    """
    Ask question → embed → FAISS search → send to Groq → return answer
    """
    # Auto-load from disk if not in memory (handles server restarts)
    if req.session_id not in sessions:
        store = FAISSStore(session_id=req.session_id)
        if store.load():
            history = load_history(req.session_id)
            sessions[req.session_id] = {
                "store": store,
                "history": history,
                "filename": f"{req.session_id}.pdf",
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="Session not found. Please upload a PDF first."
            )

    session = sessions[req.session_id]
    store: FAISSStore = session["store"]
    history: list = session["history"]

    results = store.search(req.question, top_k=3)
    if not results:
        raise HTTPException(status_code=400, detail="No relevant chunks found.")

    context = "\n\n---\n\n".join([r["chunk"] for r in results])

    history_text = ""
    for h in history[-4:]:
        history_text += f"User: {h['question']}\nAssistant: {h['answer']}\n\n"

    answer = ask_groq(
        question=req.question,
        context=context,
        history=history_text,
        filename=session["filename"]
    )

    history.append({"question": req.question, "answer": answer})
    session["history"] = history
    save_history(req.session_id, history)

    return {
        "answer": answer,
        "sources": [r["chunk"] for r in results],
        "scores": [round(r["score"], 4) for r in results],
        "session_id": req.session_id
    }


@app.post("/reset/{session_id}")
def reset_history(session_id: str):
    """Clear conversation history."""
    if session_id in sessions:
        sessions[session_id]["history"] = []
    save_history(session_id, [])
    return {"message": "Chat history cleared."}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete session from memory and disk."""
    if session_id in sessions:
        sessions[session_id]["store"].delete()
        del sessions[session_id]
    path = history_path(session_id)
    if os.path.exists(path):
        os.remove(path)
    return {"message": f"Session '{session_id}' deleted."}


@app.get("/sessions")
def list_sessions():
    return {
        "sessions": [
            {
                "session_id": k,
                "filename": v["filename"],
                "vectors": v["store"].index.ntotal if v["store"].index else 0,
                "history_turns": len(v["history"])
            }
            for k, v in sessions.items()
        ]
    }


@app.get("/saved-indexes")
def list_saved_indexes():
    index_dir = "faiss_indexes"
    if not os.path.exists(index_dir):
        return {"indexes": []}
    files = [f.replace(".faiss", "") for f in os.listdir(index_dir) if f.endswith(".faiss")]
    return {"saved_indexes": files}
