import faiss
import numpy as np
import pickle
import os
from typing import List
from sentence_transformers import SentenceTransformer

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("Loading embedding model (first time ~80MB download)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded!")
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Convert list of texts into normalized embedding matrix."""
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.astype("float32")


class FAISSStore:
    """
    Wraps a FAISS index with original text chunks.
    Supports save/load to disk so embeddings persist across server restarts.
    """

    def __init__(self, session_id: str, index_dir: str = "faiss_indexes"):
        self.session_id = session_id
        self.index_dir = index_dir
        self.index = None
        self.chunks: List[str] = []
        self.dimension = 384
        os.makedirs(index_dir, exist_ok=True)

    @property
    def index_path(self) -> str:
        return os.path.join(self.index_dir, f"{self.session_id}.faiss")

    @property
    def chunks_path(self) -> str:
        return os.path.join(self.index_dir, f"{self.session_id}.pkl")

    def build(self, chunks: List[str]) -> None:
        """Embed all chunks and build FAISS flat index."""
        self.chunks = chunks
        embeddings = embed_texts(chunks)
        # IndexFlatIP = Inner Product search (equals cosine sim for normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        print(f"FAISS index built: {self.index.ntotal} vectors stored")

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        """Embed query and retrieve top_k most similar chunks."""
        if self.index is None or self.index.ntotal == 0:
            return []
        query_embedding = embed_texts([query])
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(score),
                    "index": int(idx)
                })
        return results

    def save(self) -> None:
        """Save FAISS index and chunks to disk."""
        if self.index is None:
            return
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"FAISS index saved: {self.index_path}")

    def load(self) -> bool:
        """Load FAISS index and chunks from disk."""
        if not os.path.exists(self.index_path):
            return False
        self.index = faiss.read_index(self.index_path)
        with open(self.chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"FAISS index loaded: {self.index.ntotal} vectors")
        return True

    def delete(self) -> None:
        """Delete index files from disk."""
        for path in [self.index_path, self.chunks_path]:
            if os.path.exists(path):
                os.remove(path)
