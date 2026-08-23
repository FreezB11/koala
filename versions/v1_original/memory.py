# memory.py - Long-term vector store
#
# NOTE: The MemoryManager class that used to be duplicated here has been
# removed. It was never imported by anything (orch.py / agent0.py both
# import MemoryManager from manager.py), so it was dead, confusing code.
# manager.py's MemoryManager is now the single canonical implementation.

import os
import pickle
from typing import List, Tuple
from dataclasses import dataclass, field

import faiss
from sentence_transformers import SentenceTransformer

from config import config


@dataclass
class MemoryEntry:
    """A single memory entry."""
    text: str
    embedding: List[float] = field(default_factory=list)
    access_count: int = 0
    created_at: float = field(default_factory=lambda: __import__('time').time())


class LocalVectorStore:
    """Vector store for memory using FAISS."""

    def __init__(self):
        try:
            self.model = SentenceTransformer(config.models.embedding_model)
        except Exception as e:
            raise RuntimeError(
                f"Could not load embedding model '{config.models.embedding_model}'. "
                "If you're offline, pre-download it once with network access, or set "
                "HF_HUB_OFFLINE=1 after caching it locally so this doesn't hang "
                f"retrying HTTP calls. Original error: {e}"
            ) from e
        self.dim = self.model.get_sentence_embedding_dimension()

        os.makedirs(config.memory.memory_dir, exist_ok=True)

        index_path = os.path.join(config.memory.memory_dir, "index.faiss")
        docs_path = os.path.join(config.memory.memory_dir, "documents.pkl")

        if os.path.exists(index_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(index_path)
            with open(docs_path, "rb") as f:
                self.documents: List[MemoryEntry] = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.documents = []

    def add(self, text: str) -> int:
        """Add a new memory."""
        emb = self.model.encode([text], normalize_embeddings=True)
        self.index.add(emb)
        entry = MemoryEntry(text=text, embedding=emb[0].tolist())
        self.documents.append(entry)
        return len(self.documents) - 1

    def search(self, query: str, k: int = 5) -> List[Tuple[float, str]]:
        """Search for similar memories. Returns (score, text) pairs."""
        if len(self.documents) == 0:
            return []

        emb = self.model.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(emb, min(k, len(self.documents)))

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if 0 <= idx < len(self.documents):
                entry = self.documents[idx]
                entry.access_count += 1
                results.append((float(score), entry.text))

        return results

    def get_all(self) -> List[MemoryEntry]:
        return self.documents

    def save(self):
        index_path = os.path.join(config.memory.memory_dir, "index.faiss")
        docs_path = os.path.join(config.memory.memory_dir, "documents.pkl")

        faiss.write_index(self.index, index_path)
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    def clear(self):
        self.index = faiss.IndexFlatIP(self.dim)
        self.documents = []
        self.save()

    def __len__(self) -> int:
        return len(self.documents)