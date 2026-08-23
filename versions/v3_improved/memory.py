# memory.py - Long-term vector store (v3: with summarization, better error handling)

import os
import pickle
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import config
from retry import retry_with_backoff

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry."""
    text: str
    embedding: List[float] = field(default_factory=list)
    access_count: int = 0
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())
    # v3: summarization metadata
    is_summary: bool = False
    source_count: int = 1  # How many memories were merged into this


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
            try:
                self.index = faiss.read_index(index_path)
                with open(docs_path, "rb") as f:
                    self.documents: List[MemoryEntry] = pickle.load(f)
                logger.info(f"Loaded {len(self.documents)} memories from disk")
            except Exception as e:
                logger.warning(f"Failed to load memory, starting fresh: {e}")
                self.index = faiss.IndexFlatIP(self.dim)
                self.documents = []
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.documents = []

    def add(self, text: str) -> int:
        """Add a new memory."""
        emb = self.model.encode([text], normalize_embeddings=True)
        self.index.add(emb)
        entry = MemoryEntry(text=text, embedding=emb[0].tolist())
        self.documents.append(entry)
        
        # Check if we should trigger summarization
        if len(self.documents) >= config.memory.summarization_threshold:
            self._maybe_summarize()
            
        return len(self.documents) - 1

    def _maybe_summarize(self):
        """Summarize old memories if we have too many."""
        if len(self.documents) < config.memory.summarization_threshold:
            return
            
        # Find memories that are long and not already summaries
        candidates = [
            (i, doc) for i, doc in enumerate(self.documents)
            if not doc.is_summary and len(doc.text) > config.memory.max_memory_length
        ]
        
        if not candidates:
            return
            
        logger.info(f"Summarizing {len(candidates)} long memories...")
        # For now, just truncate very long memories
        # In the future, this could use an LLM to create proper summaries
        for i, doc in candidates[:5]:  # Limit to 5 per trigger
            if len(doc.text) > config.memory.max_memory_length:
                # Keep first and last parts, add summary marker
                truncated = doc.text[:config.memory.max_memory_length // 2] + \
                           "\n... [truncated] ...\n" + \
                           doc.text[-config.memory.max_memory_length // 2:]
                doc.text = truncated
                doc.is_summary = True
                doc.updated_at = datetime.now().timestamp()
        
        self.rebuild_index()

    def search(self, query: str, k: int = 5) -> List[Tuple[float, str]]:
        """Search for similar memories. Returns (score, text) pairs."""
        if len(self.documents) == 0:
            return []

        try:
            emb = self.model.encode([query], normalize_embeddings=True)
            scores, ids = self.index.search(emb, min(k, len(self.documents)))

            results = []
            for score, idx in zip(scores[0], ids[0]):
                if 0 <= idx < len(self.documents):
                    entry = self.documents[idx]
                    entry.access_count += 1
                    results.append((float(score), entry.text))

            return results
        except Exception as e:
            logger.error(f"Memory search error: {e}")
            return []

    def get_all(self) -> List[MemoryEntry]:
        return self.documents

    def get_all_texts(self) -> List[str]:
        return [getattr(doc, "text", doc) for doc in self.documents]

    def save(self):
        index_path = os.path.join(config.memory.memory_dir, "index.faiss")
        docs_path = os.path.join(config.memory.memory_dir, "documents.pkl")

        try:
            faiss.write_index(self.index, index_path)
            with open(docs_path, "wb") as f:
                pickle.dump(self.documents, f)
            logger.info(f"Saved {len(self.documents)} memories to disk")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def clear(self):
        self.index = faiss.IndexFlatIP(self.dim)
        self.documents = []
        self.save()

    def rebuild_index(self):
        """Rebuild FAISS index from current documents."""
        import faiss
        self.index = faiss.IndexFlatIP(self.dim)
        if not self.documents:
            return

        texts = [getattr(doc, "text", doc) for doc in self.documents]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        self.index.add(embeddings)

    def __len__(self) -> int:
        return len(self.documents)

    def __bool__(self) -> bool:
        return len(self.documents) > 0