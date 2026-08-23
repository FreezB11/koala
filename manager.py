# manager.py - MemoryManager (v3: improved merge, summarization, better error handling)

import logging
from typing import Optional, List

from memory import LocalVectorStore, MemoryEntry
from config import config
from retry import retry_with_backoff

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(
        self,
        duplicate_threshold: float = None,
        related_threshold: float = None,
    ):
        self.store = LocalVectorStore()

        self.duplicate_threshold = duplicate_threshold if duplicate_threshold is not None else config.memory.duplicate_threshold
        self.related_threshold = related_threshold if related_threshold is not None else config.memory.related_threshold

    def process(self, text: str) -> str:
        """
        Process new text into memory.
        Returns: 'duplicate' | 'update' | 'new'
        """
        if len(self.store) == 0:
            self.store.add(text)
            return "new"

        matches = self.store.search(text, k=1)

        if not matches:
            self.store.add(text)
            return "new"

        score, memory_text = matches[0]

        logger.debug(f"Best match score: {score:.4f}")
        logger.debug(f"Memory text: {memory_text[:100]}...")

        # Exact duplicate
        if score >= self.duplicate_threshold:
            logger.info("Duplicate detected, skipping")
            return "duplicate"

        # Related memory -> merge
        if score >= self.related_threshold:
            merged = self.merge(memory_text, text)

            idx = self._find_index(memory_text)
            if idx == -1:
                logger.warning("Could not find memory to update, adding as new")
                self.store.add(text)
                return "new"

            self._set_document_text(idx, merged)
            self.store.rebuild_index()
            logger.info("Memory updated (merged)")
            return "update"

        # Brand new topic
        self.store.add(text)
        return "new"

    def _find_index(self, text: str) -> int:
        for i, doc in enumerate(self.store.documents):
            doc_text = getattr(doc, "text", doc)
            if doc_text == text:
                return i
        return -1

    def _set_document_text(self, idx: int, new_text: str) -> None:
        doc = self.store.documents[idx]
        if hasattr(doc, "text"):
            doc.text = new_text
            doc.updated_at = __import__('time').time()
            doc.source_count += 1
        else:
            self.store.documents[idx] = new_text

    def merge(self, old_memory: str, new_memory: str) -> str:
        """
        Merge strategy: append with separator.
        Could be replaced with LLM summarization later.
        """
        return f"{old_memory}\n---\n{new_memory}"

    def search(self, query: str, k: int = None) -> list:
        """Search memories."""
        if k is None:
            k = config.memory.search_k
        return self.store.search(query, k=k)

    def get_all_memories(self) -> list:
        """Get all memory entries."""
        return self.store.get_all()

    def get_all_texts(self) -> list:
        """Get all memory texts."""
        return self.store.get_all_texts()

    def save(self):
        self.store.save()

    def clear(self):
        self.store.clear()

    def __len__(self) -> int:
        return len(self.store)

    def __bool__(self) -> bool:
        return bool(self.store)
    
    # v3: Summarization methods
    def summarize_old_memories(self, max_memories: int = 10) -> int:
        """Summarize the oldest non-summary memories."""
        from agents.nemo import create_nemotron_agent
        
        # Find candidates for summarization
        candidates = [
            (i, doc) for i, doc in enumerate(self.store.documents)
            if not doc.is_summary and len(doc.text) > config.memory.max_memory_length
        ]
        
        if not candidates:
            return 0
            
        # Sort by age (oldest first)
        candidates.sort(key=lambda x: x[1].created_at)
        candidates = candidates[:max_memories]
        
        # For now, just truncate - in future use LLM
        summarized = 0
        for i, doc in candidates:
            if len(doc.text) > config.memory.max_memory_length:
                truncated = doc.text[:config.memory.max_memory_length // 2] + \
                           "\n... [summarized] ...\n" + \
                           doc.text[-config.memory.max_memory_length // 2:]
                doc.text = truncated
                doc.is_summary = True
                doc.updated_at = __import__('time').time()
                summarized += 1
        
        if summarized > 0:
            self.store.rebuild_index()
            self.save()
            logger.info(f"Summarized {summarized} memories")
        
        return summarized