# manager.py - MemoryManager (v3: improved merge, LLM summarization, better error handling)

import logging
import json
from typing import Optional, List

from memory import LocalVectorStore, MemoryEntry
from config import config
from retry import retry_with_backoff, RetryPolicy

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
        self.summarization_retry_policy = RetryPolicy(
            max_retries=3, base_delay=1.0, max_delay=30.0
        )

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
        """Summarize the oldest non-summary memories using LLM."""
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
        
        # Use LLM to create proper summaries
        summarized = 0
        for i, doc in candidates:
            if len(doc.text) > config.memory.max_memory_length:
                summary = self._llm_summarize(doc.text)
                if summary:
                    doc.text = summary
                    doc.is_summary = True
                    doc.updated_at = __import__('time').time()
                    summarized += 1
        
        if summarized > 0:
            self.store.rebuild_index()
            self.save()
            logger.info(f"Summarized {summarized} memories using LLM")
        
        return summarized

    def _llm_summarize(self, text: str) -> Optional[str]:
        """Use LLM to create a concise summary of long memory text."""
        try:
            agent = create_nemotron_agent(
                system_prompt="You are a memory summarization assistant. Create a concise summary that preserves key facts, decisions, and context. Keep it under 2000 characters."
            )
            
            prompt = f"""Summarize the following memory content, preserving key facts, decisions, and context:

{text}

Provide a concise summary (under 2000 chars):"""
            
            messages = [{"role": "user", "content": prompt}]
            
            def _summarize():
                return agent.complete(messages)
            
            response = self.summarization_retry_policy.execute(_summarize)
            
            if response and response.choices and response.choices[0].message.content:
                summary = response.choices[0].message.content.strip()
                if len(summary) < len(text) * 0.8:  # Only use if actually shorter
                    return f"[SUMMARY] {summary}"
            
            return None
        except Exception as e:
            logger.warning(f"LLM summarization failed, falling back to truncation: {e}")
            # Fallback to truncation
            if len(text) > config.memory.max_memory_length:
                return text[:config.memory.max_memory_length // 2] + \
                       "\n... [truncated] ...\n" + \
                       text[-config.memory.max_memory_length // 2:]
            return None

    def auto_summarize_if_needed(self) -> int:
        """Automatically summarize if we have too many memories."""
        if len(self.store) >= config.memory.summarization_threshold:
            return self.summarize_old_memories(max_memories=5)
        return 0

    def get_memory_stats(self) -> dict:
        """Get statistics about memory store."""
        total = len(self.store)
        summaries = sum(1 for doc in self.store.documents if doc.is_summary)
        total_chars = sum(len(getattr(doc, "text", doc)) for doc in self.store.documents)
        avg_access = sum(doc.access_count for doc in self.store.documents) / max(total, 1)
        
        return {
            "total_memories": total,
            "summaries": summaries,
            "regular_memories": total - summaries,
            "total_characters": total_chars,
            "avg_access_count": avg_access,
        }