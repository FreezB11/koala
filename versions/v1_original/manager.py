# manager.py
#
# This is now the ONE canonical MemoryManager. The duplicate MemoryManager
# that used to live in memory.py has been removed (it was dead code -- every
# caller imports MemoryManager from here, not from memory.py).

from memory import LocalVectorStore
from config import config


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
        Returns: 'duplicate' | 'update' | 'new'
        """
        if len(self.store.documents) == 0:
            self.store.add(text)
            return "new"

        matches = self.store.search(text, k=1)

        if not matches:
            self.store.add(text)
            return "new"

        score, memory_text = matches[0]

        print(f"\nBest Match ({score:.4f})")
        print(memory_text)

        # Exact duplicate
        if score >= self.duplicate_threshold:
            return "duplicate"

        # Related memory -> merge
        if score >= self.related_threshold:
            merged = self.merge(memory_text, text)

            # FIX: `self.store.documents` holds MemoryEntry objects (or
            # strings, depending on which LocalVectorStore you're using) --
            # `.index(memory_text)` on a *string* against a list of entry
            # objects would always raise ValueError. Find by comparing the
            # underlying text field instead.
            idx = self._find_index(memory_text)
            if idx == -1:
                # couldn't find it (shouldn't normally happen) -- fall back
                # to just adding the new text rather than crashing.
                self.store.add(text)
                return "new"

            self._set_document_text(idx, merged)
            self.rebuild_index()
            return "update"

        # Brand new topic
        self.store.add(text)
        return "new"

    def _find_index(self, text: str) -> int:
        for i, doc in enumerate(self.store.documents):
            doc_text = getattr(doc, "text", doc)  # works whether doc is a
            if doc_text == text:                   # MemoryEntry or a raw str
                return i
        return -1

    def _set_document_text(self, idx: int, new_text: str) -> None:
        doc = self.store.documents[idx]
        if hasattr(doc, "text"):
            doc.text = new_text
        else:
            self.store.documents[idx] = new_text

    def merge(self, old_memory: str, new_memory: str) -> str:
        """
        Simple merge strategy.
        Replace later with LLM summarization.
        """
        return f"{old_memory}\n---\n{new_memory}"

    def rebuild_index(self):
        import faiss

        dim = self.store.model.get_sentence_embedding_dimension()
        self.store.index = faiss.IndexFlatIP(dim)

        if not self.store.documents:
            return

        texts = [getattr(doc, "text", doc) for doc in self.store.documents]
        embeddings = self.store.model.encode(texts, normalize_embeddings=True)
        self.store.index.add(embeddings)

    def save(self):
        self.store.save()