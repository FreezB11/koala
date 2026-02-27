"""
memory.py — Layered memory system for agents

Three layers:
  1. WorkingMemory  — current conversation window (like your Go Conversation struct)
  2. EpisodicMemory — JSON file persistence per session/agent
  3. SharedMemory   — key-value store shared between agents (contracts, stubs, artifacts)
"""

import os
import json
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from router import Message


# ─────────────────────────────────────────────
# Working Memory (in-context window)
# ─────────────────────────────────────────────

class WorkingMemory:
    """
    Sliding window conversation history.
    Mirrors your Go Conversation struct — keeps last N messages, preserving system prompt.
    """

    def __init__(self, max_messages: int = 30, max_tokens_estimate: int = 4000):
        self.messages: list[Message] = []
        self.max_messages = max_messages
        self.max_tokens_estimate = max_tokens_estimate  # Rough guard

    def add(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))
        self._trim()

    def set_system(self, prompt: str):
        """Set or replace system prompt at position 0."""
        if self.messages and self.messages[0].role == "system":
            self.messages[0] = Message(role="system", content=prompt)
        else:
            self.messages.insert(0, Message(role="system", content=prompt))

    def _trim(self):
        """Trim oldest messages, preserving system prompt."""
        while len(self.messages) > self.max_messages:
            # Find first non-system message and remove it
            for i, msg in enumerate(self.messages):
                if msg.role != "system":
                    self.messages.pop(i)
                    break
            else:
                break  # Only system messages left

    def get(self) -> list[Message]:
        return self.messages.copy()

    def clear(self, keep_system: bool = True):
        if keep_system and self.messages and self.messages[0].role == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []

    def token_estimate(self) -> int:
        """Very rough token estimate (chars / 4)."""
        return sum(len(m.content) for m in self.messages) // 4

    def __len__(self):
        return len(self.messages)


# ─────────────────────────────────────────────
# Episodic Memory (disk persistence)
# ─────────────────────────────────────────────

@dataclass
class Episode:
    """A recorded interaction."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    agent_id: str = ""
    role: str = ""
    content: str = ""
    metadata: dict = field(default_factory=dict)


class EpisodicMemory:
    """
    Persists agent conversations to disk as JSONL files.
    One file per agent, append-only.
    """

    def __init__(self, storage_dir: str = "./memory", agent_id: str = "default"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.agent_id = agent_id
        self.filepath = self.storage_dir / f"{agent_id}.jsonl"

    def save(self, role: str, content: str, metadata: dict = None):
        ep = Episode(
            agent_id=self.agent_id,
            role=role,
            content=content,
            metadata=metadata or {},
        )
        with open(self.filepath, "a") as f:
            f.write(json.dumps(asdict(ep)) + "\n")

    def load_recent(self, n: int = 20) -> list[Episode]:
        """Load last n episodes from disk."""
        if not self.filepath.exists():
            return []
        episodes = []
        with open(self.filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(Episode(**json.loads(line)))
        return episodes[-n:]

    def to_messages(self, n: int = 20) -> list[Message]:
        """Convert recent episodes back to Message objects for context injection."""
        return [
            Message(role=ep.role, content=ep.content)
            for ep in self.load_recent(n)
            if ep.role in ("user", "assistant", "system")
        ]

    def search(self, keyword: str) -> list[Episode]:
        """Simple keyword search through episodic memory."""
        if not self.filepath.exists():
            return []
        results = []
        keyword_lower = keyword.lower()
        with open(self.filepath) as f:
            for line in f:
                line = line.strip()
                if line and keyword_lower in line.lower():
                    results.append(Episode(**json.loads(line)))
        return results

    def clear(self):
        if self.filepath.exists():
            self.filepath.unlink()


# ─────────────────────────────────────────────
# Shared Memory (cross-agent key-value store)
# ─────────────────────────────────────────────

class SharedMemory:
    """
    Key-value store shared between all agents.
    Used for:
      - Function contracts/stubs  (agents share interface, not implementation)
      - Build artifacts           (files produced by one agent, used by another)
      - Project state             (current task graph, completed nodes)
      - Test results              (pass/fail per unit test)

    This is the "make-belief" system — one agent can say:
      "I have F(source_code) -> compiled_binary. Use it."
    Without sharing how F is implemented.
    """

    def __init__(self, storage_dir: str = "./memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.storage_dir / "shared.json"
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self):
        if self.filepath.exists():
            with open(self.filepath) as f:
                self._data = json.load(f)

    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump(self._data, f, indent=2)

    def set(self, key: str, value: Any, author: str = "system"):
        self._data[key] = {
            "value": value,
            "author": author,
            "timestamp": time.time(),
        }
        self._save()

    def get(self, key: str, default=None) -> Any:
        entry = self._data.get(key)
        if entry is None:
            return default
        return entry["value"]

    def get_with_meta(self, key: str) -> Optional[dict]:
        return self._data.get(key)

    def keys(self, prefix: str = "") -> list[str]:
        return [k for k in self._data.keys() if k.startswith(prefix)]

    def delete(self, key: str):
        if key in self._data:
            del self._data[key]
            self._save()

    # ── Contract / Stub system ─────────────────────────────────────────

    def register_contract(
        self,
        name: str,
        signature: str,
        description: str,
        owner_agent: str,
        implemented: bool = False,
    ):
        """
        Register a function contract (stub).
        One agent says: "I own F(a, b) -> x, described as..."
        Other agents can call it without knowing the implementation.

        Example:
            shared.register_contract(
                name="compile",
                signature="compile(source_code: str) -> binary: bytes",
                description="Compiles C source code to a binary using GCC",
                owner_agent="compiler_agent",
                implemented=False,  # Stub — not built yet
            )
        """
        self.set(f"contract:{name}", {
            "signature": signature,
            "description": description,
            "owner": owner_agent,
            "implemented": implemented,
            "calls": [],
        }, author=owner_agent)

    def get_contracts(self) -> dict[str, dict]:
        """Return all registered contracts."""
        return {
            k.replace("contract:", ""): self.get(k)
            for k in self.keys("contract:")
        }

    def call_contract(self, name: str, caller: str, args: dict) -> dict:
        """
        Log a contract call (for orchestration tracking).
        The actual execution happens in the agent that owns it.
        """
        contract = self.get(f"contract:{name}")
        if not contract:
            raise KeyError(f"Contract '{name}' not found in shared memory")

        call_record = {
            "caller": caller,
            "args": args,
            "timestamp": time.time(),
            "result": None,
        }
        contract["calls"].append(call_record)
        self.set(f"contract:{name}", contract, author=caller)
        return call_record

    def list_contracts(self) -> str:
        """Return a formatted string of all contracts for agent system prompts."""
        contracts = self.get_contracts()
        if not contracts:
            return "No contracts registered yet."
        lines = ["Available function contracts:", ""]
        for name, c in contracts.items():
            status = "✓ implemented" if c.get("implemented") else "⚙ stub"
            lines.append(f"  {name}({c['signature']}) [{status}]")
            lines.append(f"    Owner: {c['owner']}")
            lines.append(f"    {c['description']}")
            lines.append("")
        return "\n".join(lines)

    # ── Project artifacts ──────────────────────────────────────────────

    def store_artifact(self, name: str, content: str, agent: str, artifact_type: str = "file"):
        """Store a build artifact (file, test, module, etc.)."""
        self.set(f"artifact:{name}", {
            "content": content,
            "type": artifact_type,
            "agent": agent,
            "size": len(content),
        }, author=agent)

    def get_artifact(self, name: str) -> Optional[str]:
        art = self.get(f"artifact:{name}")
        return art["content"] if art else None

    def list_artifacts(self) -> list[str]:
        return [k.replace("artifact:", "") for k in self.keys("artifact:")]
