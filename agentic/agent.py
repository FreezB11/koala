"""
agent.py — ReAct Agent (Reason + Act)

Each agent has:
  - An identity / role
  - A model to think with
  - Working memory (conversation context)
  - Episodic memory (disk persistence)
  - Access to tools (file I/O, shell, etc.)
  - Access to shared memory (cross-agent contracts)

The ReAct loop:
  1. Agent receives a task
  2. Agent THINKS (reasons about it)
  3. Agent ACTs (calls a tool or calls a contract)
  4. Agent OBSERVES (gets tool result)
  5. Repeat until done or max_steps
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from router import ModelRouter, Message
from memory import WorkingMemory, EpisodicMemory, SharedMemory
from tools import execute_tool, tool_schemas, ToolResult


# ─────────────────────────────────────────────
# Agent config
# ─────────────────────────────────────────────

@dataclass
class AgentConfig:
    agent_id: str
    role: str                          # e.g. "compiler", "tester", "planner"
    model_alias: str                   # from router.py aliases
    system_prompt: str = ""
    max_steps: int = 15                # Max ReAct iterations per task
    temperature: float = 0.7
    max_tokens: int = 2048
    memory_dir: str = "./memory"
    tools_enabled: list[str] = field(default_factory=list)  # Empty = all tools
    fallback_models: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# ReAct format parser
# ─────────────────────────────────────────────

REACT_SYSTEM_TEMPLATE = """You are {role}.

{system_prompt}

## Tools available
You can use the following tools by outputting JSON in this exact format:

```json
{{
  "thought": "your reasoning here",
  "action": "tool_name",
  "action_input": {{...tool arguments...}}
}}
```

When you have a final answer (no more tool calls needed), output:

```json
{{
  "thought": "your final reasoning",
  "action": "finish",
  "action_input": {{
    "result": "your final answer or summary"
  }}
}}
```

## Available tools
{tool_list}

## Shared contracts (functions you can reference)
{contracts}

## Rules
- Always output valid JSON in the code block
- Use tools when you need to read/write files, run code, or search
- One tool call per response
- Be precise and efficient
"""


def parse_react_output(text: str) -> Optional[dict]:
    """Extract the JSON action block from an agent response."""
    # Try to find ```json ... ``` block
    patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
        r"(\{[^{}]*\"action\"[^{}]*\})",  # fallback: inline JSON
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    # Last resort: try the whole text as JSON
    try:
        return json.loads(text.strip())
    except Exception:
        return None


# ─────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────

class Agent:
    """
    A single AI agent with ReAct capabilities.
    """

    def __init__(
        self,
        config: AgentConfig,
        router: ModelRouter,
        shared_memory: Optional[SharedMemory] = None,
    ):
        self.config = config
        self.router = router
        self.shared = shared_memory or SharedMemory(config.memory_dir)
        self.working = WorkingMemory(max_messages=40)
        self.episodic = EpisodicMemory(config.memory_dir, config.agent_id)
        self._build_system_prompt()

    def _build_system_prompt(self):
        tools = self.config.tools_enabled or list(__import__("tools").TOOL_REGISTRY.keys())
        tool_list = "\n".join(
            f"- **{name}**: {__import__('tools').TOOL_REGISTRY[name].description}"
            for name in tools
            if name in __import__("tools").TOOL_REGISTRY
        )
        contracts_text = self.shared.list_contracts()

        base = self.config.system_prompt or f"You are a helpful {self.config.role} agent."

        system = REACT_SYSTEM_TEMPLATE.format(
            role=self.config.role,
            system_prompt=base,
            tool_list=tool_list,
            contracts=contracts_text,
        )
        self.working.set_system(system)

    def refresh_contracts(self):
        """Refresh system prompt with latest contracts (call when new contracts added)."""
        self._build_system_prompt()

    # ── Core task execution ────────────────────────────────────────────

    def run(self, task: str, verbose: bool = True) -> str:
        """
        Run a task through the ReAct loop.
        Returns the final result string.
        """
        if verbose:
            print(f"\n{'═'*60}")
            print(f"  Agent: {self.config.agent_id} ({self.config.role})")
            print(f"  Model: {self.config.model_alias}")
            print(f"  Task:  {task[:80]}...")
            print(f"{'═'*60}")

        # Add task to working memory
        self.working.add("user", task)
        self.episodic.save("user", task)

        final_result = ""
        steps = 0

        while steps < self.config.max_steps:
            steps += 1
            if verbose:
                print(f"\n  [Step {steps}/{self.config.max_steps}]", end=" ", flush=True)

            # ── Think ────────────────────────────────────────────────
            try:
                response, model_used = self.router.send_with_fallback(
                    preferred_alias=self.config.model_alias,
                    messages=self.working.get(),
                    fallback_pool=self.config.fallback_models or None,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
            except Exception as e:
                error_msg = f"Model error: {e}"
                if verbose:
                    print(f"✗ {error_msg}")
                return error_msg

            if verbose and model_used != self.config.model_alias:
                print(f"(fallback→{model_used})", end=" ", flush=True)

            # Add raw response to working memory
            self.working.add("assistant", response)
            self.episodic.save("assistant", response, {"model": model_used, "step": steps})

            # ── Parse action ─────────────────────────────────────────
            action = parse_react_output(response)

            if action is None:
                # No JSON found — treat as a free-text answer
                if verbose:
                    print("(free text response)")
                final_result = response
                break

            thought = action.get("thought", "")
            action_name = action.get("action", "")
            action_input = action.get("action_input", {})

            if verbose:
                print(f"→ {action_name}")
                if thought:
                    print(f"    Thought: {thought[:100]}{'...' if len(thought) > 100 else ''}")

            # ── Finish ───────────────────────────────────────────────
            if action_name == "finish":
                final_result = action_input.get("result", response)
                if verbose:
                    print(f"  ✓ Done: {final_result[:120]}{'...' if len(final_result) > 120 else ''}")
                break

            # ── Execute tool ─────────────────────────────────────────
            tool_result = execute_tool(action_name, action_input)

            if verbose:
                status = "✓" if tool_result.success else "✗"
                print(f"    {status} {str(tool_result)[:150].replace(chr(10), ' ')}")

            # Feed observation back into context
            observation = f"Observation from {action_name}:\n{tool_result}"
            self.working.add("user", observation)
            self.episodic.save("user", observation, {"tool": action_name, "success": tool_result.success})

        else:
            final_result = f"Reached max steps ({self.config.max_steps}) without finishing."
            if verbose:
                print(f"  ⚠  {final_result}")

        return final_result

    # ── Chat mode (no ReAct loop) ──────────────────────────────────────

    def chat(self, message: str, verbose: bool = False) -> str:
        """Simple chat — no tool use, just conversation."""
        self.working.add("user", message)
        response, _ = self.router.send_with_fallback(
            preferred_alias=self.config.model_alias,
            messages=self.working.get(),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        self.working.add("assistant", response)
        return response

    def clear_context(self):
        self.working.clear(keep_system=True)
