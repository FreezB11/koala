# orchestrator.py - Main orchestrator agent (Nemo)

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from google import genai
from google.genai import types

from config import config
from memory import MemoryManager
from tools import (
    FUNCTION_MAP,
    OPENAI_TOOLS,
    GEMINI_TOOLS,
    ToolResult,
    read_file,
    write_file,
    list_files,
    run_command,
)
from ascii import asciii

# Colors
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
ORANGE = "\033[38;2;255;165;0m"
GREEN = "\033[92m"
DIM = "\033[2m"
RED = "\033[91m"
RESET = "\033[0m"

# System prompt for Nemo
SYSTEM_PROMPT = """You are Nemo, an orchestrator agent. You have two worker tools:
- ask_google_agent_1: Delegate to Google Gemini worker 1
- ask_google_agent_2: Delegate to Google Gemini worker 2

Plus local tools: read_file, write_file, list_files, run_command

Break the user's task down, delegate self-contained subtasks to the Gemini workers when that genuinely helps (you can call both in the same turn to run them in parallel), and use the file/shell tools yourself to write, save, or run code and inspect results. Only call a tool when it actually helps -- otherwise just answer directly.

Every turn, before doing anything else, output one short line stating your delegation decision -- e.g. 'delegation: not needed, this is a local file/compile task' or 'delegation: asking both Gemini workers for independent approaches to compare'. Then proceed. This line is for the user's visibility into your reasoning, not a tool call.

When you're done, reply with plain text and no further tool calls."""


@dataclass
class GoogleWorker:
    """A Google Gemini worker agent."""
    client: genai.Client
    label: str
    color: str
    system_prompt: str = "You are a helpful coding assistant. Be concise and practical."

    def run(self, prompt: str, stream: bool = True) -> str:
        """Run the worker on a prompt, streaming output."""
        print(f"{self.color}[{self.label}] {RESET}", end="", flush=True)

        response = self.client.models.generate_content(
            model=config.models.gemini_model,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=config.models.gemini_temperature,
                tools=GEMINI_TOOLS,
            ),
            stream=stream,
        )

        text = ""
        if stream:
            for chunk in response:
                if chunk.text:
                    print(f"{self.color}{chunk.text}{RESET}", end="", flush=True)
                    text += chunk.text
        else:
            text = response.text or ""
            print(f"{self.color}{text}{RESET}", end="", flush=True)

        print()
        return text.strip()


class Orchestrator:
    """Main orchestrator agent (Nemo)."""

    def __init__(self):
        self.setup_clients()
        self.setup_memory()
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.max_turns = config.agent.max_turns

    def setup_clients(self):
        """Initialize API clients."""
        # NVIDIA Nemotron client
        self.nvidia_client = OpenAI(
            base_url=config.models.nemo_base_url,
            api_key=config.models.nvidia_api_key,
        )

        # Google Gemini clients (two workers, potentially different keys)
        self.google_client_1 = genai.Client(api_key=config.models.google_api_key)
        self.google_client_2 = genai.Client(
            api_key=config.models.google_api_key_2 or config.models.google_api_key
        )

        # Worker agents
        self.worker_1 = GoogleWorker(
            self.google_client_1, "gemini-1", MAGENTA,
            "You are a helpful coding assistant. Be concise and practical. Use tools when needed."
        )
        self.worker_2 = GoogleWorker(
            self.google_client_2, "gemini-2", BLUE,
            "You are a helpful coding assistant. Be concise and practical. Use tools when needed."
        )

    def setup_memory(self):
        """Initialize memory system."""
        self.memory = MemoryManager()

    def should_consider_memory(self, text: str) -> bool:
        """Check if input should be stored in memory."""
        text = text.strip()
        if len(text) < config.agent.memory_trigger_threshold:
            return False
        junk = {"ok", "okay", "thanks", "thank you", "cool", "nice", "yep", "yes", "no", "hi", "hello"}
        return text.lower() not in junk

    def retrieve_memory_context(self, query: str) -> str:
        """Get relevant memories for a query."""
        if len(self.memory) == 0:
            return ""
        try:
            results = self.memory.search(query, k=config.memory.search_k)
            if not results:
                return ""
            lines = ["Relevant memories from previous conversations:"]
            for i, (score, text) in enumerate(results, 1):
                if score >= config.memory.search_score_threshold:
                    lines.append(f"\n[{i}] (relevance: {score:.2f})")
                    lines.append(text)
            return "\n".join(lines)
        except Exception as e:
            logging.error(f"Memory search error: {e}")
            return ""

    def stream_nemo_turn(self) -> tuple[str, List[Dict[str, Any]]]:
        """Stream one Nemo turn, return (reply_text, tool_calls)."""
        stream = self.nvidia_client.chat.completions.create(
            model=config.models.nemo_model,
            messages=self.messages,
            temperature=config.models.nemo_temperature,
            top_p=config.models.nemo_top_p,
            max_tokens=config.models.nemo_max_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": config.models.nemo_thinking},
                "reasoning_budget": config.models.nemo_reasoning_budget,
            },
            tools=OPENAI_TOOLS + self._get_google_tools(),
            tool_choice="auto",
            stream=True,
        )

        reply_text = ""
        tool_calls_acc = {}

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Reasoning content
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                print(f"{DIM}{reasoning}{RESET}", end="", flush=True)

            # Regular content
            if delta.content:
                print(delta.content, end="", flush=True)
                reply_text += delta.content

            # Tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    slot = tool_calls_acc.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["name"] += tc.function.name
                        if tc.function.arguments:
                            slot["arguments"] += tc.function.arguments

        print()
        tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc.keys())]
        return reply_text, tool_calls

    def _get_google_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions for Google workers."""
        desc = (
            "Delegate a subtask, question, or piece of work to a Google Gemini "
            "worker agent. Use this to parallelize independent subtasks, get a "
            "second pass on something, or offload part of the task. Send it a "
            "clear, self-contained instruction -- it has no memory of this "
            "conversation beyond what you put in `message`."
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": "ask_google_agent_1",
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Self-contained task or question for this worker."
                            }
                        },
                        "required": ["message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_google_agent_2",
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Self-contained task or question for this worker."
                            }
                        },
                        "required": ["message"],
                    },
                },
            },
        ]

    def execute_tool(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """Execute a local tool."""
        if name in {"ask_google_agent_1", "ask_google_agent_2"}:
            worker = self.worker_1 if name == "ask_google_agent_1" else self.worker_2
            return ToolResult(True, worker.run(args.get("message", "")))

        fn = FUNCTION_MAP.get(name)
        if not fn:
            return ToolResult(False, error=f"Unknown tool: {name}")

        try:
            return fn(**args)
        except Exception as e:
            return ToolResult(False, error=str(e))

    def run_turn(self, task: str) -> str:
        """Run a complete turn with the orchestrator."""
        # Add memory context if available
        memory_context = self.retrieve_memory_context(task)
        enhanced_task = task
        if memory_context:
            enhanced_task = f"{memory_context}\n\nCurrent user message:\n\n{task}"

        self.messages.append({"role": "user", "content": enhanced_task})

        for round_no in range(1, self.max_turns + 1):
            if round_no > 1:
                print(f"{ORANGE}[nemo/round {round_no}]{RESET}")

            reply_text, tool_calls = self.stream_nemo_turn()

            if not tool_calls:
                # No tool calls, we're done
                self.messages.append({"role": "assistant", "content": reply_text})
                return reply_text

            # Add assistant message with tool calls
            assistant_msg = {"role": "assistant", "content": reply_text}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls
            ]
            self.messages.append(assistant_msg)

            # Execute each tool call
            for tc in tool_calls:
                fn_name = tc["name"]
                try:
                    fn_args = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    fn_args = {}

                print(f"{CYAN}[tool call: {fn_name}({fn_args})]{RESET}")

                result = self.execute_tool(fn_name, fn_args)

                print(f"{CYAN}[tool result: {str(result)[:200]}]{RESET}")

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": fn_name,
                    "content": str(result),
                })

        return "[orchestrator] did not resolve in max rounds"

    def run_interactive(self):
        """Run interactive REPL."""
        asciii()
        print(f"{GREEN}Nemo Orchestrator ready. Type 'exit' to quit.{RESET}\n")

        while True:
            try:
                inp = input(f"{ORANGE}orch $> {RESET}").strip()

                if not inp:
                    continue

                # Built-in commands
                if inp.lower() == "exit":
                    self.memory.save()
                    break

                if inp.lower() == "/memory":
                    self.show_memory()
                    continue

                if inp.lower() == "/memory count":
                    print(f"Total Memories: {len(self.memory)}")
                    continue

                if inp.lower().startswith("/memory search "):
                    query = inp[len("/memory search "):]
                    self.search_memory(query)
                    continue

                if inp.lower() == "/memory clear":
                    self.memory.clear()
                    print("Memory cleared.")
                    continue

                if inp.lower() == "/history":
                    self.show_history()
                    continue

                if inp.lower() == "/help":
                    self.show_help()
                    continue

                # Run the turn
                result = self.run_turn(inp)
                print()

                # Store in memory if appropriate
                if self.should_consider_memory(inp):
                    action = self.memory.process(inp)
                    print(f"{GREEN}[memory: {action}]{RESET}")
                    self.memory.save()

            except KeyboardInterrupt:
                print("\nSaving memory...")
                self.memory.save()
                break

            except Exception as e:
                print(f"\n{RED}Error: {e}{RESET}")
                logging.exception("Error in main loop")

    def show_memory(self):
        """Display all memories."""
        print("\n=== LONG TERM MEMORY ===")
        memories = self.memory.get_all_memories()
        if not memories:
            print("No memories stored.")
        else:
            for i, mem in enumerate(memories, 1):
                print(f"\n[{i}] (accessed: {mem.access_count}x)")
                print(mem.text)
        print(f"\nTotal Memories: {len(self.memory)}\n")

    def search_memory(self, query: str):
        """Search memories."""
        results = self.memory.search(query, k=10)
        print(f"\n=== SEARCH RESULTS FOR '{query}' ===")
        for score, text in results:
            print(f"\nScore: {score:.4f}")
            print(text[:300] + ("..." if len(text) > 300 else ""))
        print()

    def show_history(self):
        """Show conversation history."""
        print("\n=== CONVERSATION HISTORY ===")
        for i, msg in enumerate(self.messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "tool":
                print(f"  [{i}] TOOL: {msg.get('name')} -> {content[:100]}")
            elif role == "assistant" and msg.get("tool_calls"):
                print(f"  [{i}] ASSISTANT (with {len(msg['tool_calls'])} tool calls): {content[:100]}")
            else:
                print(f"  [{i}] {role.upper()}: {content[:100]}")
        print()

    def show_help(self):
        """Show help."""
        print(f"""
{GREEN}Available Commands:{RESET}
  /help           - Show this help
  /history        - Show conversation history
  /memory         - Show all memories
  /memory count   - Show memory count
  /memory search <query> - Search memories
  /memory clear   - Clear all memories
  exit            - Exit and save memory

{GREEN}Delegation:{RESET}
  The orchestrator will automatically delegate to Gemini workers when helpful.
  You can also explicitly ask: "ask agent 1 to..." or "ask both agents to..."

{GREEN}Local Tools:{RESET}
  read_file(path), write_file(path, content), list_files(path), run_command(cmd)
""")


def main():
    """Entry point."""
    # Validate config
    errors = config.validate()
    if errors:
        print(f"{RED}Configuration errors:{RESET}")
        for e in errors:
            print(f"  - {e}")
        return

    orchestrator = Orchestrator()
    orchestrator.run_interactive()


if __name__ == "__main__":
    main()