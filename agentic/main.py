"""
main.py — Nexus Agent CLI

Modes:
  chat      — simple interactive chat with any model
  agent     — run a single ReAct agent on a task
  project   — demo multi-agent project with contract system
  models    — list available models

Usage:
  python main.py chat --model llama-3.3-70b
  python main.py agent --model gemma-3-27b --task "Write a Python fibonacci function to fib.py"
  python main.py project --demo compiler
"""

import sys
import os
import argparse

# ── Coloring ─────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
GREY   = "\033[90m"
MAGENTA= "\033[35m"

def c(text, color): return f"{color}{text}{RESET}"


# ─────────────────────────────────────────────
# Default model chain — ordered by reliability
# Edit this to match your working API keys
PREFERRED_CHAIN = [
    "groq-llama-70b",    # Groq — fast, reliable, 6000 RPM free
    "groq-mixtral",      # Groq fallback
    "mistral-small",     # Mistral fallback
    "codestral",         # Mistral code model
    "llama-3.3-70b",     # OpenRouter (rate-limited but free)
    "gemma-3-27b",       # OpenRouter fallback
    "hermes-3-405b",     # OpenRouter fallback
]
DEFAULT_MODEL = PREFERRED_CHAIN[0]

# ─────────────────────────────────────────────
# Chat mode
# ─────────────────────────────────────────────

def chat_mode(router, args):
    from memory import WorkingMemory
    from router import Message

    model = args.model or DEFAULT_MODEL
    available = router.available_models()
    if not available:
        print(c("No API keys found. Set at least OPENROUTER_API_KEY in .env", RED))
        sys.exit(1)

    if model not in available:
        print(c(f"Model '{model}' not available. Available: {', '.join(available[:5])}...", YELLOW))
        model = available[0]
        print(f"Using: {model}")

    mem = WorkingMemory()
    if args.system:
        mem.set_system(args.system)

    # Show active fallback chain (only models with valid keys)
    active_chain = [m for m in PREFERRED_CHAIN if m in available]
    chain_str = " → ".join(active_chain[:4])
    if len(active_chain) > 4:
        chain_str += f" (+{len(active_chain)-4} more)"

    print(c(f"\n  Nexus Chat", BOLD + CYAN))
    print(c(f"  Primary : {model}", GREEN))
    print(c(f"  Fallback: {chain_str}", GREY))
    print(c("  Commands: /model <alias>, /clear, /models, /exit, /chain", GREY))
    print(c("─" * 55, GREY))

    while True:
        try:
            user_input = input(c("\n You: ", GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input[1:].split()
            if cmd[0] == "exit":
                break
            elif cmd[0] == "clear":
                mem.clear()
                print(c("  Context cleared.", GREY))
            elif cmd[0] == "models":
                router.list_models()
            elif cmd[0] == "model" and len(cmd) > 1:
                model = cmd[1]
                print(c(f"  Switched to: {model}", GREY))
            elif cmd[0] == "history":
                for msg in mem.get():
                    print(c(f"  [{msg.role}]", GREY), msg.content[:80])
            elif cmd[0] == "chain":
                active = [m for m in PREFERRED_CHAIN if m in available]
                print(c("  Fallback chain:", GREY))
                for i, m in enumerate(active):
                    tag = " ← current" if m == model else ""
                    print(c(f"    {i+1}. {m}{tag}", GREEN if m == model else GREY))
            else:
                print(c(f"  Unknown command: /{cmd[0]}", YELLOW))
            continue

        mem.add("user", user_input)
        print(c("\n  Assistant: ", CYAN), end="", flush=True)

        try:
            response, used = router.send_with_fallback(
                preferred_alias=model,
                messages=mem.get(),
            )
            if used != model:
                print(c(f"(via {used}) ", GREY), end="")
            mem.add("assistant", response)
            print(response)
        except Exception as e:
            print(c(f"Error: {e}", RED))


# ─────────────────────────────────────────────
# Agent mode
# ─────────────────────────────────────────────

def agent_mode(router, args):
    from agent import Agent, AgentConfig

    model = args.model or "groq-llama-70b"
    task = args.task or "List the files in the current directory and describe what you see."

    config = AgentConfig(
        agent_id="cli_agent",
        role="general assistant",
        model_alias=model,
        system_prompt=args.system or "",
    )

    agent = Agent(config, router)
    result = agent.run(task, verbose=True)
    print(c("\n  Final result:", BOLD + GREEN))
    print(result)


# ─────────────────────────────────────────────
# Demo: Compiler project (make-belief contracts)
# ─────────────────────────────────────────────

def demo_compiler_project(router):
    """
    Demonstrates the contract/stub system by simulating a multi-agent
    compiler build project.

    Agents:
      - planner    — defines contracts, creates task graph
      - lexer      — implements tokenize()
      - parser     — uses tokenize(), implements parse()
      - codegen    — uses parse(), implements codegen()
      - tester     — generates unit tests for each contract
    """
    from orchestrator import Orchestrator, Task
    from agent import AgentConfig

    print(c("\n  DEMO: Multi-Agent Compiler Build\n", BOLD + MAGENTA))

    orc = Orchestrator(router)

    # ── Register agents ────────────────────────────────────────────────
    # Use the biggest free model for planning/complex work, smaller for simple tasks
    available = router.available_models()
    big = next((m for m in ["hermes-3-405b", "llama-3.3-70b", "gemma-3-27b"] if m in available), available[0] if available else None)
    small = next((m for m in ["qwen3-4b", "gemma-3-4b", "llama-3.2-3b"] if m in available), big)

    if not big:
        print(c("No models available — check your API keys!", RED))
        return

    print(f"  Using: big={big}, small={small}")

    orc.add_agent(AgentConfig(
        agent_id="planner",
        role="project planner",
        model_alias=big,
        system_prompt="You plan software projects. Break them into clear tasks with dependencies.",
    ))
    orc.add_agent(AgentConfig(
        agent_id="lexer",
        role="lexer engineer",
        model_alias=big,
        system_prompt="You implement tokenizers and lexers. Write clean, working Python code.",
        tools_enabled=["read_file", "write_file", "run_shell"],
    ))
    orc.add_agent(AgentConfig(
        agent_id="parser",
        role="parser engineer",
        model_alias=big,
        system_prompt="You build parsers and AST generators. Use the tokenize() contract provided.",
        tools_enabled=["read_file", "write_file", "run_shell"],
    ))
    orc.add_agent(AgentConfig(
        agent_id="tester",
        role="QA/test engineer",
        model_alias=small,
        system_prompt="You write thorough unit tests. Test the interface, not the implementation.",
    ))

    # ── Register contracts (stubs) ─────────────────────────────────────
    orc.register_contract(
        name="tokenize",
        signature="tokenize(source_code: str) -> list[Token]",
        description="Splits source code into a list of tokens. Each Token has type and value.",
        owner_agent_id="lexer",
        implemented=False,
    )
    orc.register_contract(
        name="parse",
        signature="parse(tokens: list[Token]) -> ASTNode",
        description="Builds an AST from tokens. Uses tokenize() internally.",
        owner_agent_id="parser",
        implemented=False,
    )

    # ── Build task graph ───────────────────────────────────────────────
    orc.add_task(Task(
        id="design",
        description="""
You are planning a simple arithmetic expression compiler in Python.
We support: numbers, +, -, *, /, parentheses.
Define the Token class and ASTNode class structures.
Write these to ./compiler/ast_nodes.py
Create the file with class definitions only (no logic).
""",
        assigned_to="planner",
    ))

    orc.add_task(Task(
        id="lexer_impl",
        description="""
Implement the tokenize() contract.
Read ./compiler/ast_nodes.py to understand the Token class.
Implement tokenize(source_code: str) -> list[Token] in ./compiler/lexer.py.
Test it works with: tokenize("3 + 4 * (2 - 1)")
""",
        assigned_to="lexer",
        depends_on=["design"],
    ))

    orc.add_task(Task(
        id="parser_impl",
        description="""
Implement the parse() contract.
Read ./compiler/ast_nodes.py for AST node types.
The tokenize() function is available in ./compiler/lexer.py.
Implement parse(tokens) -> ASTNode in ./compiler/parser.py.
Use recursive descent parsing.
""",
        assigned_to="parser",
        depends_on=["lexer_impl"],
    ))

    orc.add_task(Task(
        id="generate_tests",
        description="""
Generate unit tests for the tokenize() and parse() contracts.
Write tests to ./compiler/test_compiler.py.
Test: tokenize("1+2"), tokenize("(3*4)"), parse on simple expressions, edge cases.
""",
        assigned_to="tester",
        depends_on=["lexer_impl"],
    ))

    # ── Run the project ────────────────────────────────────────────────
    results = orc.run_all(verbose=True)
    orc.status()

    print(c("\n  Project files created:", BOLD + GREEN))
    from tools import list_directory
    r = list_directory("./compiler")
    print(r.output)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nexus Agent — free model router & multi-agent framework")
    parser.add_argument("mode", nargs="?", default="chat", choices=["chat", "agent", "project", "models"])
    parser.add_argument("--model", "-m", help="Model alias to use")
    parser.add_argument("--task", "-t", help="Task for agent mode")
    parser.add_argument("--system", "-s", help="System prompt")
    parser.add_argument("--demo", default="compiler", help="Demo project (compiler)")
    args = parser.parse_args()

    from router import ModelRouter
    router = ModelRouter()

    if args.mode == "models":
        router.list_models(show_all=True)
    elif args.mode == "chat":
        chat_mode(router, args)
    elif args.mode == "agent":
        agent_mode(router, args)
    elif args.mode == "project":
        if args.demo == "compiler":
            demo_compiler_project(router)


if __name__ == "__main__":
    main()