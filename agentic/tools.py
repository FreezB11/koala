"""
tools.py — Agent tools
Agents call these tools via the tool-use loop.
Each tool returns a structured result dict.
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Any, Optional
import traceback


# ─────────────────────────────────────────────
# Tool definition
# ─────────────────────────────────────────────

@dataclass
class ToolResult:
    success: bool
    output: str
    error: str = ""

    def __str__(self):
        if self.success:
            return f"[OK]\n{self.output}"
        return f"[ERROR] {self.error}\n{self.output}"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON Schema for parameters
    fn: Callable[..., ToolResult]

    def to_schema(self) -> dict:
        """Describe this tool to an LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# ─────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────

def read_file(path: str, start_line: int = 1, end_line: int = -1) -> ToolResult:
    """Read a file from disk, optionally sliced by line range."""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return ToolResult(False, "", f"File not found: {path}")
        if not p.is_file():
            return ToolResult(False, "", f"Not a file: {path}")

        lines = p.read_text(errors="replace").splitlines(keepends=True)
        if end_line == -1:
            end_line = len(lines)
        sliced = lines[start_line - 1: end_line]
        content = "".join(sliced)

        return ToolResult(
            True,
            f"[{path}] lines {start_line}-{end_line} ({len(sliced)} lines)\n\n{content}",
        )
    except Exception as e:
        return ToolResult(False, "", str(e))


def write_file(path: str, content: str, append: bool = False) -> ToolResult:
    """Write content to a file. Creates parent dirs automatically."""
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        p.write_text(content) if not append else open(p, "a").write(content)
        return ToolResult(True, f"Written {len(content)} chars to {path}")
    except Exception as e:
        return ToolResult(False, "", str(e))


def list_directory(path: str = ".", pattern: str = "*") -> ToolResult:
    """List files in a directory, optionally filtered by glob pattern."""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return ToolResult(False, "", f"Directory not found: {path}")
        files = sorted(p.glob(pattern))
        lines = []
        for f in files:
            size = f.stat().st_size if f.is_file() else 0
            kind = "DIR" if f.is_dir() else f.suffix or "FILE"
            lines.append(f"  {kind:<8} {size:>8} bytes   {f.name}")
        output = f"Directory: {path}\n" + "\n".join(lines) if lines else f"Empty: {path}"
        return ToolResult(True, output)
    except Exception as e:
        return ToolResult(False, "", str(e))


def run_shell(command: str, cwd: str = ".", timeout: int = 30) -> ToolResult:
    """
    Run a shell command. Agents use this to compile, test, run scripts.
    ⚠  Runs in a subprocess — not sandboxed. Add restrictions as needed.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            return ToolResult(
                False,
                output,
                f"Command exited with code {result.returncode}",
            )
        return ToolResult(True, output or "(no output)")
    except subprocess.TimeoutExpired:
        return ToolResult(False, "", f"Command timed out after {timeout}s")
    except Exception as e:
        return ToolResult(False, "", str(e))


def search_codebase(query: str, path: str = ".", file_pattern: str = "*.py") -> ToolResult:
    """Simple grep-style search through project files."""
    try:
        matches = []
        p = Path(path).expanduser()
        for filepath in sorted(p.rglob(file_pattern)):
            try:
                lines = filepath.read_text(errors="replace").splitlines()
                for i, line in enumerate(lines, 1):
                    if query.lower() in line.lower():
                        matches.append(f"{filepath}:{i}:  {line.rstrip()}")
            except Exception:
                continue

        if not matches:
            return ToolResult(True, f"No matches for '{query}' in {path}")
        output = f"Matches for '{query}' in {path}:\n" + "\n".join(matches[:50])
        if len(matches) > 50:
            output += f"\n... and {len(matches) - 50} more"
        return ToolResult(True, output)
    except Exception as e:
        return ToolResult(False, "", str(e))


def create_directory(path: str) -> ToolResult:
    """Create a directory (and parents)."""
    try:
        Path(path).expanduser().mkdir(parents=True, exist_ok=True)
        return ToolResult(True, f"Created directory: {path}")
    except Exception as e:
        return ToolResult(False, "", str(e))


def delete_file(path: str) -> ToolResult:
    """Delete a file (not a directory)."""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return ToolResult(False, "", f"File not found: {path}")
        if p.is_dir():
            return ToolResult(False, "", f"Use remove_directory for directories: {path}")
        p.unlink()
        return ToolResult(True, f"Deleted: {path}")
    except Exception as e:
        return ToolResult(False, "", str(e))


def python_eval(code: str) -> ToolResult:
    """
    Safely evaluate Python expressions or small scripts.
    Good for math, JSON transforms, quick checks.
    """
    try:
        import io
        import contextlib
        output = io.StringIO()
        local_vars = {}
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": __builtins__}, local_vars)
        result = output.getvalue()
        if not result and local_vars:
            # Show last defined variable
            last = list(local_vars.values())[-1]
            result = repr(last)
        return ToolResult(True, result or "(executed, no output)")
    except Exception as e:
        return ToolResult(False, "", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────

TOOL_REGISTRY: dict[str, Tool] = {

    "read_file": Tool(
        name="read_file",
        description="Read a file from disk. Optionally slice by line range.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
                "start_line": {"type": "integer", "description": "First line (1-indexed)", "default": 1},
                "end_line": {"type": "integer", "description": "Last line (-1 for all)", "default": -1},
            },
            "required": ["path"],
        },
        fn=read_file,
    ),

    "write_file": Tool(
        name="write_file",
        description="Write or append content to a file. Creates parent dirs if needed.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"},
                "append": {"type": "boolean", "description": "Append instead of overwrite", "default": False},
            },
            "required": ["path", "content"],
        },
        fn=write_file,
    ),

    "list_directory": Tool(
        name="list_directory",
        description="List files in a directory.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path", "default": "."},
                "pattern": {"type": "string", "description": "Glob filter (e.g. '*.py')", "default": "*"},
            },
        },
        fn=list_directory,
    ),

    "run_shell": Tool(
        name="run_shell",
        description="Run a shell command (compile, test, build, etc.).",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "cwd": {"type": "string", "description": "Working directory", "default": "."},
                "timeout": {"type": "integer", "description": "Max seconds", "default": 30},
            },
            "required": ["command"],
        },
        fn=run_shell,
    ),

    "search_codebase": Tool(
        name="search_codebase",
        description="Search for a string in project files.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term"},
                "path": {"type": "string", "description": "Root path to search", "default": "."},
                "file_pattern": {"type": "string", "description": "Glob pattern", "default": "*.py"},
            },
            "required": ["query"],
        },
        fn=search_codebase,
    ),

    "create_directory": Tool(
        name="create_directory",
        description="Create a directory.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to create"},
            },
            "required": ["path"],
        },
        fn=create_directory,
    ),

    "python_eval": Tool(
        name="python_eval",
        description="Execute a Python expression or short script and return the output.",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to run"},
            },
            "required": ["code"],
        },
        fn=python_eval,
    ),
}


def get_tool(name: str) -> Optional[Tool]:
    return TOOL_REGISTRY.get(name)


def tool_schemas() -> list[dict]:
    """Return all tool schemas for injection into agent system prompts."""
    return [t.to_schema() for t in TOOL_REGISTRY.values()]


# Common argument aliases — LLMs often use these wrong names
ARG_ALIASES = {
    "write_file":      {"file_path": "path", "filename": "path", "file_content": "content", "text": "content"},
    "read_file":       {"file_path": "path", "filename": "path"},
    "list_directory":  {"directory": "path", "dir": "path", "folder": "path"},
    "run_shell":       {"cmd": "command", "shell_command": "command", "script": "command"},
    "search_codebase": {"term": "query", "search_term": "query", "keyword": "query"},
    "python_eval":     {"expression": "code", "script": "code", "python_code": "code"},
}

def _normalize_kwargs(tool_name: str, kwargs: dict) -> dict:
    """Remap common LLM argument name mistakes to the correct names."""
    aliases = ARG_ALIASES.get(tool_name, {})
    normalized = {}
    for k, v in kwargs.items():
        normalized[aliases.get(k, k)] = v
    return normalized


def execute_tool(name: str, kwargs: dict) -> ToolResult:
    """Execute a tool by name with given kwargs."""
    tool = TOOL_REGISTRY.get(name)
    if not tool:
        return ToolResult(False, "", f"Unknown tool: {name}")
    try:
        return tool.fn(**_normalize_kwargs(name, kwargs))
    except TypeError as e:
        return ToolResult(False, "", f"Bad arguments for {name}: {e}")
    except Exception as e:
        return ToolResult(False, "", f"Tool error in {name}: {e}")