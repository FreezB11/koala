# tools.py - Local file/shell tools with proper structure

import subprocess
import os
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

from config import config


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str
    error: str = ""

    def __str__(self) -> str:
        if self.success:
            return self.output
        return f"Error: {self.error}\nOutput: {self.output}"

    def __bool__(self) -> bool:
        return self.success


def read_file(path: str) -> ToolResult:
    """Read contents of a file."""
    try:
        # Security: restrict to current working directory
        abs_path = os.path.abspath(path)
        cwd = os.path.abspath(os.getcwd())
        if not abs_path.startswith(cwd):
            return ToolResult(False, "", f"Access denied: path outside working directory")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return ToolResult(True, content)
    except FileNotFoundError:
        return ToolResult(False, "", f"File not found: {path}")
    except Exception as e:
        return ToolResult(False, "", str(e))


def write_file(path: str, content: str) -> ToolResult:
    """Write content to a file."""
    try:
        # Security: restrict to current working directory
        abs_path = os.path.abspath(path)
        cwd = os.path.abspath(os.getcwd())
        if not abs_path.startswith(cwd):
            return ToolResult(False, "", f"Access denied: path outside working directory")

        # Create parent directories if needed
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResult(True, "File written successfully")
    except Exception as e:
        return ToolResult(False, "", str(e))


def list_files(path: str = ".") -> ToolResult:
    """List files in a directory."""
    try:
        abs_path = os.path.abspath(path)
        cwd = os.path.abspath(os.getcwd())
        if not abs_path.startswith(cwd):
            return ToolResult(False, "", f"Access denied: path outside working directory")

        entries = os.listdir(abs_path)
        files = []
        dirs = []
        for entry in sorted(entries):
            full = os.path.join(abs_path, entry)
            if os.path.isdir(full):
                dirs.append(f"{entry}/")
            else:
                size = os.path.getsize(full)
                files.append(f"{entry} ({size} bytes)")

        output = "Directories:\n" + ("\n".join(dirs) if dirs else "(none)")
        output += "\n\nFiles:\n" + ("\n".join(files) if files else "(none)")
        return ToolResult(True, output)
    except Exception as e:
        return ToolResult(False, "", str(e))


def run_command(command: str, timeout: int = 60) -> ToolResult:
    """Run shell command with timeout."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd()
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr] {result.stderr}"
        if result.returncode != 0:
            return ToolResult(False, output, f"Exit code: {result.returncode}")
        return ToolResult(True, output)
    except subprocess.TimeoutExpired:
        return ToolResult(False, "", f"Command timed out after {timeout}s")
    except Exception as e:
        return ToolResult(False, "", str(e))


# Function map for execution
FUNCTION_MAP = {
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
    "run_command": run_command,
}

# OpenAI-compatible tool definitions (for NVIDIA Nemotron)
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    },
]

# Google Gemini tool definitions
GEMINI_TOOLS = [
    {
        "type": "function",
        "name": "read_file",
        "description": "Read contents of a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "type": "function",
        "name": "list_files",
        "description": "List files in a directory",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": []
        }
    },
    {
        "type": "function",
        "name": "run_command",
        "description": "Run shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }
    },
]