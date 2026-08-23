# tools.py - Local file/shell tools (v3: with retry, better error handling)

import subprocess
import os
import json
import shlex
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

from config import config
from retry import retry_with_backoff, RetryPolicy

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str
    error: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def __str__(self) -> str:
        if self.success:
            return self.output
        return f"Error: {self.error}\nOutput: {self.output}"

    def __bool__(self) -> bool:
        return self.success

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata
        }


def _validate_path(path: str) -> tuple[bool, str]:
    """Validate path is within working directory. Returns (is_valid, abs_path_or_error)."""
    try:
        abs_path = os.path.abspath(path)
        cwd = os.path.abspath(os.getcwd())
        if not abs_path.startswith(cwd):
            return False, f"Access denied: path outside working directory"
        return True, abs_path
    except Exception as e:
        return False, str(e)


def read_file(path: str) -> ToolResult:
    """Read contents of a file."""
    valid, result = _validate_path(path)
    if not valid:
        return ToolResult(False, "", result)

    try:
        abs_path = result
        # Check file size
        size_mb = os.path.getsize(abs_path) / (1024 * 1024)
        if size_mb > config.tools.max_file_size_mb:
            return ToolResult(False, "", f"File too large: {size_mb:.1f}MB > {config.tools.max_file_size_mb}MB limit")

        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        return ToolResult(True, content, metadata={"size_bytes": len(content), "path": abs_path})
    except FileNotFoundError:
        return ToolResult(False, "", f"File not found: {path}")
    except Exception as e:
        logger.error(f"read_file error: {e}")
        return ToolResult(False, "", str(e))


def write_file(path: str, content: str) -> ToolResult:
    """Write content to a file."""
    valid, result = _validate_path(path)
    if not valid:
        return ToolResult(False, "", result)

    try:
        abs_path = result
        # Create parent directories if needed
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)

        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResult(True, "File written successfully", metadata={"size_bytes": len(content), "path": abs_path})
    except Exception as e:
        logger.error(f"write_file error: {e}")
        return ToolResult(False, "", str(e))


def list_files(path: str = ".") -> ToolResult:
    """List files in a directory."""
    valid, result = _validate_path(path)
    if not valid:
        return ToolResult(False, "", result)

    try:
        abs_path = result
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
        return ToolResult(True, output, metadata={"dirs": len(dirs), "files": len(files), "path": abs_path})
    except Exception as e:
        logger.error(f"list_files error: {e}")
        return ToolResult(False, "", str(e))


# Retry policy for commands (only for transient failures)
command_retry_policy = RetryPolicy(
    max_retries=2,
    base_delay=0.5,
    max_delay=5.0,
    retryable_status_codes=()  # Commands don't have HTTP status codes
)


def run_command(command: str, timeout: int = None) -> ToolResult:
    """Run shell command with timeout (no shell=True for security)."""
    if timeout is None:
        timeout = config.tools.command_timeout

    # Parse command safely without shell=True
    try:
        args = shlex.split(command)
    except ValueError as e:
        return ToolResult(False, "", f"Invalid command syntax: {e}")

    # Validate command is allowed
    if args:
        base_cmd = os.path.basename(args[0])
        if base_cmd not in config.tools.allowed_commands:
            return ToolResult(False, "", f"Command not allowed: {base_cmd}. Allowed: {config.tools.allowed_commands}")

    try:
        result = subprocess.run(
            args,
            shell=False,  # Security: no shell=True
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd()
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr] {result.stderr}"
        if result.returncode != 0:
            return ToolResult(False, output, f"Exit code: {result.returncode}", metadata={"exit_code": result.returncode})
        return ToolResult(True, output, metadata={"exit_code": 0})
    except subprocess.TimeoutExpired:
        return ToolResult(False, "", f"Command timed out after {timeout}s")
    except FileNotFoundError:
        return ToolResult(False, "", f"Command not found: {args[0]}")
    except Exception as e:
        logger.error(f"run_command error: {e}")
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
            "description": "Run shell command (restricted to allowed commands)",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    },
]

# Google Gemini tool definitions (flat format for ggl.py conversion)
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
        "description": "Run shell command (restricted to allowed commands)",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }
    },
]