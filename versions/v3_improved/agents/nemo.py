# nemo.py - NVIDIA Nemotron agent wrapper (v3: with retry, health checks, multi-provider support)

from openai import OpenAI
from openai.types.chat import ChatCompletionChunk
from typing import List, Dict, Any, Tuple, Optional
import json
import logging

from tools import OPENAI_TOOLS, FUNCTION_MAP, ToolResult
from config import config
from retry import retry_with_backoff, RetryPolicy

logger = logging.getLogger(__name__)


class NemotronAgent:
    """Wrapper for NVIDIA Nemotron API with tool support (primary orchestrator)."""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        system_prompt: str = ""
    ):
        self.client = OpenAI(
            base_url=base_url or config.models.nemo_base_url,
            api_key=api_key or config.nvidia_api_key
        )
        self.model = model or config.models.nemo_model
        self.system_prompt = system_prompt
        self.tools = OPENAI_TOOLS
        self.retry_policy = RetryPolicy(
            max_retries=config.agent.max_retries,
            base_delay=config.agent.retry_base_delay,
            max_delay=config.agent.retry_max_delay,
        )

    def _build_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build messages for Nemotron API."""
        result = [{"role": "system", "content": self.system_prompt}]

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                continue  # Already added

            if role == "tool":
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "name": msg.get("name", ""),
                    "content": msg.get("content", "")
                })
            elif role == "assistant" and msg.get("tool_calls"):
                result.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": msg["tool_calls"]
                })
            else:
                result.append({"role": role, "content": content})

        return result

    @retry_with_backoff(
        max_retries=5,
        base_delay=1.0,
        max_delay=60.0,
        exceptions=(Exception,),
        should_retry=lambda e: hasattr(e, 'status_code') and e.status_code in (429, 500, 502, 503, 504)
    )
    def stream(self, messages: List[Dict[str, Any]]) -> ChatCompletionChunk:
        """Stream a completion with retry."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(messages),
            tools=self.tools,
            tool_choice="auto",
            temperature=config.models.nemo_temperature,
            top_p=config.models.nemo_top_p,
            max_tokens=config.models.nemo_max_tokens,
            stream=True
        )

    @retry_with_backoff(
        max_retries=5,
        base_delay=1.0,
        max_delay=60.0,
        exceptions=(Exception,),
        should_retry=lambda e: hasattr(e, 'status_code') and e.status_code in (429, 500, 502, 503, 504)
    )
    def complete(self, messages: List[Dict[str, Any]]) -> Any:
        """Non-streaming completion with retry."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(messages),
            tools=self.tools,
            tool_choice="auto",
            temperature=config.models.nemo_temperature,
            top_p=config.models.nemo_top_p,
            max_tokens=config.models.nemo_max_tokens,
            stream=False
        )


class OpenAICompatibleWorker:
    """Generic worker for any OpenAI-compatible API (Mistral, Groq, Together, etc.)."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        label: str,
        color: str,
        system_prompt: str = "You are a helpful coding assistant. Be concise and practical.",
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.label = label
        self.color = color
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_policy = RetryPolicy(
            max_retries=config.agent.max_retries,
            base_delay=config.agent.retry_base_delay,
            max_delay=config.agent.retry_max_delay,
        )

    def run(self, prompt: str, stream: bool = True) -> str:
        """Run the worker on a prompt, streaming output."""
        print(f"{self.color}[{self.label}] \033[0m", end="", flush=True)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        def _run_with_retry():
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream
            )

        try:
            response = self.retry_policy.execute(_run_with_retry)
        except Exception as e:
            logger.error(f"Worker {self.label} failed after retries: {e}")
            return f"[Worker {self.label} error: {e}]"

        text = ""
        if stream:
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(f"{self.color}{content}\033[0m", end="", flush=True)
                    text += content
        else:
            text = response.choices[0].message.content or ""
            print(f"{self.color}{text}\033[0m", end="", flush=True)

        print()
        return text.strip()

    def complete(self, prompt: str) -> str:
        """Non-streaming completion."""
        return self.run(prompt, stream=False)


def create_mistral_worker(
    api_key: str = None,
    label: str = "mistral",
    color: str = "\033[93m",  # Yellow
    system_prompt: str = "You are a helpful coding assistant. Be concise and practical."
) -> OpenAICompatibleWorker:
    """Create a Mistral worker."""
    return OpenAICompatibleWorker(
        api_key=api_key or config.mistral_api_key_1,
        base_url=config.models.mistral_base_url,
        model=config.models.mistral_model,
        label=label,
        color=color,
        system_prompt=system_prompt,
        temperature=config.models.mistral_temperature,
        max_tokens=config.models.mistral_max_tokens,
    )


def create_groq_worker(
    api_key: str = None,
    label: str = "groq",
    color: str = "\033[96m",  # Cyan
    system_prompt: str = "You are a helpful coding assistant. Be concise and practical."
) -> OpenAICompatibleWorker:
    """Create a Groq worker."""
    return OpenAICompatibleWorker(
        api_key=api_key or config.groq_api_key,
        base_url=config.models.groq_base_url,
        model=config.models.groq_model,
        label=label,
        color=color,
        system_prompt=system_prompt,
        temperature=config.models.groq_temperature,
        max_tokens=config.models.groq_max_tokens,
    )


def create_together_worker(
    api_key: str = None,
    label: str = "together",
    color: str = "\033[38;2;255;165;0m",  # Orange
    system_prompt: str = "You are a helpful coding assistant. Be concise and practical."
) -> OpenAICompatibleWorker:
    """Create a Together AI worker."""
    return OpenAICompatibleWorker(
        api_key=api_key or config.together_api_key,
        base_url=config.models.together_base_url,
        model=config.models.together_model,
        label=label,
        color=color,
        system_prompt=system_prompt,
        temperature=config.models.together_temperature,
        max_tokens=config.models.together_max_tokens,
    )


def stream_nemotron_turn(
    client: OpenAI,
    memory: List[Dict[str, Any]],
    input_text: str,
    tools: List[Dict[str, Any]],
    stream: bool = True
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Stream one Nemotron turn against the running messages history.
    Returns (reply_text, tool_calls) where tool_calls is a list of
    {"id", "name", "arguments"} dicts (arguments = raw JSON string).
    """
    messages = list(memory)
    if input_text:
        messages = messages + [{"role": "user", "content": input_text}]

    resp_stream = client.chat.completions.create(
        model=config.models.nemo_model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=config.models.nemo_temperature,
        top_p=config.models.nemo_top_p,
        max_tokens=config.models.nemo_max_tokens,
        stream=True
    )

    reply_text = ""
    tool_calls_acc = {}

    for chunk in resp_stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            print(f"\033[2m{reasoning}\033[0m", end="", flush=True)

        if delta.content:
            print(delta.content, end="", flush=True)
            reply_text += delta.content

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


def nvidia_nemo(
    client: OpenAI,
    memory: List[Dict[str, Any]],
    input: str = "",
    tools: List[Dict[str, Any]] = None,
    stream: bool = True,
):
    """
    Wrapper matching how orchestrator calls this:
        nvidia_nemo(nvidia_client, memory=messages, input="", tools=ALL_TOOLS, stream=True)

    `memory` is the full running OpenAI-style message list (already including
    the latest user turn), `input` is an optional extra piece of text to append.

    Returns a raw stream object when stream=True (orchestrator iterates it chunk
    by chunk itself), matching the previous behavior.
    """
    messages = list(memory)
    if input:
        messages = messages + [{"role": "user", "content": input}]

    if stream:
        return client.chat.completions.create(
            model=config.models.nemo_model,
            messages=messages,
            tools=tools or OPENAI_TOOLS,
            tool_choice="auto",
            temperature=config.models.nemo_temperature,
            top_p=config.models.nemo_top_p,
            max_tokens=config.models.nemo_max_tokens,
            stream=True,
        )

    return client.chat.completions.create(
        model=config.models.nemo_model,
        messages=messages,
        tools=tools or OPENAI_TOOLS,
        tool_choice="auto",
        temperature=config.models.nemo_temperature,
        top_p=config.models.nemo_top_p,
        max_tokens=config.models.nemo_max_tokens,
        stream=False,
    )


def create_nemotron_agent(
    api_key: str = None,
    base_url: str = None,
    model: str = None,
    system_prompt: str = ""
) -> NemotronAgent:
    """Factory function to create a NemotronAgent."""
    return NemotronAgent(
        api_key=api_key or config.nvidia_api_key,
        base_url=base_url or config.models.nemo_base_url,
        model=model or config.models.nemo_model,
        system_prompt=system_prompt
    )