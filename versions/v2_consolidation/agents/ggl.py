# ggl.py - Google Gemini agent wrapper (v2: cleaned up, single interface)

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator, Union
import json
import itertools

from google import genai
from google.genai import types

from tools import GEMINI_TOOLS, FUNCTION_MAP, ToolResult
from config import config

__all__ = ["goog", "GoogleAgent", "create_google_agent"]


def _build_gemini_tools(flat_tools: List[Dict[str, Any]]) -> List[types.Tool]:
    """Convert flat tool dicts to google-genai types.Tool objects."""
    declarations = [
        types.FunctionDeclaration(
            name=t["name"],
            description=t.get("description", ""),
            parameters=t.get("parameters", {"type": "object", "properties": {}}),
        )
        for t in flat_tools
    ]
    return [types.Tool(function_declarations=declarations)]


_GEMINI_TOOLS_SDK = _build_gemini_tools(GEMINI_TOOLS)


# --------------------------------------------------------------------------
# Minimal event objects mimicking a "step" streaming API
# --------------------------------------------------------------------------

@dataclass
class _Delta:
    type: str                      # "text" | "arguments_delta"
    text: str = ""
    arguments: str = ""


@dataclass
class _Step:
    type: str                      # "message" | "function_call"
    id: Optional[str] = None
    name: Optional[str] = None
    arguments: str = ""


@dataclass
class _Event:
    event_type: str                 # "step.start" | "step.delta" | "step.stop"
    step: Optional[_Step] = None
    delta: Optional[_Delta] = None


def _thinking_config(thinking_level: str) -> Optional[types.ThinkingConfig]:
    """Map a simple low/medium/high string to a Gemini thinking budget."""
    if not thinking_level:
        return None
    budgets = {"low": 512, "medium": 4096, "high": 16384, "off": 0}
    budget = budgets.get(thinking_level.lower())
    if budget is None:
        return None
    return types.ThinkingConfig(thinking_budget=budget)


def _build_contents(memory: List[Dict[str, Any]], input_text: str) -> List[types.Content]:
    """Convert memory history + new input into Gemini Content objects."""
    contents: List[types.Content] = []

    for msg in memory:
        mtype = msg.get("type")
        parts_in = msg.get("content", [])

        if mtype == "user_input":
            text = "".join(p.get("text", "") for p in parts_in if p.get("type") == "text")
            if text:
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))

        elif mtype == "model_output":
            text = "".join(p.get("text", "") for p in parts_in if p.get("type") == "text")
            if text:
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))

        elif mtype == "tool_result":
            for p in parts_in:
                if p.get("type") == "function_response":
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_function_response(
                            name=p.get("name", ""),
                            response=p.get("response", {}),
                        )],
                    ))

    if input_text:
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=input_text)]))

    return contents


def _stream_events(
    client: genai.Client,
    memory: List[Dict[str, Any]],
    input: str = "",
    thinking_level: str = "low",
    temperature: float = 0.7,
    model: Optional[str] = None,
    system_prompt: str = "",
) -> Iterator[_Event]:
    """Do the actual Gemini streaming call and yield step-shaped events."""
    contents = _build_contents(memory, input)

    gen_config_kwargs: Dict[str, Any] = dict(
        temperature=temperature,
        max_output_tokens=config.models.gemini_max_output_tokens,
        tools=_GEMINI_TOOLS_SDK,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.AUTO
            )
        ),
    )
    if system_prompt:
        gen_config_kwargs["system_instruction"] = system_prompt
    thinking = _thinking_config(thinking_level)
    if thinking is not None:
        gen_config_kwargs["thinking_config"] = thinking

    gen_config = types.GenerateContentConfig(**gen_config_kwargs)

    stream = client.models.generate_content_stream(
        model=model or config.models.gemini_model,
        contents=contents,
        config=gen_config,
    )

    call_counter = itertools.count()
    open_call: Optional[_Step] = None

    for chunk in stream:
        if not chunk.candidates:
            continue
        candidate = chunk.candidates[0]
        if not candidate.content or not candidate.content.parts:
            continue

        for part in candidate.content.parts:
            text = getattr(part, "text", None)
            if text:
                yield _Event(event_type="step.delta", delta=_Delta(type="text", text=text))

            fc = getattr(part, "function_call", None)
            if fc is not None:
                call_id = f"call_{next(call_counter)}"
                step = _Step(type="function_call", id=call_id, name=fc.name)
                yield _Event(event_type="step.start", step=step)

                args_json = json.dumps(dict(fc.args or {}))
                yield _Event(
                    event_type="step.delta",
                    delta=_Delta(type="arguments_delta", arguments=args_json),
                )
                yield _Event(event_type="step.stop", step=step)


def goog(
    client: genai.Client,
    memory: List[Dict[str, Any]] = None,
    input: str = "",
    thinking_level: str = "low",
    stream: bool = True,
    temperature: float = 0.7,
    model: Optional[str] = None,
    system_prompt: str = "",
) -> Union[Iterator[_Event], str]:
    """
    Function-style entrypoint matching how orchestrator_tools.py calls `goog(...)`.

    Only `stream=True` is supported (that's the only mode any caller uses);
    it returns a generator of step events.
    """
    memory = memory or []
    if not stream:
        events = list(_stream_events(client, memory, input, thinking_level, temperature, model, system_prompt))
        text = "".join(e.delta.text for e in events if e.delta and e.delta.type == "text")
        return text
    return _stream_events(client, memory, input, thinking_level, temperature, model, system_prompt)


# --------------------------------------------------------------------------
# Class-based wrapper for OpenAI-style message lists
# --------------------------------------------------------------------------

class GoogleAgent:
    """Wrapper for Google Gemini API with OpenAI-style message lists."""

    def __init__(self, api_key: str = None, model: str = None, system_prompt: str = ""):
        self.client = genai.Client(api_key=api_key or config.google_api_key)
        self.model = model or config.models.gemini_model
        self.system_prompt = system_prompt
        self.tools = _GEMINI_TOOLS_SDK
        self.function_map = FUNCTION_MAP

    def _build_contents(self, messages: List[Dict[str, Any]]) -> List[types.Content]:
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                continue

            if role == "tool":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_function_response(
                        name=msg.get("name", ""),
                        response={"result": msg.get("content", "")}
                    )]
                ))
            elif role == "assistant" and msg.get("tool_calls"):
                parts = []
                if content:
                    parts.append(types.Part.from_text(text=content))
                for tc in msg["tool_calls"]:
                    parts.append(types.Part.from_function_call(
                        name=tc["function"]["name"],
                        args=json.loads(tc["function"]["arguments"] or "{}")
                    ))
                contents.append(types.Content(role="model", parts=parts))
            else:
                gemini_role = "user" if role == "user" else "model"
                contents.append(types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content)]
                ))
        return contents

    def _build_generate_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=self.system_prompt if self.system_prompt else None,
            tools=self.tools,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.AUTO
                )
            ),
            temperature=config.agent.worker_temperature,
            max_output_tokens=config.models.gemini_max_output_tokens,
        )

    def execute_tools(self, function_calls: List[Any]) -> List[Dict[str, Any]]:
        results = []
        for fc in function_calls:
            fn_name = fc.name
            fn_args = fc.args or {}
            fn = self.function_map.get(fn_name)
            if not fn:
                result = ToolResult(False, "", f"Unknown tool: {fn_name}")
            else:
                try:
                    result = fn(**fn_args)
                except Exception as e:
                    result = ToolResult(False, "", str(e))
            results.append({"name": fn_name, "result": str(result)})
        return results

    def chat(self, messages: List[Dict[str, Any]], stream: bool = False) -> Any:
        contents = self._build_contents(messages)
        gen_config = self._build_generate_config()

        if stream:
            return self.client.models.generate_content_stream(
                model=self.model, contents=contents, config=gen_config
            )
        return self.client.models.generate_content(
            model=self.model, contents=contents, config=gen_config
        )

    def stream_tool_calls(self, messages: List[Dict[str, Any]]):
        tool_calls = []
        full_text = ""
        for chunk in self.chat(messages, stream=True):
            if not chunk.candidates:
                continue
            candidate = chunk.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text:
                        full_text += part.text
                    if part.function_call:
                        tool_calls.append({
                            "name": part.function_call.name,
                            "arguments": json.dumps(dict(part.function_call.args))
                        })
        return full_text, tool_calls


def create_google_agent(system_prompt: str = "", api_key: str = None, model: str = None) -> GoogleAgent:
    return GoogleAgent(
        api_key=api_key or config.google_api_key,
        model=model or config.models.gemini_model,
        system_prompt=system_prompt,
    )