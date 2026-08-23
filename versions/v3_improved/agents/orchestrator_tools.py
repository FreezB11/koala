# orchestrator_tools.py - Factory for all worker tools (v3: multi-provider with retry)

from .ggl import goog  # function entrypoint
from .nemo import create_mistral_worker, create_groq_worker, create_together_worker

MAGENTA = "\033[95m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
ORANGE = "\033[38;2;255;165;0m"
RESET = "\033[0m"


def _collect_google_reply(client, prompt: str, label: str, color: str,
                           thinking_level: str = None, temperature: float = None) -> str:
    """Run one turn against a Gemini worker, streaming it live, and return the full text."""
    from config import config
    
    if thinking_level is None:
        thinking_level = config.agent.worker_thinking_level
    if temperature is None:
        temperature = config.agent.worker_temperature

    stream = goog(
        client,
        memory=[],
        input=prompt,
        thinking_level=thinking_level,
        stream=True,
        temperature=temperature,
    )

    print(f"{color}[{label}] {RESET}", end="", flush=True)

    text = ""
    for event in stream:
        if event.event_type == "step.delta" and event.delta.type == "text":
            print(f"{color}{event.delta.text}{RESET}", end="", flush=True)
            text += event.delta.text

    print()
    return text.strip()


def _collect_openai_compatible_reply(worker, prompt: str) -> str:
    """Run one turn against an OpenAI-compatible worker."""
    return worker.run(prompt, stream=True)


def make_all_worker_tools(
    google_client_1, 
    google_client_2=None,
    mistral_api_key=None,
    groq_api_key=None,
    together_api_key=None
):
    """Create tools for all available workers."""
    google_client_2 = google_client_2 or google_client_1
    
    # Create workers
    mistral_worker = create_mistral_worker(mistral_api_key) if mistral_api_key else None
    groq_worker = create_groq_worker(groq_api_key) if groq_api_key else None
    together_worker = create_together_worker(together_api_key) if together_api_key else None

    def ask_google_agent_1(message: str) -> str:
        return _collect_google_reply(google_client_1, message, label="gemini-1", color=MAGENTA)

    def ask_google_agent_2(message: str) -> str:
        return _collect_google_reply(google_client_2, message, label="gemini-2", color=BLUE)

    def ask_mistral_agent(message: str) -> str:
        if mistral_worker:
            return _collect_openai_compatible_reply(mistral_worker, message)
        return "[Mistral worker not available]"

    def ask_groq_agent(message: str) -> str:
        if groq_worker:
            return _collect_openai_compatible_reply(groq_worker, message)
        return "[Groq worker not available]"

    def ask_together_agent(message: str) -> str:
        if together_worker:
            return _collect_openai_compatible_reply(together_worker, message)
        return "[Together AI worker not available]"

    function_map = {
        "ask_google_agent_1": ask_google_agent_1,
        "ask_google_agent_2": ask_google_agent_2,
        "ask_mistral_agent": ask_mistral_agent,
        "ask_groq_agent": ask_groq_agent,
        "ask_together_agent": ask_together_agent,
    }

    tool_description = (
        "Delegate a subtask, question, or piece of work to a worker agent. "
        "Use this to parallelize independent subtasks, get a second pass on something, "
        "or offload part of the task. Send it a clear, self-contained instruction -- "
        "it has no memory of this conversation beyond what you put in `message`."
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "ask_google_agent_1",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Self-contained task or question for this worker.",
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
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Self-contained task or question for this worker.",
                        }
                    },
                    "required": ["message"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_mistral_agent",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Self-contained task or question for this worker.",
                        }
                    },
                    "required": ["message"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_groq_agent",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Self-contained task or question for this worker.",
                        }
                    },
                    "required": ["message"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_together_agent",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Self-contained task or question for this worker.",
                        }
                    },
                    "required": ["message"],
                },
            },
        },
    ]

    return tools, function_map


# Backward compatibility
def make_google_agent_tools(client_a, client_b=None):
    """Legacy function for backward compatibility."""
    tools, func_map = make_all_worker_tools(client_a, client_b)
    # Filter to only Google tools
    google_tools = [t for t in tools if t["function"]["name"].startswith("ask_google")]
    google_func_map = {k: v for k, v in func_map.items() if k.startswith("ask_google")}
    return google_tools, google_func_map