from .ggl import goog  # FIX: was `GoogleAgent as goog`. This file calls
                        # goog() as a function (memory=/input=/thinking_level=/
                        # stream=), so it needs the function entrypoint,
                        # not the GoogleAgent class.

MAGENTA = "\033[95m"
BLUE = "\033[94m"
RESET = "\033[0m"


def _collect_google_reply(client, prompt: str, label: str, color: str,
                           thinking_level: str = "low", temperature: float = 0.7) -> str:
    """Run one turn against a Gemini worker, streaming it live, and return the full text."""
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


def make_google_agent_tools(client_a, client_b=None):
    client_b = client_b or client_a

    def ask_google_agent_1(message: str) -> str:
        return _collect_google_reply(client_a, message, label="gemini-1", color=MAGENTA)

    def ask_google_agent_2(message: str) -> str:
        return _collect_google_reply(client_b, message, label="gemini-2", color=BLUE)

    function_map = {
        "ask_google_agent_1": ask_google_agent_1,
        "ask_google_agent_2": ask_google_agent_2,
    }

    tool_description = (
        "Delegate a subtask, question, or piece of work to a Google Gemini "
        "worker agent. Use this to parallelize independent subtasks, get a "
        "second pass on something, or offload part of the task. Send it a "
        "clear, self-contained instruction -- it has no memory of this "
        "conversation beyond what you put in `message`."
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
    ]

    return tools, function_map