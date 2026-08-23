import os
import json

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from google import genai

from agents.nemo import nvidia_nemo
from agents.orchestrator_tools import make_google_agent_tools
from agents.tools import TOOLS as FILE_TOOLS, FUNCTION_MAP as FILE_FUNCTION_MAP, to_openai_tools
from ascii import asciii
from manager import MemoryManager

CYAN = "\033[96m"
ORANGE = "\033[38;2;255;165;0m"
GREEN = "\033[92m"
DIM = "\033[2m"
RESET = "\033[0m"

SYSTEM_PROMPT = (
    "You are Nemo, an orchestrator agent. You have two worker tools, "
    "ask_google_agent_1 and ask_google_agent_2, backed by Google Gemini, "
    "plus read_file/write_file/run_command tools that act directly on the "
    "local machine. Break the user's task down, delegate self-contained "
    "subtasks to the Gemini workers when that genuinely helps (you can "
    "call both in the same turn to run them in parallel), and use the "
    "file/shell tools yourself to write, save, or run code and inspect "
    "results. Only call a tool when it actually helps -- otherwise just "
    "answer directly.\n\n"
    "Every turn, before doing anything else, output one short line stating "
    "your delegation decision -- e.g. 'delegation: not needed, this is a "
    "local file/compile task' or 'delegation: asking both Gemini workers "
    "for independent approaches to compare'. Then proceed. This line is "
    "for the user's visibility into your reasoning, not a tool call.\n\n"
    "When you're done, reply with plain text and no further tool calls."
)

asciii()


def build_clients():
    nvidia_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
    )
    # Reuse one Google key for both workers by default. Pass a second
    # genai.Client (e.g. keyed off GOOGLE_API_KEY_4) as client_b below if
    # you want separate quota per worker.
    goog_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_3"))
    return nvidia_client, goog_client


nvidia_client, goog_client = build_clients()

google_tools, google_function_map = make_google_agent_tools(goog_client)
ALL_TOOLS = google_tools + to_openai_tools(FILE_TOOLS)
ALL_FUNCTIONS = {**google_function_map, **FILE_FUNCTION_MAP}

memdb = MemoryManager()
messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def should_consider_memory(text: str) -> bool:
    text = text.strip()

    if len(text) < 20:
        return False

    junk = {
        "ok", "okay", "thanks", "thank you", "cool", "nice", "yep", "yes", "no", "hi", "hello"
    }

    if text.lower() in junk:
        return False

    return True


def retrieve_memories(query: str, k: int = 5) -> str:
    if len(memdb.store.documents) == 0:
        return ""

    try:
        results = memdb.store.search(query, k=k)
        memories = []

        for score, memory in results:
            if score > 0.65:
                memories.append(f"[similarity={score:.2f}] {memory}")

        return "\n".join(memories)

    except Exception as e:
        print(f"\nMemory Search Error: {e}")
        return ""


def stream_nemo_turn():
    """
    Stream one Nemo call against the running `messages` history.
    Prints text/reasoning tokens live as they arrive. Returns
    (reply_text, tool_calls) where tool_calls is a list of
    {"id", "name", "arguments"} dicts (arguments = raw JSON string).
    """
    stream = nvidia_nemo(nvidia_client, memory=messages, input="", tools=ALL_TOOLS, stream=True)

    reply_text = ""
    tool_calls_acc = {}

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            print(f"{DIM}{reasoning}{RESET}", end="", flush=True)

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


def run_turn(task: str, max_rounds: int = 6) -> str:
    messages.append({"role": "user", "content": task})

    for round_no in range(1, max_rounds + 1):
        if round_no > 1:
            print(f"{ORANGE}[nemo/round {round_no}]{RESET}")

        reply_text, tool_calls = stream_nemo_turn()

        if not tool_calls:
            return reply_text

        assistant_msg = {"role": "assistant", "content": reply_text}
        assistant_msg["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in tool_calls
        ]
        messages.append(assistant_msg)

        for tc in tool_calls:
            fn_name = tc["name"]
            try:
                fn_args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError:
                fn_args = {}

            fn = ALL_FUNCTIONS.get(fn_name)
            print(f"{CYAN}[tool call: {fn_name}({fn_args})]{RESET}")

            try:
                result = fn(**fn_args) if fn else f"unknown tool: {fn_name}"
            except Exception as e:
                result = f"tool error: {e}"

            print(f"{CYAN}[tool result: {str(result)[:200]}]{RESET}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": fn_name,
                "content": str(result),
            })

    return "[orchestrator] did not resolve in max rounds"


if __name__ == "__main__":
    while True:
        try:
            inp = input(f"{ORANGE}orch $> {RESET}").strip()

            if not inp:
                continue

            if inp.lower() == "/memory":
                print("\n=== LONG TERM MEMORY ===")

                if not memdb.store.documents:
                    print("No memories stored.")
                else:
                    for i, memory in enumerate(memdb.store.documents, start=1):
                        print(f"\n[{i}]")
                        print(memory)

                print(f"\nTotal Memories: {len(memdb.store.documents)}\n")
                continue

            if inp.lower() == "/memory count":
                print(f"Total Memories: {len(memdb.store.documents)}")
                continue

            if inp.lower().startswith("/memory search "):
                query = inp[len("/memory search "):]
                results = memdb.store.search(query, k=10)

                print(f"\n=== SEARCH RESULTS FOR '{query}' ===")
                for score, memory in results:
                    print(f"\nScore: {score:.4f}")
                    print(memory)
                print()
                continue

            if inp.lower() == "/memory clear":
                memdb.store.documents.clear()

                import faiss
                dim = memdb.store.model.get_sentence_embedding_dimension()
                memdb.store.index = faiss.IndexFlatIP(dim)

                memdb.save()
                print("Memory cleared.")
                continue

            if inp.lower() in {"exit", "quit"}:
                memdb.save()
                break

            memory_context = retrieve_memories(inp)
            enhanced_input = inp

            if memory_context:
                enhanced_input = (
                    f"Relevant memories from previous conversations:\n\n"
                    f"{memory_context}\n\n"
                    f"Current user message:\n\n{inp}"
                )

            run_turn(enhanced_input)
            print()

            if should_consider_memory(inp):
                action = memdb.process(inp)
                print(f"{GREEN}[memory: {action}]{RESET}")
                memdb.save()

        except KeyboardInterrupt:
            print("\nSaving memory...")
            memdb.save()
            break

        except Exception as e:
            print(f"\nError: {e}")