import os
import json
from dotenv import load_dotenv
load_dotenv()

from google import genai
from openai import OpenAI

from agents.ggl import goog  # FIX: was `GoogleAgent as goog`. goog() is the
                              # function-style, step-streaming entrypoint that
                              # this file actually calls (memory=/input=/
                              # thinking_level=/stream=), not the GoogleAgent class.
from agents.nemo import nvidia_nemo

from agents.tools import FUNCTION_MAP
from ascii import asciii

CYAN = "\033[96m"
from manager import MemoryManager

asciii()
ORANGE = "\033[38;2;255;165;0m"
GREEN = "\033[92m"
RESET = "\033[0m"

AKL = [
    "MISTRAL_API_KEY_{i}",
    "GOOGLE_API_KEY_{i}",
    "GROQ_API_KEY",
    "NVIDIA_API_KEY",
]

goog_client = genai.Client(api_key=os.getenv(AKL[1].format(i=3)))
nvidia_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv(AKL[3]))

memdb = MemoryManager()
MEM = []


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


while True:
    try:
        inp = input(f"{ORANGE}agent0 $> {RESET}").strip()

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

        stream = goog(
            goog_client,
            memory=MEM,
            input=enhanced_input,
            thinking_level="low",
            stream=True,
            temperature=0.7,
        )

        reply_text = ""
        pending_tool_calls = []
        current_call = None

        for event in stream:
            if event.event_type == "step.delta" and event.delta.type == "text":
                print(event.delta.text, end="")
                reply_text += event.delta.text

            elif event.event_type == "step.delta" and event.delta.type == "arguments_delta":
                if current_call is not None:
                    current_call["raw_args"] += event.delta.arguments

            elif event.event_type == "step.start":
                step = event.step
                if hasattr(step, "type") and step.type == "function_call":
                    current_call = {
                        "name": step.name,
                        "raw_args": "",
                        "call_id": step.id,
                    }

            elif event.event_type == "step.stop":
                if current_call is not None:
                    args = json.loads(current_call["raw_args"]) if current_call["raw_args"] else {}
                    pending_tool_calls.append({
                        "name": current_call["name"],
                        "args": args,
                        "call_id": current_call["call_id"],
                    })
                    current_call = None

        print()

        for call in pending_tool_calls:
            fn = FUNCTION_MAP.get(call["name"])
            print(f"{CYAN}[tool: {call['name']}]{RESET}")
            result = fn(**call["args"]) if fn else f"unknown tool: {call['name']}"
            print(f"{CYAN}[result: {str(result)[:100]}]{RESET}")

            MEM.append({
                "type": "tool_result",
                "content": [{"type": "function_response", "name": call["name"], "response": {"result": str(result)}}]
            })

        MEM.append({
            "type": "user_input",
            "content": [{"type": "text", "text": inp}]
        })

        MEM.append({
            "type": "model_output",
            "content": [{"type": "text", "text": reply_text}]
        })

        if len(MEM) > 100:
            MEM = MEM[-100:]

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