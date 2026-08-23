"""
Minimal Mistral API client using raw `requests`.
Covers: chat completions, tool calling, and embeddings.

Usage:
    export MISTRAL_API_KEY="your-key-here"
    python mistral_client.py
"""
from dotenv import load_dotenv
load_dotenv()  # reads .env in the current directory and populates os.environ

import os
import json
import time
import requests


class MistralClient:
    BASE_URL = "https://api.mistral.ai/v1"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY_1")
        if not self.api_key:
            raise ValueError("Set MISTRAL_API_KEY env var or pass api_key explicitly.")

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    # ---------- internal helper ----------

    def _post(self, path: str, payload: dict, max_retries: int = 3, timeout: int = 60) -> dict:
        url = f"{self.BASE_URL}{path}"
        for attempt in range(max_retries):
            resp = self.session.post(url, json=payload, timeout=timeout)

            if resp.status_code == 429:
                wait = 2 ** attempt
                time.sleep(wait)
                continue

            if resp.status_code >= 500:
                wait = 2 ** attempt
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

            return resp.json()

        raise RuntimeError(f"Max retries exceeded calling {path}")

    # ---------- chat ----------

    def chat(
        self,
        messages: list[dict],
        model: str = "mistral-large-latest",
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> dict:
        """Single (non-streaming) chat completion. Returns the raw API response dict."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        return self._post("/chat/completions", payload)

    def chat_stream(
        self,
        messages: list[dict],
        model: str = "mistral-large-latest",
        **kwargs,
    ):
        """Generator yielding text chunks as they arrive."""
        payload = {"model": model, "messages": messages, "stream": True, **kwargs}
        url = f"{self.BASE_URL}/chat/completions"

        with self.session.post(url, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                chunk = raw_line[len("data: "):]
                if chunk == "[DONE]":
                    break
                event = json.loads(chunk)
                delta = event["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta

    # ---------- tool calling ----------

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_functions: dict,
        model: str = "mistral-large-latest",
        max_rounds: int = 5,
        **kwargs,
    ) -> dict:
        """
        Runs the full tool-calling loop:
        1. Send messages + tools to the model.
        2. If the model requests tool calls, execute them locally via `tool_functions`.
        3. Feed results back as `tool` role messages.
        4. Repeat until the model returns a plain text answer or max_rounds is hit.

        `tool_functions` maps tool name -> a Python callable(**args) -> str/dict.
        Returns the final API response dict.
        """
        messages = list(messages)  # don't mutate caller's list

        for _ in range(max_rounds):
            response = self.chat(messages, model=model, tools=tools, **kwargs)
            choice = response["choices"][0]
            msg = choice["message"]

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                return response  # model gave a final answer, done

            # Append the assistant's tool-call message to history
            messages.append(msg)

            for call in tool_calls:
                fn_name = call["function"]["name"]
                fn_args = json.loads(call["function"]["arguments"])

                if fn_name not in tool_functions:
                    result = {"error": f"Unknown tool '{fn_name}'"}
                else:
                    result = tool_functions[fn_name](**fn_args)

                if not isinstance(result, str):
                    result = json.dumps(result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": fn_name,
                    "content": result,
                })

        raise RuntimeError(f"Tool-calling loop did not resolve in {max_rounds} rounds")

    # ---------- embeddings ----------

    def embed(
        self,
        input_text: str | list[str],
        model: str = "mistral-embed",
        **kwargs,
    ) -> dict:
        """Returns the raw API response dict (response['data'][i]['embedding'] for vectors)."""
        payload = {"model": model, "input": input_text, **kwargs}
        return self._post("/embeddings", payload)

    def embed_vectors(self, input_text: str | list[str], model: str = "mistral-embed", **kwargs) -> list[list[float]]:
        """Convenience: returns just the list of embedding vectors, in input order."""
        result = self.embed(input_text, model=model, **kwargs)
        return [item["embedding"] for item in result["data"]]


# ---------------- example usage ----------------

if __name__ == "__main__":
    client = MistralClient()

    # --- 1. Plain chat ---
    resp = client.chat(
        messages=[{"role": "user", "content": "Say hello in 5 words."}],
        temperature=0.3,
    )
    print("Chat:", resp["choices"][0]["message"]["content"])

    # --- 2. Tool calling ---
    def get_weather(location: str) -> dict:
        # Replace with a real API call
        return {"location": location, "temp_c": 22, "condition": "sunny"}

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    final = client.chat_with_tools(
        messages=[{"role": "user", "content": "What's the weather in Bengaluru right now?"}],
        tools=tools,
        tool_functions={"get_weather": get_weather},
    )
    print("Tool-calling result:", final["choices"][0]["message"]["content"])

    # --- 3. Embeddings ---
    vectors = client.embed_vectors(["cat", "kitten", "airplane"])
    print("Embedding dims:", [len(v) for v in vectors])