"""
router.py — Multi-provider model router
Supports: OpenRouter (free models), Google Gemini, Mistral, Groq
Mirrors the Go main.go logic but extended with fallback, load-balancing, and health tracking.
"""

import os
import json
import time
import random
import requests
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────

@dataclass
class Model:
    id: str
    name: str
    provider: str          # "openrouter" | "google" | "mistral" | "groq"
    context_window: int = 32768
    free: bool = True
    failures: int = 0      # Consecutive failures
    last_used: float = 0.0
    last_failure: float = 0.0   # Timestamp of last failure for cooldown
    cooldown_secs: int = 60     # Seconds to cool down after 3 failures

    def is_healthy(self) -> bool:
        """Healthy if fewer than 3 failures, OR cooldown has expired (auto-reset)."""
        if self.failures < 3:
            return True
        if time.time() - self.last_failure > self.cooldown_secs:
            self.failures = 0
            return True
        return False

    def health_bar(self) -> str:
        if not hasattr(self, 'last_failure'):
            return "●●●"
        if self.is_healthy():
            remaining = max(0, 3 - self.failures)
            return "●" * remaining + "○" * (3 - remaining)
        secs_left = int(self.cooldown_secs - (time.time() - self.last_failure))
        return f"❄ {secs_left}s cooldown"

# All free OpenRouter models from your list
OPENROUTER_FREE_MODELS: dict[str, Model] = {
    "gemma-3n-4b": Model(
        id="google/gemma-3n-e4b-it:free",
        name="Gemma 3n 4B",
        provider="openrouter",
        context_window=8192,
    ),
    "qwen3-4b": Model(
        id="qwen/qwen3-4b:free",
        name="Qwen3 4B",
        provider="openrouter",
        context_window=40960,
    ),
    "mistral-small-3.1": Model(
        id="mistralai/mistral-small-3.1-24b-instruct:free",
        name="Mistral Small 3.1 24B",
        provider="openrouter",
        context_window=131072,
    ),
    "gemma-3-4b": Model(
        id="google/gemma-3-4b-it:free",
        name="Gemma 3 4B",
        provider="openrouter",
        context_window=32768,
    ),
    "gemma-3-12b": Model(
        id="google/gemma-3-12b-it:free",
        name="Gemma 3 12B",
        provider="openrouter",
        context_window=32768,
    ),
    "gemma-3-27b": Model(
        id="google/gemma-3-27b-it:free",
        name="Gemma 3 27B",
        provider="openrouter",
        context_window=131072,
    ),
    "llama-3.3-70b": Model(
        id="meta-llama/llama-3.3-70b-instruct:free",
        name="Llama 3.3 70B",
        provider="openrouter",
        context_window=131072,
    ),
    "llama-3.2-3b": Model(
        id="meta-llama/llama-3.2-3b-instruct:free",
        name="Llama 3.2 3B",
        provider="openrouter",
        context_window=131072,
    ),
    "hermes-3-405b": Model(
        id="nousresearch/hermes-3-llama-3.1-405b:free",
        name="Hermes 3 405B",
        provider="openrouter",
        context_window=131072,
    ),
}

# Google Gemini models (direct API — from your Go code)
GOOGLE_MODELS: dict[str, Model] = {
    "gemini-2.0-flash": Model(
        id="gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        provider="google",
        context_window=1048576,
    ),
    "gemini-1.5-flash": Model(
        id="gemini-1.5-flash",
        name="Gemini 1.5 Flash",
        provider="google",
        context_window=1048576,
    ),
    "gemini-1.5-pro": Model(
        id="gemini-1.5-pro",
        name="Gemini 1.5 Pro",
        provider="google",
        context_window=2097152,
    ),
}

# Mistral models (direct API — from your Go code)
MISTRAL_MODELS: dict[str, Model] = {
    "mistral-small": Model(
        id="mistral-small-latest",
        name="Mistral Small",
        provider="mistral",
        context_window=32768,
    ),
    "codestral": Model(
        id="codestral-latest",
        name="Codestral",
        provider="mistral",
        context_window=32768,
    ),
}

# Groq models (direct API — from your Go code)
GROQ_MODELS: dict[str, Model] = {
    "groq-llama-70b": Model(
        id="llama-3.3-70b-versatile",
        name="Llama 3.3 70B (Groq)",
        provider="groq",
        context_window=131072,
    ),
    "groq-mixtral": Model(
        id="mixtral-8x7b-32768",
        name="Mixtral 8x7B (Groq)",
        provider="groq",
        context_window=32768,
    ),
}

ALL_MODELS = {**OPENROUTER_FREE_MODELS, **GOOGLE_MODELS, **MISTRAL_MODELS, **GROQ_MODELS}

# ─────────────────────────────────────────────
# Message format
# ─────────────────────────────────────────────

@dataclass
class Message:
    role: str    # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

# ─────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────

class ModelRouter:
    """
    Routes requests to the right provider.
    Supports fallback: if a model fails, tries the next healthy one.
    """

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
    MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
    GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self):
        self.api_keys = {
            "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
            "google": os.getenv("GOOGLE_API_KEY", ""),
            "mistral": os.getenv("MISTRAL_API_KEY", ""),
            "groq": os.getenv("GROQ_API_KEY", ""),
        }
        self.timeout = 60
        self._validate_keys()

    def _validate_keys(self):
        PLACEHOLDERS = {"your_openrouter_key_here", "your_key_here", "xxx", ""}
        valid = []
        for p, k in self.api_keys.items():
            if k and k.strip().lower() not in PLACEHOLDERS and not k.startswith("your_"):
                valid.append(p)
            elif k:
                print(f"⚠  {p.upper()}_API_KEY looks like a placeholder — replace it in .env")
        if not valid:
            print("⚠  No valid API keys found. Get a free key at openrouter.ai/keys")
        else:
            print(f"✓  Active providers: {', '.join(valid)}")
        # Update api_keys to only include real keys
        self.api_keys = {
            p: k for p, k in self.api_keys.items()
            if k and k.strip().lower() not in PLACEHOLDERS and not k.startswith("your_")
        }

    def available_models(self) -> list[str]:
        """Return model aliases where we have an API key."""
        result = []
        for alias, model in ALL_MODELS.items():
            if self.api_keys.get(model.provider):
                result.append(alias)
        return result

    def get_model(self, alias: str) -> Optional[Model]:
        return ALL_MODELS.get(alias)

    def healthy_models(self, provider: Optional[str] = None) -> list[str]:
        """Return models that are healthy (auto-recovers after cooldown)."""
        result = []
        for alias, model in ALL_MODELS.items():
            if provider and model.provider != provider:
                continue
            if self.api_keys.get(model.provider) and model.is_healthy():
                result.append(alias)
        return result

    # ── Core send ──────────────────────────────────────────────────────

    def send(
        self,
        model_alias: str,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        model = ALL_MODELS.get(model_alias)
        if not model:
            raise ValueError(f"Unknown model alias: {model_alias}")

        key = self.api_keys.get(model.provider, "")
        if not key:
            raise ValueError(f"No API key for provider: {model.provider}")

        try:
            if model.provider == "google":
                result = self._send_google(model, messages, temperature, max_tokens, key)
            else:
                result = self._send_openai_compat(model, messages, temperature, max_tokens, key)
            model.failures = 0
            model.last_used = time.time()
            return result
        except Exception as e:
            model.failures += 1
            model.last_failure = time.time()
            raise

    # Preferred provider order — reliable providers first
    FALLBACK_PRIORITY = ["groq", "mistral", "openrouter", "google"]

    def _sorted_fallbacks(self, exclude: str) -> list[str]:
        """Return all available models sorted by provider reliability."""
        by_provider: dict[str, list[str]] = {}
        for alias, model in ALL_MODELS.items():
            if alias == exclude:
                continue
            if not self.api_keys.get(model.provider):
                continue
            by_provider.setdefault(model.provider, []).append(alias)

        ordered = []
        for provider in self.FALLBACK_PRIORITY:
            ordered += by_provider.get(provider, [])
        # Any provider not in priority list goes last
        for provider, aliases in by_provider.items():
            if provider not in self.FALLBACK_PRIORITY:
                ordered += aliases

        # Healthy models first within each group
        healthy_set = set(self.healthy_models())
        return sorted(ordered, key=lambda a: (0 if a in healthy_set else 1))

    def send_with_fallback(
        self,
        preferred_alias: str,
        messages: list[Message],
        fallback_pool: Optional[list[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> tuple[str, str]:
        """Try preferred model, then fall back in provider-priority order."""
        candidates = [preferred_alias]
        if fallback_pool:
            candidates += [m for m in fallback_pool if m != preferred_alias]
        else:
            candidates += self._sorted_fallbacks(preferred_alias)

        last_error = None
        rate_limit_count = 0
        for alias in candidates:
            model = ALL_MODELS.get(alias)
            try:
                response = self.send(alias, messages, temperature, max_tokens)
                return response, alias
            except Exception as e:
                err_str = str(e)
                last_error = e
                is_rate_limit = "429" in err_str or "rate" in err_str.lower()
                if is_rate_limit:
                    rate_limit_count += 1
                    print(f"  ↳ {alias} rate-limited — trying next...")
                else:
                    print(f"  ↳ {alias} failed: {err_str[:120]} — trying next...")
                continue

        if rate_limit_count == len(candidates):
            raise RuntimeError(
                f"All {len(candidates)} models are rate-limited. "
                "Wait ~60s or add provider keys for higher limits."
            )
        raise RuntimeError(f"All models failed. Last error: {last_error}")

    # ── Provider implementations ───────────────────────────────────────

    def _send_openai_compat(
        self, model: Model, messages: list[Message],
        temperature: float, max_tokens: int, api_key: str
    ) -> str:
        """OpenAI-compatible endpoint (OpenRouter, Mistral, Groq)."""
        url_map = {
            "openrouter": self.OPENROUTER_URL,
            "mistral": self.MISTRAL_URL,
            "groq": self.GROQ_URL,
        }
        url = url_map[model.provider]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        if model.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/nexus-agent"
            headers["X-Title"] = "Nexus Agent"

        payload = {
            "model": model.id,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"API error: {data['error'].get('message', data['error'])}")

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("Empty response (no choices)")

        return choices[0]["message"]["content"]

    def _send_google(
        self, model: Model, messages: list[Message],
        temperature: float, max_tokens: int, api_key: str
    ) -> str:
        """Google Gemini API — mirrors your Go sendGoogleWithHistory."""
        url = self.GOOGLE_URL.format(model_id=model.id) + f"?key={api_key}"

        contents = []
        system_prompt = ""

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
                continue
            role = "model" if msg.role == "assistant" else "user"
            content = msg.content
            # Prepend system prompt to first user message
            if role == "user" and system_prompt:
                content = system_prompt + "\n\n" + content
                system_prompt = ""
            contents.append({"role": role, "parts": [{"text": content}]})

        if system_prompt:  # Wasn't consumed
            contents.insert(0, {"role": "user", "parts": [{"text": system_prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        resp = requests.post(url, json=payload, timeout=self.timeout)

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"API error: {data['error'].get('message', data['error'])}")

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("No candidates in response")

        return candidates[0]["content"]["parts"][0]["text"]

    # ── Utility ────────────────────────────────────────────────────────

    def list_models(self, show_all: bool = False):
        """Pretty-print available models."""
        print(f"\n{'─'*60}")
        print(f"  {'ALIAS':<22} {'NAME':<30} {'PROVIDER'}")
        print(f"{'─'*60}")
        for alias, model in ALL_MODELS.items():
            has_key = bool(self.api_keys.get(model.provider))
            if not show_all and not has_key:
                continue
            status = "✓" if has_key else "✗"
            health = model.health_bar() if has_key else "---"
            ctx = f"{model.context_window // 1024}K"
            print(f"  {status} {alias:<20} {model.name:<30} {model.provider} [{ctx}] {health}")
        print(f"{'─'*60}\n")