# config.py - Centralized configuration

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Model configurations."""
    # NVIDIA Nemotron
    nemo_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    nemo_base_url: str = "https://integrate.api.nvidia.com/v1"
    nemo_temperature: float = 0.6
    nemo_top_p: float = 0.95
    nemo_max_tokens: int = 16384
    nemo_thinking: bool = True
    nemo_reasoning_budget: int = 16384

    # Google Gemini
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.7
    gemini_max_output_tokens: int = 8192

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    duplicate_threshold: float = 0.97
    related_threshold: float = 0.80
    search_k: int = 5
    search_score_threshold: float = 0.65
    max_memories_in_context: int = 10
    memory_dir: str = "agent_mem"


@dataclass
class AgentConfig:
    """Agent behavior configuration."""
    max_turns: int = 6
    short_input_threshold: int = 20
    memory_trigger_threshold: int = 20
    max_conversation_history: int = 100

    # --- Fields nemo.py / ggl.py actually reference ---
    # (previously missing -> AttributeError as soon as those modules ran)
    nvidia_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    google_model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 8192


@dataclass
class Config:
    """Main configuration container."""
    models: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # API Keys (loaded from env)
    nvidia_api_key: Optional[str] = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY"))
    google_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY_3"))
    google_api_key_2: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY_4"))

    def validate(self) -> list[str]:
        """Validate required configuration. Returns list of errors."""
        errors = []
        if not self.nvidia_api_key:
            errors.append("NVIDIA_API_KEY not set")
        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY_3 not set")
        return errors


# Global config instance
config = Config()