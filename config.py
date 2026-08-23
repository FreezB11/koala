# config.py - Centralized configuration (v3: added retry/health settings)

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Model configurations - single source for all model settings."""
    # NVIDIA Nemotron
    nemo_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    nemo_base_url: str = "https://integrate.api.nvidia.com/v1"
    nemo_temperature: float = 0.6
    nemo_top_p: float = 0.95
    nemo_max_tokens: int = 16384

    # Google Gemini
    gemini_model: str = "gemini-2.5-flash"  # Correct model name from API
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
    # v3: summarization
    summarization_threshold: int = 50  # summarize after N memories
    max_memory_length: int = 5000  # chars before summarization


@dataclass
class AgentConfig:
    """Agent behavior configuration."""
    max_turns: int = 50
    short_input_threshold: int = 20
    memory_trigger_threshold: int = 20
    max_conversation_history: int = 100
    worker_thinking_level: str = "low"
    worker_temperature: float = 0.7
    # v3: retry settings
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0
    # v3: health check
    health_check_timeout: int = 10


@dataclass
class ToolConfig:
    """Tool execution configuration."""
    command_timeout: int = 60
    max_file_size_mb: int = 10
    allowed_commands: list = field(default_factory=lambda: [
        "ls", "cat", "head", "tail", "grep", "find", "python", "python3",
        "pip", "npm", "node", "git", "mkdir", "touch", "cp", "mv", "rm", "echo"
    ])


@dataclass
class LoggingConfig:
    """Logging and observability configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    json_format: bool = False
    log_file: Optional[str] = None


@dataclass
class Config:
    """Main configuration container."""
    models: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

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