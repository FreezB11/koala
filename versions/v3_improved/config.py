# config.py - Centralized configuration (v3: added all API providers, retry/health settings)

import os
from dataclasses import dataclass, field
from typing import Optional, List
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

    # Mistral
    mistral_model: str = "mistral-large-latest"
    mistral_base_url: str = "https://api.mistral.ai/v1"
    mistral_temperature: float = 0.7
    mistral_max_tokens: int = 8192

    # Groq
    groq_model: str = "llama-3.1-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_temperature: float = 0.7
    groq_max_tokens: int = 8192

    # Together AI
    together_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    together_base_url: str = "https://api.together.xyz/v1"
    together_temperature: float = 0.7
    together_max_tokens: int = 8192

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
    # Skill/Progress tracking
    skill_file: str = "skill.md"
    progress_file: str = "progress.md"


@dataclass
class AgentConfig:
    """Agent behavior configuration."""
    max_turns: int = 100  # Increased from 50
    short_input_threshold: int = 20
    memory_trigger_threshold: int = 20
    max_conversation_history: int = 200
    worker_thinking_level: str = "low"
    worker_temperature: float = 0.7
    # v3: retry settings
    max_retries: int = 5  # Increased from 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0  # Increased from 30
    # v3: health check
    health_check_timeout: int = 15
    # Parallel worker execution
    parallel_workers: bool = True
    # Fallback providers
    fallback_providers: List[str] = field(default_factory=lambda: ["google", "mistral", "groq", "together"])


@dataclass
class ToolConfig:
    """Tool execution configuration."""
    command_timeout: int = 120  # Increased from 60
    max_file_size_mb: int = 20  # Increased from 10
    allowed_commands: list = field(default_factory=lambda: [
        "ls", "cat", "head", "tail", "grep", "find", "python", "python3",
        "pip", "npm", "node", "git", "mkdir", "touch", "cp", "mv", "rm", "echo",
        "mkdir", "chmod", "chown", "tar", "gzip", "gunzip", "unzip", "curl", "wget"
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
    google_api_key_1: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY_1"))
    google_api_key_2_alt: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY_2"))
    mistral_api_key_1: Optional[str] = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY_1"))
    mistral_api_key_2: Optional[str] = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY_2"))
    mistral_api_key_3: Optional[str] = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY_3"))
    mistral_api_key_4: Optional[str] = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY_4"))
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    together_api_key: Optional[str] = field(default_factory=lambda: os.getenv("TOGETHER_API_KEY"))

    def validate(self) -> list[str]:
        """Validate required configuration. Returns list of errors."""
        errors = []
        if not self.nvidia_api_key:
            errors.append("NVIDIA_API_KEY not set")
        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY_3 not set")
        return errors

    def get_available_providers(self) -> List[str]:
        """Get list of available API providers based on configured keys."""
        providers = []
        if self.nvidia_api_key:
            providers.append("nvidia")
        if self.google_api_key or self.google_api_key_1 or self.google_api_key_2_alt:
            providers.append("google")
        if self.mistral_api_key_1 or self.mistral_api_key_2 or self.mistral_api_key_3 or self.mistral_api_key_4:
            providers.append("mistral")
        if self.groq_api_key:
            providers.append("groq")
        if self.together_api_key:
            providers.append("together")
        return providers


# Global config instance
config = Config()