# health.py - Health checks for API services (v3: all providers)

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

import requests

from config import config
from retry import RetryPolicy

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    service: str
    status: HealthStatus
    latency_ms: float
    details: Dict[str, Any]
    error: Optional[str] = None


class HealthChecker:
    """Health checks for external API dependencies."""
    
    def __init__(self, timeout: int = None):
        self.timeout = timeout or config.agent.health_check_timeout
        self.retry_policy = RetryPolicy(
            max_retries=1,  # Quick fail for health checks
            base_delay=0.5,
            max_delay=2.0,
        )
    
    def check_nvidia(self) -> HealthCheckResult:
        """Check NVIDIA Nemotron API availability."""
        start = time.time()
        try:
            headers = {"Authorization": f"Bearer {config.nvidia_api_key}"}
            response = requests.get(
                f"{config.models.nemo_base_url}/models",
                headers=headers,
                timeout=self.timeout
            )
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                models = response.json().get("data", [])
                nemotron_models = [m for m in models if "nemotron" in m.get("id", "").lower()]
                return HealthCheckResult(
                    service="nvidia",
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    details={
                        "models_available": len(models),
                        "nemotron_models": len(nemotron_models),
                        "target_model": config.models.nemo_model in [m.get("id") for m in models]
                    }
                )
            elif response.status_code == 401:
                return HealthCheckResult(
                    service="nvidia",
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=latency,
                    details={},
                    error="Invalid API key"
                )
            else:
                return HealthCheckResult(
                    service="nvidia",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    details={"status_code": response.status_code},
                    error=f"HTTP {response.status_code}"
                )
        except requests.Timeout:
            return HealthCheckResult(
                service="nvidia",
                status=HealthStatus.UNHEALTHY,
                latency_ms=self.timeout * 1000,
                details={},
                error="Timeout"
            )
        except Exception as e:
            return HealthCheckResult(
                service="nvidia",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                details={},
                error=str(e)
            )
    
    def check_google(self) -> HealthCheckResult:
        """Check Google Gemini API availability."""
        start = time.time()
        try:
            from google import genai
            client = genai.Client(api_key=config.google_api_key)
            
            response = client.models.generate_content(
                model=config.models.gemini_model,
                contents="health check",
                config={"max_output_tokens": 10}
            )
            latency = (time.time() - start) * 1000
            
            if response.text:
                return HealthCheckResult(
                    service="google",
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    details={"model": config.models.gemini_model}
                )
            else:
                return HealthCheckResult(
                    service="google",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    details={},
                    error="Empty response"
                )
        except Exception as e:
            return HealthCheckResult(
                service="google",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                details={},
                error=str(e)
            )
    
    def check_mistral(self) -> HealthCheckResult:
        """Check Mistral API availability."""
        if not config.mistral_api_key_1:
            return HealthCheckResult(
                service="mistral",
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                details={},
                error="API key not configured"
            )
        
        start = time.time()
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=config.models.mistral_base_url,
                api_key=config.mistral_api_key_1,
            )
            
            response = client.chat.completions.create(
                model=config.models.mistral_model,
                messages=[{"role": "user", "content": "health check"}],
                max_tokens=10,
                timeout=self.timeout
            )
            latency = (time.time() - start) * 1000
            
            if response.choices and response.choices[0].message.content:
                return HealthCheckResult(
                    service="mistral",
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    details={"model": config.models.mistral_model}
                )
            else:
                return HealthCheckResult(
                    service="mistral",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    details={},
                    error="Empty response"
                )
        except Exception as e:
            return HealthCheckResult(
                service="mistral",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                details={},
                error=str(e)
            )
    
    def check_groq(self) -> HealthCheckResult:
        """Check Groq API availability."""
        if not config.groq_api_key:
            return HealthCheckResult(
                service="groq",
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                details={},
                error="API key not configured"
            )
        
        start = time.time()
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=config.models.groq_base_url,
                api_key=config.groq_api_key,
            )
            
            response = client.chat.completions.create(
                model=config.models.groq_model,
                messages=[{"role": "user", "content": "health check"}],
                max_tokens=10,
                timeout=self.timeout
            )
            latency = (time.time() - start) * 1000
            
            if response.choices and response.choices[0].message.content:
                return HealthCheckResult(
                    service="groq",
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    details={"model": config.models.groq_model}
                )
            else:
                return HealthCheckResult(
                    service="groq",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    details={},
                    error="Empty response"
                )
        except Exception as e:
            return HealthCheckResult(
                service="groq",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                details={},
                error=str(e)
            )
    
    def check_together(self) -> HealthCheckResult:
        """Check Together AI API availability."""
        if not config.together_api_key:
            return HealthCheckResult(
                service="together",
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                details={},
                error="API key not configured"
            )
        
        start = time.time()
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=config.models.together_base_url,
                api_key=config.together_api_key,
            )
            
            response = client.chat.completions.create(
                model=config.models.together_model,
                messages=[{"role": "user", "content": "health check"}],
                max_tokens=10,
                timeout=self.timeout
            )
            latency = (time.time() - start) * 1000
            
            if response.choices and response.choices[0].message.content:
                return HealthCheckResult(
                    service="together",
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    details={"model": config.models.together_model}
                )
            else:
                return HealthCheckResult(
                    service="together",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    details={},
                    error="Empty response"
                )
        except Exception as e:
            return HealthCheckResult(
                service="together",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                details={},
                error=str(e)
            )
    
    def check_embedding_model(self) -> HealthCheckResult:
        """Check if embedding model loads correctly."""
        start = time.time()
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config.models.embedding_model)
            # Test encode
            _ = model.encode(["test"], normalize_embeddings=True)
            latency = (time.time() - start) * 1000
            
            return HealthCheckResult(
                service="embedding",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                details={
                    "model": config.models.embedding_model,
                    "dimension": model.get_sentence_embedding_dimension()
                }
            )
        except Exception as e:
            return HealthCheckResult(
                service="embedding",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                details={},
                error=str(e)
            )
    
    def check_memory_store(self) -> HealthCheckResult:
        """Check memory store accessibility."""
        start = time.time()
        try:
            from memory import LocalVectorStore
            store = LocalVectorStore()
            latency = (time.time() - start) * 1000
            
            return HealthCheckResult(
                service="memory",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                details={
                    "memory_count": len(store),
                    "memory_dir": config.memory.memory_dir
                }
            )
        except Exception as e:
            return HealthCheckResult(
                service="memory",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                details={},
                error=str(e)
            )
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}
        results["nvidia"] = self.check_nvidia()
        results["google"] = self.check_google()
        results["mistral"] = self.check_mistral()
        results["groq"] = self.check_groq()
        results["together"] = self.check_together()
        results["embedding"] = self.check_embedding_model()
        results["memory"] = self.check_memory_store()
        return results
    
    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Determine overall system health."""
        # Only consider configured providers
        configured = [r for r in results.values() if r.status != HealthStatus.UNKNOWN]
        if not configured:
            return HealthStatus.UNKNOWN
        
        statuses = [r.status for r in configured]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED


def print_health_report(results: Dict[str, HealthCheckResult]):
    """Print a formatted health report."""
    print("\n=== HEALTH CHECK REPORT ===")
    for service, result in results.items():
        status_color = {
            HealthStatus.HEALTHY: "\033[92m",    # Green
            HealthStatus.DEGRADED: "\033[93m",   # Yellow
            HealthStatus.UNHEALTHY: "\033[91m",  # Red
            HealthStatus.UNKNOWN: "\033[90m",    # Gray
        }.get(result.status, "\033[0m")
        reset = "\033[0m"
        
        print(f"  {service:12} {status_color}{result.status.value:10}{reset} "
              f"({result.latency_ms:.0f}ms)")
        if result.error:
            print(f"    Error: {result.error}")
        for k, v in result.details.items():
            print(f"    {k}: {v}")
    
    checker = HealthChecker()
    overall = checker.get_overall_status(results)
    overall_color = {
        HealthStatus.HEALTHY: "\033[92m",
        HealthStatus.DEGRADED: "\033[93m",
        HealthStatus.UNHEALTHY: "\033[91m",
        HealthStatus.UNKNOWN: "\033[90m",
    }.get(overall, "\033[0m")
    print(f"\n  Overall: {overall_color}{overall.value}{reset}\n")