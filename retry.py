# retry.py - Retry utilities with exponential backoff (v3)

import time
import logging
import random
from typing import Callable, TypeVar, Optional, Tuple, Type
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True
) -> float:
    """Calculate delay with exponential backoff and optional jitter."""
    delay = min(base_delay * (2 ** attempt), max_delay)
    if jitter:
        delay *= (0.5 + random.random())  # 0.5x to 1.5x
    return delay


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    should_retry: Optional[Callable[[Exception], bool]] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Exception types to catch and retry
        should_retry: Optional function to determine if an exception should trigger retry
        on_retry: Optional callback when retry occurs (exception, attempt)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # Check if we should retry this specific exception
                    if should_retry and not should_retry(e):
                        logger.debug(f"Not retrying {type(e).__name__}: {e}")
                        raise
                    
                    if attempt < max_retries:
                        delay = exponential_backoff(attempt, base_delay, max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        if on_retry:
                            on_retry(e, attempt)
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )
            
            raise last_exception
        return wrapper
    return decorator


class RetryPolicy:
    """Configurable retry policy for API clients."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        retryable_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504),
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retryable_status_codes = retryable_status_codes
    
    def is_retryable(self, exception: Exception) -> bool:
        """Check if an exception is retryable."""
        # Check for HTTP status codes in exception
        if hasattr(exception, 'status_code'):
            return exception.status_code in self.retryable_status_codes
        if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
            return exception.response.status_code in self.retryable_status_codes
        # Network errors are generally retryable
        if isinstance(exception, (ConnectionError, TimeoutError, IOError)):
            return True
        return False
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function with this retry policy."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if not self.is_retryable(e):
                    raise
                
                if attempt < self.max_retries:
                    delay = exponential_backoff(attempt, self.base_delay, self.max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All attempts failed: {e}")
        
        raise last_exception