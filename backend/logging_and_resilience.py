"""
Production Logging and Resilience Utilities.

Provides:
  - Structured logging with context
  - Retry logic with exponential backoff
  - Timeout handling
  - Circuit breaker pattern
  - Error tracking and recovery
"""

from __future__ import annotations

import logging
import time
import functools
from typing import Any, Callable, Optional, TypeVar
from datetime import datetime, timezone
from enum import Enum
import json

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
#  Structured Logging Context
# ════════════════════════════════════════════════════════════════════════════════

class LogContext:
    """Context manager for structured logging with consistent fields."""
    
    _context_stack = []
    
    def __init__(
        self,
        document_id: str | None = None,
        chunk_id: str | None = None,
        image_id: str | None = None,
        stage: str | None = None,
        **kwargs
    ):
        self.document_id = document_id
        self.chunk_id = chunk_id
        self.image_id = image_id
        self.stage = stage
        self.extra_fields = kwargs
        self.start_time = time.time()
    
    def __enter__(self):
        self._context_stack.append(self)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context_stack.pop()
        return False
    
    @classmethod
    def current(cls) -> LogContext | None:
        return cls._context_stack[-1] if cls._context_stack else None
    
    def log(
        self,
        level: int,
        msg: str,
        status: str | None = None,
        error: str | None = None,
        **kwargs
    ):
        """Log a message with context fields."""
        latency = time.time() - self.start_time
        
        # Build structured log entry
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": msg,
            "status": status or "info",
            "latency_ms": round(latency * 1000, 2),
        }
        
        # Add context fields
        if self.document_id:
            log_data["document_id"] = self.document_id
        if self.chunk_id:
            log_data["chunk_id"] = self.chunk_id
        if self.image_id:
            log_data["image_id"] = self.image_id
        if self.stage:
            log_data["stage"] = self.stage
        
        # Add extra fields
        log_data.update(self.extra_fields)
        log_data.update(kwargs)
        
        # Add error if present
        if error:
            log_data["error"] = str(error)
        
        # Log as JSON for structured parsing
        formatted_msg = json.dumps(log_data)
        logger.log(level, formatted_msg)
    
    def info(self, msg: str, **kwargs):
        self.log(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        self.log(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, error: Exception | None = None, **kwargs):
        self.log(logging.ERROR, msg, error=error, **kwargs)
    
    def debug(self, msg: str, **kwargs):
        self.log(logging.DEBUG, msg, **kwargs)


# ════════════════════════════════════════════════════════════════════════════════
#  Retry Logic with Exponential Backoff
# ════════════════════════════════════════════════════════════════════════════════

class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay_ms: int = 100,
        max_delay_ms: int = 10000,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retriable_exceptions: tuple[type, ...] = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.initial_delay_ms = initial_delay_ms
        self.max_delay_ms = max_delay_ms
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retriable_exceptions = retriable_exceptions
    
    def get_delay(self, attempt: int) -> float:
        """Get delay in seconds for given attempt number."""
        import random
        
        delay_ms = min(
            self.max_delay_ms,
            self.initial_delay_ms * (self.backoff_factor ** attempt)
        )
        
        if self.jitter:
            delay_ms *= (0.5 + random.random())
        
        return delay_ms / 1000.0


def retry(config: RetryConfig | None = None) -> Callable:
    """Decorator for retry logic with exponential backoff."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.retriable_exceptions as exc:
                    last_exception = exc
                    if attempt < config.max_attempts - 1:
                        delay = config.get_delay(attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {exc}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed for {func.__name__}: {exc}"
                        )
            
            raise last_exception or Exception(f"Failed after {config.max_attempts} attempts")
        
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
#  Timeout Handling
# ════════════════════════════════════════════════════════════════════════════════

def with_timeout(timeout_seconds: float) -> Callable:
    """
    Decorator to add timeout to a function.
    
    Note: This uses signals and only works on Unix systems.
    For cross-platform, consider using thread-based timeout.
    """
    import signal
    
    class TimeoutException(Exception):
        pass
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise TimeoutException(f"Function {func.__name__} timed out after {timeout_seconds}s")
            
            # Save old handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel alarm
                return result
            except TimeoutException:
                raise
            finally:
                signal.signal(signal.SIGALRM, old_handler)
                signal.alarm(0)
        
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
#  Circuit Breaker Pattern
# ════════════════════════════════════════════════════════════════════════════════

class CircuitBreakerState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Stop calling, fail fast
    HALF_OPEN = "half_open"  # Test if recovered


class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures.
    
    Transitions:
    - CLOSED -> OPEN: after failure_threshold consecutive failures
    - OPEN -> HALF_OPEN: after timeout_seconds
    - HALF_OPEN -> CLOSED: if call succeeds
    - HALF_OPEN -> OPEN: if call fails
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 60,
        expected_exception: type = Exception,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.expected_exception = expected_exception
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
            else:
                raise Exception(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as exc:
            self._on_failure()
            raise
    
    def _on_success(self):
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 2:
                self.state = CircuitBreakerState.CLOSED
                logger.info(f"Circuit breaker '{self.name}' recovered to CLOSED state")
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.error(
                f"Circuit breaker '{self.name}' opened after "
                f"{self.failure_count} failures"
            )
    
    def _should_attempt_reset(self) -> bool:
        if not self.last_failure_time:
            return True
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.recovery_timeout_seconds


# ════════════════════════════════════════════════════════════════════════════════
#  Service Health Tracking
# ════════════════════════════════════════════════════════════════════════════════

class ServiceHealthTracker:
    """Track health of external services and dependencies."""
    
    def __init__(self):
        self.services: dict[str, dict] = {}
    
    def register(self, service_name: str):
        """Register a service for health tracking."""
        if service_name not in self.services:
            self.services[service_name] = {
                "status": "unknown",
                "last_check": None,
                "error": None,
                "consecutive_failures": 0,
            }
    
    def mark_success(self, service_name: str):
        """Mark a service as healthy."""
        self.register(service_name)
        service = self.services[service_name]
        service["status"] = "healthy"
        service["last_check"] = datetime.now(timezone.utc).isoformat()
        service["error"] = None
        service["consecutive_failures"] = 0
    
    def mark_failure(self, service_name: str, error: str):
        """Mark a service as unhealthy."""
        self.register(service_name)
        service = self.services[service_name]
        service["status"] = "unhealthy"
        service["last_check"] = datetime.now(timezone.utc).isoformat()
        service["error"] = error
        service["consecutive_failures"] += 1
    
    def get_status(self, service_name: str) -> dict:
        """Get current status of a service."""
        self.register(service_name)
        return self.services[service_name]
    
    def get_all_status(self) -> dict[str, dict]:
        """Get status of all services."""
        return dict(self.services)
    
    def is_healthy(self, service_name: str) -> bool:
        """Check if service is currently healthy."""
        return self.get_status(service_name)["status"] == "healthy"


# ════════════════════════════════════════════════════════════════════════════════
#  Graceful Degradation Helpers
# ════════════════════════════════════════════════════════════════════════════════

class DegradationStrategy:
    """Strategy for gracefully degrading when services fail."""
    
    @staticmethod
    def fallback_to_text(
        func_with_images: Callable,
        func_text_only: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Try to use image-based function, fall back to text-only.
        """
        try:
            return func_with_images(*args, **kwargs)
        except Exception as exc:
            logger.warning(
                f"Image processing failed ({exc}), falling back to text-only approach"
            )
            return func_text_only(*args, **kwargs)
    
    @staticmethod
    def fallback_with_cache(
        func: Callable,
        cache_get: Callable[[str], Any | None],
        cache_key: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Try function, fall back to cached result if available.
        """
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as exc:
            cached = cache_get(cache_key)
            if cached is not None:
                logger.warning(
                    f"Function failed ({exc}), using cached result"
                )
                return cached
            raise
    
    @staticmethod
    def fallback_partial(
        func: Callable,
        partial_handler: Callable,
        *args,
        **kwargs
    ) -> tuple[Any, dict]:
        """
        Execute function and return both result and partial failure info.
        Allows partial success (e.g., 8 of 10 images embedded successfully).
        
        Returns:
            (result, partial_failures)
            partial_failures is a dict with details on what failed
        """
        partial_failures = {}
        try:
            result = func(*args, **kwargs)
            return result, partial_failures
        except Exception as exc:
            # Attempt partial handling
            return partial_handler(*args, **kwargs), {
                "error": str(exc),
                "handler": "partial"
            }


# ════════════════════════════════════════════════════════════════════════════════
#  Error Recovery Helpers
# ════════════════════════════════════════════════════════════════════════════════

class RecoverableError(Exception):
    """Error that can be recovered with retry or fallback."""
    pass


class PermanentError(Exception):
    """Error that cannot be recovered."""
    pass


def is_recoverable(exc: Exception) -> bool:
    """Determine if an exception is recoverable."""
    # Network errors, timeouts, temporary failures are recoverable
    recoverable_types = (
        TimeoutError,
        ConnectionError,
        IOError,
        RecoverableError,
    )
    return isinstance(exc, recoverable_types)
