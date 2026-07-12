"""
Observability and Monitoring Module.
Tracks execution times (OCR, embeddings, search, reranking, LLM) and cache hits.
"""

from __future__ import annotations

import time
import logging
from contextlib import contextmanager
from typing import Any, Callable

logger = logging.getLogger(__name__)

# In-memory metrics storage
_metrics = {
    "runs_count": 0,
    "upload_times": [],
    "ocr_times": [],
    "embedding_times": [],
    "dense_search_latencies": [],
    "bm25_search_latencies": [],
    "reranking_times": [],
    "llm_generation_times": [],
    "total_response_times": [],
}

def record_metric(name: str, value: float) -> None:
    """Record a numerical latency metric."""
    if name in _metrics and isinstance(_metrics[name], list):
        _metrics[name].append(value)
        # Cap length to avoid memory leaks
        if len(_metrics[name]) > 1000:
            _metrics[name] = _metrics[name][-1000:]

@contextmanager
def time_stage(stage_name: str):
    """Context manager to measure latency of a specific RAG pipeline stage."""
    start_time = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start_time
    record_metric(stage_name, elapsed)
    logger.info("[OBSERVABILITY] Stage '%s' took %.4f seconds", stage_name, elapsed)

def time_function(stage_name: str) -> Callable:
    """Decorator to measure latency of a function call."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            record_metric(stage_name, elapsed)
            logger.info("[OBSERVABILITY] Function '%s.%s' took %.4f seconds", func.__module__, func.__name__, elapsed)
            return result
        return wrapper
    return decorator

def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)

def get_observability_metrics() -> dict[str, Any]:
    """Retrieve summarized latency averages and database statistics."""
    from utils.cache import get_cache_metrics
    cache_stats = get_cache_metrics()
    
    return {
        "cache_hit_rate": cache_stats.get("hit_rate", 0.0),
        "redis_active": cache_stats.get("redis_active", False),
        "averages": {
            "upload_time_seconds": _average(_metrics["upload_times"]),
            "ocr_time_seconds": _average(_metrics["ocr_times"]),
            "embedding_time_seconds": _average(_metrics["embedding_times"]),
            "vector_search_latency_seconds": _average(_metrics["dense_search_latencies"]),
            "bm25_search_latency_seconds": _average(_metrics["bm25_search_latencies"]),
            "rerank_time_seconds": _average(_metrics["reranking_times"]),
            "llm_generation_time_seconds": _average(_metrics["llm_generation_times"]),
            "total_response_time_seconds": _average(_metrics["total_response_times"]),
        },
        "raw_counts": {
            "total_requests": cache_stats.get("total_requests", 0),
            "cache_hits": cache_stats.get("hits", 0),
            "cache_misses": cache_stats.get("misses", 0)
        }
    }
