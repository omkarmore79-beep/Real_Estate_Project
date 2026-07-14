"""
Centralized Caching Service for the RAG system.
Integrates with Redis for production caching and falls back to a thread-safe
in-memory cache for local development/offline environments.
Tracks cache hit rates for observability.
"""

from __future__ import annotations

import logging
import os
import json
import hashlib
from typing import Any

logger = logging.getLogger(__name__)

# ── Caching Metrics ───────────────────────────────────────────────────────────
_cache_metrics = {
    "hits": 0,
    "misses": 0,
    "total": 0
}

# ── Fallback Memory Store ──────────────────────────────────────────────────────
_memory_cache: dict[str, str] = {}

# ── Redis Client Initialization ────────────────────────────────────────────────
_redis_client: Any = None
_redis_available = False
_redis_initialized = False

def get_redis_client():
    global _redis_client, _redis_available, _redis_initialized
    if _redis_initialized:
        return _redis_client if _redis_available else None

    _redis_initialized = True
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_DB", "0"))
    
    try:
        import redis
        client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            socket_timeout=1.0,
            socket_connect_timeout=1.0,
            decode_responses=True,
            retry_on_timeout=False,
            retry=None
        )
        # Test connection
        client.ping()
        _redis_client = client
        _redis_available = True
        logger.info("Successfully connected to Redis at %s:%d", redis_host, redis_port)
    except Exception as exc:
        logger.warning("Redis is not available, falling back to local in-memory cache: %s", exc)
        _redis_client = None
        _redis_available = False
        
    return _redis_client if _redis_available else None

def _get_hash_key(key: str) -> str:
    """If a key is extremely long (like a text chunk), hash it to a manageable string."""
    if len(key) > 120:
        return "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest()
    return key

# ── Public API ────────────────────────────────────────────────────────────────

def get_cache(key: str) -> str | None:
    """Retrieve string value from cache."""
    _cache_metrics["total"] += 1
    safe_key = _get_hash_key(key)
    
    # 1. Try Redis
    redis_client = get_redis_client()
    if redis_client:
        try:
            val = redis_client.get(safe_key)
            if val is not None:
                _cache_metrics["hits"] += 1
                return val
        except Exception as exc:
            logger.debug("Redis read failed: %s", exc)
            
    # 2. Try In-Memory Fallback
    if safe_key in _memory_cache:
        _cache_metrics["hits"] += 1
        return _memory_cache[safe_key]
        
    _cache_metrics["misses"] += 1
    return None

def set_cache(key: str, value: str, expire_seconds: int = 3600) -> bool:
    """Store string value in cache with optional expiry."""
    safe_key = _get_hash_key(key)
    
    # 1. Try Redis
    redis_client = get_redis_client()
    if redis_client:
        try:
            redis_client.set(safe_key, value, ex=expire_seconds)
            return True
        except Exception as exc:
            logger.debug("Redis write failed: %s", exc)
            
    # 2. Try In-Memory Fallback
    _memory_cache[safe_key] = value
    return True

def get_json_cache(key: str) -> Any | None:
    """Retrieve JSON-deserialized object from cache."""
    val = get_cache(key)
    if val is not None:
        try:
            return json.loads(val)
        except Exception:
            return None
    return None

def set_json_cache(key: str, value: Any, expire_seconds: int = 3600) -> bool:
    """Store JSON-serializable object in cache."""
    try:
        val_str = json.dumps(value)
        return set_cache(key, val_str, expire_seconds)
    except Exception:
        return False

def get_cache_metrics() -> dict[str, Any]:
    """Retrieve cache performance statistics."""
    hits = _cache_metrics["hits"]
    total = _cache_metrics["total"]
    hit_rate = (hits / total) if total > 0 else 0.0
    return {
        "redis_active": _redis_available,
        "hits": hits,
        "misses": _cache_metrics["misses"],
        "total_requests": total,
        "hit_rate": round(hit_rate, 4)
    }

def clear_cache() -> None:
    """Clear all items from cache (Redis flush + local dict clear)."""
    global _memory_cache
    _memory_cache.clear()
    
    redis_client = get_redis_client()
    if redis_client:
        try:
            redis_client.flushdb()
            logger.info("Flushed Redis cache.")
        except Exception as exc:
            logger.error("Failed to flush Redis: %s", exc)
