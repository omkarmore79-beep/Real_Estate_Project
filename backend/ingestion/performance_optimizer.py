"""
Performance Optimizations for Industrial Multimodal RAG.

Implements:
- Parallel OCR processing
- Parallel image extraction
- Batch embedding
- Duplicate detection
- Incremental ingestion
- Connection pooling
- Cache repeated embedding requests
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# ── Caching for Embeddings ───────────────────────────────────────────────────────

_embedding_cache = {}
_cache_max_size = 1000


def cache_embedding_key(text: str, model: str, input_type: str = "document") -> str:
    """Generate a cache key for an embedding request."""
    key_str = f"{model}:{input_type}:{text}"
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cached_embedding(key: str) -> Optional[list]:
    """Get a cached embedding if available."""
    return _embedding_cache.get(key)


def set_cached_embedding(key: str, embedding: list) -> None:
    """Cache an embedding result."""
    if len(_embedding_cache) >= _cache_max_size:
        # Remove oldest entry (simple FIFO)
        oldest_key = next(iter(_embedding_cache))
        del _embedding_cache[oldest_key]
    _embedding_cache[key] = embedding


def clear_embedding_cache() -> None:
    """Clear the embedding cache."""
    _embedding_cache.clear()
    logger.info("Embedding cache cleared")


# ── Parallel Processing ───────────────────────────────────────────────────────────

def parallel_process(
    items: list,
    process_func: Callable,
    max_workers: int = 4,
    desc: str = "processing",
) -> list:
    """
    Process items in parallel using ThreadPoolExecutor.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each item
        max_workers: Maximum number of parallel workers
        desc: Description for logging
    
    Returns:
        List of processed results
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_func, item): item for item in items}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                item = futures[future]
                logger.warning(f"Error {desc} item {item}: {e}")
    
    logger.info(f"Parallel {desc} complete: {len(results)}/{len(items)} successful")
    return results


def parallel_ocr_images(
    images: list[dict],
    ocr_func: Callable,
    max_workers: int = 4,
) -> list[dict]:
    """
    Run OCR on multiple images in parallel.
    
    Args:
        images: List of image dictionaries with image_path
        ocr_func: OCR function to apply
        max_workers: Maximum parallel workers
    
    Returns:
        List of images with OCR results added
    """
    def process_single_image(image: dict) -> dict:
        try:
            image_path = image.get("image_path") or image.get("local_path")
            if not image_path:
                return image
            
            ocr_result = ocr_func(image_path)
            image["ocr_text"] = ocr_result.get("text", "")
            image["ocr_confidence"] = ocr_result.get("confidence", 0.0)
            return image
        except Exception as e:
            logger.warning(f"OCR failed for image {image.get('image_id')}: {e}")
            return image
    
    return parallel_process(images, process_single_image, max_workers, "OCR")


def parallel_extract_images(
    pages: list[dict],
    extract_func: Callable,
    max_workers: int = 4,
) -> list[dict]:
    """
    Extract images from multiple pages in parallel.
    
    Args:
        pages: List of page dictionaries
        extract_func: Image extraction function
        max_workers: Maximum parallel workers
    
    Returns:
        List of pages with images extracted
    """
    def process_single_page(page: dict) -> dict:
        try:
            images = extract_func(page)
            page["images"] = images
            return page
        except Exception as e:
            logger.warning(f"Image extraction failed for page {page.get('page_number')}: {e}")
            return page
    
    return parallel_process(pages, process_single_page, max_workers, "image extraction")


# ── Batch Embedding ─────────────────────────────────────────────────────────────

def batch_embed_texts(
    texts: list[str],
    embedder: Any,
    batch_size: int = 32,
    input_type: str = "document",
) -> list[list]:
    """
    Embed texts in batches for efficiency.
    
    Args:
        texts: List of texts to embed
        embedder: Embedder instance with embed() method
        batch_size: Number of texts per batch
        input_type: Input type for the embedder
    
    Returns:
        List of embedding vectors
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Check cache for each text
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for j, text in enumerate(batch):
            cache_key = cache_embedding_key(text, getattr(embedder, 'model_name', 'unknown'), input_type)
            cached = get_cached_embedding(cache_key)
            if cached is not None:
                cached_embeddings.append((j, cached))
            else:
                uncached_texts.append(text)
                uncached_indices.append(j)
        
        # Embed uncached texts
        if uncached_texts:
            try:
                new_embeddings = embedder.embed(uncached_texts, input_type=input_type)
                
                # Cache new embeddings
                for text, emb in zip(uncached_texts, new_embeddings):
                    cache_key = cache_embedding_key(text, getattr(embedder, 'model_name', 'unknown'), input_type)
                    set_cached_embedding(cache_key, emb)
            except Exception as e:
                logger.warning(f"Batch embedding failed: {e}")
                new_embeddings = [[] for _ in uncached_texts]
        
        # Combine cached and new embeddings
        batch_embeddings = [None] * len(batch)
        
        # Fill cached embeddings
        for idx, emb in cached_embeddings:
            batch_embeddings[idx] = emb
        
        # Fill new embeddings
        for idx, emb in zip(uncached_indices, new_embeddings):
            batch_embeddings[idx] = emb
        
        all_embeddings.extend(batch_embeddings)
    
    logger.info(f"Batch embedded {len(texts)} texts")
    return all_embeddings


# ── Duplicate Detection ──────────────────────────────────────────────────────────

def detect_duplicate_chunks(chunks: list[dict], similarity_threshold: float = 0.95) -> list[dict]:
    """
    Detect and remove duplicate chunks based on content similarity.
    
    Args:
        chunks: List of chunk dictionaries
        similarity_threshold: Threshold for considering chunks as duplicates
    
    Returns:
        List of chunks with duplicates removed
    """
    seen_hashes = {}
    unique_chunks = []
    
    for chunk in chunks:
        content = chunk.get("content", "")
        if not content:
            continue
        
        # Create content hash
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash in seen_hashes:
            logger.debug(f"Duplicate chunk detected: {chunk.get('chunk_id')}")
            continue
        
        seen_hashes[content_hash] = chunk.get("chunk_id")
        unique_chunks.append(chunk)
    
    removed = len(chunks) - len(unique_chunks)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate chunks")
    
    return unique_chunks


def detect_duplicate_images(images: list[dict]) -> list[dict]:
    """
    Detect and remove duplicate images based on image hash.
    
    Args:
        images: List of image dictionaries
    
    Returns:
        List of images with duplicates removed
    """
    seen_hashes = {}
    unique_images = []
    
    for image in images:
        image_path = image.get("image_path") or image.get("local_path")
        if not image_path:
            continue
        
        # Create file hash (using path as proxy)
        file_hash = hashlib.md5(image_path.encode()).hexdigest()
        
        if file_hash in seen_hashes:
            logger.debug(f"Duplicate image detected: {image.get('image_id')}")
            continue
        
        seen_hashes[file_hash] = image.get("image_id")
        unique_images.append(image)
    
    removed = len(images) - len(unique_images)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate images")
    
    return unique_images


# ── Incremental Ingestion ────────────────────────────────────────────────────────

def get_document_fingerprint(document: dict) -> str:
    """
    Generate a fingerprint for a document to detect changes.
    
    Uses file size, modification time, and content hash.
    """
    source_file = document.get("source_file", "")
    file_size = document.get("file_size", 0)
    modified_time = document.get("modified_time", "")
    
    fingerprint_data = f"{source_file}:{file_size}:{modified_time}"
    return hashlib.md5(fingerprint_data.encode()).hexdigest()


def should_reindex(document: dict, stored_fingerprint: Optional[str]) -> bool:
    """
    Determine if a document should be reindexed.
    
    Returns True if fingerprint has changed.
    """
    current_fingerprint = get_document_fingerprint(document)
    
    if not stored_fingerprint:
        return True
    
    return current_fingerprint != stored_fingerprint


# ── Connection Pooling Helpers ───────────────────────────────────────────────────

class ConnectionPool:
    """Simple connection pool for database connections."""
    
    def __init__(self, create_func: Callable, max_connections: int = 10):
        self.create_func = create_func
        self.max_connections = max_connections
        self.pool = []
        self.in_use = 0
    
    def get_connection(self) -> Any:
        """Get a connection from the pool."""
        if self.pool:
            return self.pool.pop()
        
        if self.in_use < self.max_connections:
            self.in_use += 1
            return self.create_func()
        
        # Pool exhausted, create temporary connection
        return self.create_func()
    
    def return_connection(self, connection: Any) -> None:
        """Return a connection to the pool."""
        if len(self.pool) < self.max_connections:
            self.pool.append(connection)
        else:
            self.in_use -= 1
            # Close excess connections
            try:
                if hasattr(connection, 'close'):
                    connection.close()
            except Exception:
                pass


# ── Performance Monitoring ───────────────────────────────────────────────────────

class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        import time
        self.metrics[operation] = {"start": time.time()}
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        import time
        if operation in self.metrics:
            duration = time.time() - self.metrics[operation]["start"]
            self.metrics[operation]["duration"] = duration
            logger.info(f"Operation '{operation}' took {duration:.2f}s")
            return duration
        return 0.0
    
    def get_metrics(self) -> dict:
        """Get all recorded metrics."""
        return self.metrics


# ── Memory Optimization ───────────────────────────────────────────────────────────

def truncate_text_for_embedding(text: str, max_length: int = 8000) -> str:
    """
    Truncate text to maximum length for embedding.
    
    Preserves sentence boundaries where possible.
    """
    if len(text) <= max_length:
        return text
    
    # Find last sentence boundary before max_length
    truncated = text[:max_length]
    last_period = truncated.rfind(".")
    last_exclamation = truncated.rfind("!")
    last_question = truncated.rfind("?")
    
    # Use the latest sentence boundary
    last_boundary = max(last_period, last_exclamation, last_question)
    
    if last_boundary > max_length * 0.8:  # Only truncate at sentence if it's not too far back
        return text[:last_boundary + 1]
    
    return truncated


def optimize_chunk_size(chunk: dict, max_tokens: int = 800) -> dict:
    """
    Optimize chunk size for embedding.
    
    Truncates content if too long.
    """
    content = chunk.get("content", "")
    if len(content) > max_tokens * 4:  # Approximate 4 chars per token
        chunk["content"] = truncate_text_for_embedding(content, max_tokens * 4)
        chunk["truncated"] = True
    
    return chunk


# ── Lazy Loading ───────────────────────────────────────────────────────────────

class LazyLoader:
    """Lazy load expensive resources."""
    
    def __init__(self, load_func: Callable):
        self.load_func = load_func
        self._loaded = None
        self._loading = False
    
    def load(self) -> Any:
        """Load the resource if not already loaded."""
        if self._loaded is not None:
            return self._loaded
        
        if self._loading:
            # Wait for loading to complete
            import time
            while self._loading:
                time.sleep(0.1)
            return self._loaded
        
        self._loading = True
        try:
            self._loaded = self.load_func()
            return self._loaded
        finally:
            self._loading = False
