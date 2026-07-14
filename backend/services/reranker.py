import time
import asyncio
import logging
from typing import Any, List
import voyageai

from config import VOYAGE_API_KEY, VOYAGE_RERANK_MODEL, RERANK_TIMEOUT

logger = logging.getLogger(__name__)

# Metrics tracking
_total_requests = 0
_total_latencies = 0.0
_total_tokens = 0
_total_failures = 0
_total_retries = 0

def get_rerank_metrics() -> dict:
    """Return metrics gathered by the reranker service."""
    avg_latency = (_total_latencies / _total_requests) if _total_requests > 0 else 0.0
    return {
        "total_requests": _total_requests,
        "total_latencies": _total_latencies,
        "average_latency": avg_latency,
        "total_tokens_billed": _total_tokens,
        "total_failures": _total_failures,
        "total_retries": _total_retries
    }

def get_chunk_text_representation(r: dict) -> str:
    """
    Format a text or image chunk into a single cohesive string for reranking.
    Incorporates image captions, nearby paragraphs, and OCR labels.
    """
    meta = r.get("metadata") or {}
    if r.get("source_type") == "image":
        parts = []
        caption = r.get("caption") or meta.get("caption", "")
        if caption:
            parts.append(f"Image Caption: {caption}")
        nearby = meta.get("nearby_text", "") or meta.get("nearby_page_text", "")
        if nearby:
            parts.append(f"Nearby Paragraphs: {nearby}")
        ocr = meta.get("ocr_context", "") or meta.get("ocr_labels", "")
        if ocr:
            parts.append(f"OCR Labels inside diagram: {ocr}")
        
        # If nothing populated, fallback to content
        if not parts:
            parts.append(r.get("content", ""))
        return "\n".join(parts)
    else:
        return r.get("content", "")

def rerank_sync(query: str, chunks: List[dict], top_k: int = 10) -> List[dict]:
    """
    Synchronously rerank chunks using Voyage Hosted Reranker.
    Gracefully falls back to original ordering if API errors persist.
    """
    global _total_requests, _total_latencies, _total_tokens, _total_failures, _total_retries
    
    if not chunks:
        return []
    
    if not VOYAGE_API_KEY:
        logger.warning("VOYAGE_API_KEY is not set. Reranker returning chunks as-is.")
        return chunks

    docs = [get_chunk_text_representation(c) for c in chunks]
    client = voyageai.Client(api_key=VOYAGE_API_KEY)
    
    max_retries = 3
    backoff = 1.0
    
    start_time = time.perf_counter()
    _total_requests += 1
    
    for attempt in range(max_retries):
        try:
            logger.info("Calling Voyage Rerank API (model=%s, candidates=%d)", VOYAGE_RERANK_MODEL, len(chunks))
            response = client.rerank(
                query=query,
                documents=docs,
                model=VOYAGE_RERANK_MODEL,
                top_k=top_k,
                truncation=True
            )
            
            latency = time.perf_counter() - start_time
            _total_latencies += latency
            
            # Track usage tokens if present
            try:
                tokens = getattr(response, "usage", {}).get("total_tokens", 0)
                _total_tokens += tokens
            except Exception:
                pass
            
            # Map rerank scores back to original chunks
            reranked = []
            for r in response.results:
                orig_chunk = dict(chunks[r.index])
                orig_chunk["rerank_score"] = float(r.relevance_score)
                reranked.append(orig_chunk)
                
            # If some chunks were not returned, append them at the end with score 0
            returned_indices = {r.index for r in response.results}
            for idx, c in enumerate(chunks):
                if idx not in returned_indices:
                    orig_chunk = dict(c)
                    orig_chunk["rerank_score"] = 0.0
                    reranked.append(orig_chunk)
                    
            return reranked
            
        except Exception as exc:
            _total_retries += 1
            logger.warning(
                "Voyage sync rerank attempt %d failed: %s. Retrying in %.2fs...",
                attempt + 1, exc, backoff
            )
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2.0
            else:
                _total_failures += 1
                logger.error("Voyage sync rerank failed after %d attempts. Falling back to vector scores.", max_retries)
                
    # Fallback: keep original order and assign normalized RRF or vector scores
    for c in chunks:
        c["rerank_score"] = c.get("score", 0.0)
    return chunks

async def rerank_async(query: str, chunks: List[dict], top_k: int = 10) -> List[dict]:
    """
    Asynchronously rerank chunks using Voyage Hosted Reranker.
    Gracefully falls back to original ordering if API errors persist.
    """
    global _total_requests, _total_latencies, _total_tokens, _total_failures, _total_retries
    
    if not chunks:
        return []
        
    if not VOYAGE_API_KEY:
        logger.warning("VOYAGE_API_KEY is not set. Reranker returning chunks as-is.")
        return chunks

    docs = [get_chunk_text_representation(c) for c in chunks]
    
    max_retries = 3
    backoff = 1.0
    
    start_time = time.perf_counter()
    _total_requests += 1
    
    for attempt in range(max_retries):
        try:
            logger.info("Calling Voyage Async Rerank API (model=%s, candidates=%d)", VOYAGE_RERANK_MODEL, len(chunks))
            
            # Using AsyncClient
            async_client = voyageai.AsyncClient(api_key=VOYAGE_API_KEY)
            
            # Wrap API call with timeout
            response = await asyncio.wait_for(
                async_client.rerank(
                    query=query,
                    documents=docs,
                    model=VOYAGE_RERANK_MODEL,
                    top_k=top_k,
                    truncation=True
                ),
                timeout=float(RERANK_TIMEOUT)
            )
            
            latency = time.perf_counter() - start_time
            _total_latencies += latency
            
            # Track usage tokens if present
            try:
                tokens = getattr(response, "usage", {}).get("total_tokens", 0)
                _total_tokens += tokens
            except Exception:
                pass
            
            reranked = []
            for r in response.results:
                orig_chunk = dict(chunks[r.index])
                orig_chunk["rerank_score"] = float(r.relevance_score)
                reranked.append(orig_chunk)
                
            # Append any un-returned chunks at the end
            returned_indices = {r.index for r in response.results}
            for idx, c in enumerate(chunks):
                if idx not in returned_indices:
                    orig_chunk = dict(c)
                    orig_chunk["rerank_score"] = 0.0
                    reranked.append(orig_chunk)
                    
            return reranked
            
        except Exception as exc:
            _total_retries += 1
            logger.warning(
                "Voyage async rerank attempt %d failed: %s. Retrying in %.2fs...",
                attempt + 1, exc, backoff
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(backoff)
                backoff *= 2.0
            else:
                _total_failures += 1
                logger.error("Voyage async rerank failed after %d attempts. Falling back to vector scores.", max_retries)
                
    # Fallback
    for c in chunks:
        c["rerank_score"] = c.get("score", 0.0)
    return chunks
