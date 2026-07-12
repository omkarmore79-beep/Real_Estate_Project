"""
Hybrid Retriever — Dense + BM25 keyword + Image search with Multi-Query Expansion,
RRF fusion, cross-encoder reranking, confidence scoring, and caching.

Pipeline:
  1. Expand user query using LLM (generate 3 synonyms/formulations)
  2. For each query variation:
     - Perform dense text search (Qdrant cosine similarity)
     - Perform BM25 keyword search over payload text
  3. Image search (when visual query detected or include_images=True)
  4. Reciprocal Rank Fusion (RRF) to merge all text result lists
  5. Cross-encoder reranking (BAAI/bge-reranker-large) using the original user query
  6. Confidence scoring to filter low-relevance results (threshold = 0.35)
  7. Return top-k results with full metadata.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from utils.cache import get_json_cache, set_json_cache

logger = logging.getLogger(__name__)

# ── Image-intent vocabulary ────────────────────────────────────────────────────
_IMAGE_INTENT_PHRASES = (
    "floor plan", "floor plans", "unit plan", "flat plan",
    "master plan", "site layout", "township layout",
    "location map", "location plan", "connectivity map", "route map",
    "amenity", "amenities", "clubhouse", "gym", "pool", "park",
    "elevation", "building view", "tower view", "exterior",
    "inside view", "room view",
    "parking plan", "parking layout",
    "show", "image", "photo", "picture", "display", "view", "layout",
)

_IMAGE_TYPE_MAP = {
    "floor plan": "floor_plan", "unit plan": "floor_plan", "flat plan": "floor_plan",
    "master plan": "master_plan", "site layout": "master_plan",
    "location map": "location_map", "location plan": "location_map",
    "amenity": "amenity", "amenities": "amenity",
    "elevation": "exterior", "building view": "exterior",
    "exterior": "exterior", "interior": "interior",
}

# ── RRF constant ───────────────────────────────────────────────────────────────
RRF_K = 60

# ── Reranker singleton ─────────────────────────────────────────────────────────
_reranker = None

def _get_reranker():
    """Lazy-load the BGE reranker cross-encoder."""
    global _reranker
    if _reranker is None:
        from config import RERANKER_MODEL
        logger.info("Loading reranker model: %s …", RERANKER_MODEL)
        try:
            from FlagEmbedding import FlagReranker
            _reranker = FlagReranker(RERANKER_MODEL, use_fp16=False)
            logger.info("Reranker loaded (FlagEmbedding).")
        except Exception as exc:
            logger.warning("FlagEmbedding reranker failed (%s), trying sentence-transformers CrossEncoder: %s", RERANKER_MODEL, exc)
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
            logger.info("Reranker loaded (CrossEncoder).")
    return _reranker

def reranker_status() -> dict:
    return {"loaded": _reranker is not None}

# ════════════════════════════════════════════════════════════════════════════════
#  Public API
# ════════════════════════════════════════════════════════════════════════════════

def retrieve(
    query: str,
    document_id: str | None = None,
    project_name: str | None = None,
    builder: str | None = None,
    document_type: str | None = None,
    include_images: bool = False,
    top_k: int = 8,
) -> list[dict]:
    """
    Run hybrid retrieval and return ranked results. Uses caching and multi-query expansion.
    """
    # 0. Check cache
    cache_key = f"retrieve:{query}:{document_id}:{project_name}:{builder}:{document_type}:{include_images}:{top_k}"
    cached_results = get_json_cache(cache_key)
    if cached_results is not None:
        logger.info("Retrieval cache hit for query: %s", query[:40])
        return cached_results

    from retrieval.qdrant_service import (
        search_images,
        search_text_dense,
        search_text_keyword,
    )
    from retrieval.embeddings import get_image_embedder, get_text_embedder
    from retrieval.query_analyzer import expand_query, classify_query_intent

    # Build filters
    text_filters: dict[str, Any] = {}
    image_filters: dict[str, Any] = {}
    if document_id:
        text_filters["document_id"] = document_id
        image_filters["document_id"] = document_id
    if project_name:
        text_filters["project_name"] = project_name
        image_filters["project_name"] = project_name
    if builder:
        text_filters["builder"] = builder
    if document_type:
        text_filters["document_type"] = document_type

    # Detect intent
    image_intent = detect_image_intent(query)
    should_search_images = include_images or image_intent["requires_image"]

    # 1. Multi-Query Expansion
    queries = expand_query(query)
    logger.info("Expanded query: '%s' into %d variations: %s", query, len(queries), queries)

    all_dense_results = []
    all_keyword_results = []

    # 2. Run Dense + Sparse Search for each query variation
    text_embedder = get_text_embedder()
    for q_var in queries:
        # A. Dense text search
        try:
            query_vec = text_embedder.embed(q_var)
            dense_res = search_text_dense(query_vec, text_filters or None, top_k=top_k * 2)
            all_dense_results.append(dense_res)
        except Exception as exc:
            logger.warning("Dense text search failed for variation '%s': %s", q_var, exc)

        # B. BM25 keyword search
        try:
            keyword_res = search_text_keyword(q_var, text_filters or None, top_k=top_k * 2)
            all_keyword_results.append(keyword_res)
        except Exception as exc:
            logger.warning("BM25 keyword search failed for variation '%s': %s", q_var, exc)

    # 3. Image search
    image_results: list[dict] = []
    if should_search_images:
        try:
            img_embedder = get_image_embedder()
            img_query_vec = img_embedder.embed_text(query)
            image_results = search_images(
                img_query_vec,
                image_filters or None,
                top_k=6,
            )
        except Exception as exc:
            logger.warning("Image search failed: %s", exc)

    # 4. RRF fusion across all variations and dense/sparse lists
    all_lists = all_dense_results + all_keyword_results
    text_fused = _rrf_fuse(all_lists)
    all_results = _merge_text_and_image(text_fused, image_results, image_intent)

    # 5. Reranking using Cross-Encoder against the original query
    text_only = [r for r in all_results if r.get("source_type") == "text"]
    image_only = [r for r in all_results if r.get("source_type") == "image"]

    if len(text_only) > 1:
        text_only = _rerank(query, text_only)

    # 6. Confidence Filtering
    # Filter text candidates whose confidence score is below 0.35
    confident_text = []
    for r in text_only:
        # Normalize rerank score
        score = r.get("rerank_score", 0.0)
        # If score is in logits (negative), apply soft mapping
        if score < 0:
            confidence = float(1.0 / (1.0 + float(math.exp(-score))))
        else:
            confidence = score
            
        r["confidence_score"] = confidence
        
        # Keep if confidence is above threshold
        if confidence >= 0.35:
            confident_text.append(r)

    # Interleave: top text + top images, then slice to top_k
    final = _interleave(confident_text, image_only, top_k)

    # Cache result
    set_json_cache(cache_key, final, expire_seconds=3600)

    return final


def detect_image_intent(query: str) -> dict:
    """
    Detect whether a query requires image results.
    """
    q = query.lower()
    detected_types: list[str] = []

    for phrase, img_type in _IMAGE_TYPE_MAP.items():
        if phrase in q and img_type not in detected_types:
            detected_types.append(img_type)

    requires_image = any(phrase in q for phrase in _IMAGE_INTENT_PHRASES)
    return {"requires_image": requires_image, "detected_types": detected_types}


# ════════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ════════════════════════════════════════════════════════════════════════════════

def _rrf_fuse(result_lists: list[list[dict]]) -> list[dict]:
    """
    Apply Reciprocal Rank Fusion across multiple ranked result lists.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            rid = result.get("id", "")
            if not rid:
                continue
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (RRF_K + rank + 1)
            payloads.setdefault(rid, result)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    merged: list[dict] = []
    for rid, score in fused:
        item = dict(payloads[rid])
        item["rrf_score"] = score
        item["score"] = score
        merged.append(item)

    return merged


def _merge_text_and_image(
    text_results: list[dict],
    image_results: list[dict],
    image_intent: dict,
) -> list[dict]:
    """Tag each result with source_type and citation_id, merge lists, preserving metadata keys."""
    merged: list[dict] = []

    for idx, r in enumerate(text_results):
        payload = r.get("payload", {})
        merged.append(
            {
                "id": r.get("id", ""),
                "score": r.get("score", 0.0),
                "source_type": "text",
                "content": r.get("content", payload.get("text", payload.get("content", ""))),
                "document_id": payload.get("document_id", ""),
                "source_file": payload.get("source_file", ""),
                "page_number": payload.get("page_number"),
                "image_path": None,
                "image_url": None,
                "image_id": None,
                "image_type": None,
                "caption": None,
                "ocr_used": payload.get("ocr_used", False),
                "citation_id": f"text_{idx + 1}",
                "metadata": payload,
            }
        )

    for idx, r in enumerate(image_results):
        payload = r.get("payload", {})
        merged.append(
            {
                "id": r.get("id", ""),
                "score": r.get("score", 0.0),
                "source_type": "image",
                "content": payload.get("caption", payload.get("nearby_page_text", "")),
                "document_id": payload.get("document_id", ""),
                "source_file": payload.get("source_file", ""),
                "page_number": payload.get("page_number"),
                "image_path": payload.get("image_path"),
                "image_url": payload.get("image_url", payload.get("image_path", "")),
                "image_id": payload.get("image_id"),
                "image_type": payload.get("image_type"),
                "caption": payload.get("caption"),
                "ocr_used": payload.get("ocr_used", False),
                "citation_id": f"image_{idx + 1}",
                "metadata": payload,
            }
        )

    return merged


def _rerank(query: str, results: list[dict]) -> list[dict]:
    """
    Rerank results using BAAI/bge-reranker-large cross-encoder.
    """
    try:
        reranker = _get_reranker()
        pairs = [(query, r.get("content", "")[:512]) for r in results]

        try:
            from FlagEmbedding import FlagReranker
            if isinstance(reranker, FlagReranker):
                scores = reranker.compute_score(pairs, normalize=True)
            else:
                scores = reranker.predict(pairs).tolist()
        except Exception:
            scores = reranker.predict(pairs).tolist()

        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i]) if i < len(scores) else 0.0

        results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    except Exception as exc:
        logger.warning("Reranking failed, using RRF order: %s", exc)

    return results


def _interleave(
    text_results: list[dict],
    image_results: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Merge text and image results respecting top_k.
    """
    final: list[dict] = []
    final.extend(text_results[: max(top_k - len(image_results[:3]), top_k // 2)])
    final.extend(image_results[:3])
    return final[:top_k]
