"""
Hybrid Retriever — Dense + BM25 keyword + Image search with RRF fusion
and cross-encoder reranking.

Pipeline:
  1. Dense text search (Qdrant cosine similarity)
  2. BM25 keyword search over payload text
  3. Image search (when visual query detected or include_images=True)
  4. Reciprocal Rank Fusion (RRF) to merge result lists
  5. Cross-encoder reranking (BAAI/bge-reranker-large)

Returns top-k results with full metadata for grounded answer generation.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Image-intent vocabulary ────────────────────────────────────────────────────
_IMAGE_INTENT_PHRASES = (
    "floor plan", "floor plans", "unit plan", "flat plan",
    "master plan", "site layout", "township layout",
    "location map", "location plan", "connectivity map", "route map",
    "amenity", "amenities", "clubhouse", "gym", "pool", "park",
    "elevation", "building view", "tower view", "exterior",
    "interior", "inside view", "room view",
    "parking plan", "parking layout",
    "show", "image", "photo", "picture", "display", "view", "layout",
    "diagram", "schematic", "cross-section", "cross section", "blueprint", "circuit"
)

_IMAGE_TYPE_MAP = {
    "floor plan": "floor_plan", "unit plan": "floor_plan", "flat plan": "floor_plan",
    "master plan": "master_plan", "site layout": "master_plan",
    "location map": "location_map", "location plan": "location_map",
    "amenity": "amenity", "amenities": "amenity",
    "elevation": "exterior", "building view": "exterior",
    "exterior": "exterior", "interior": "interior",
    "diagram": "diagram", "schematic": "diagram", "cross-section": "diagram"
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
            logger.warning("FlagEmbedding reranker failed (%s), trying sentence-transformers CrossEncoder", exc)
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
    domain: str | None = None,
    include_images: bool = False,
    top_k: int = 8,
) -> list[dict]:
    """
    Run hybrid retrieval and return ranked results.

    Parameters
    ----------
    query:          User's natural language question.
    document_id:    Optional — scope search to a specific document.
    project_name:   Optional metadata filter.
    builder:        Optional metadata filter.
    document_type:  Optional metadata filter.
    include_images: Force image retrieval even for text-only queries.
    top_k:          Number of final results to return.

    Returns
    -------
    List of result dicts with content, score, source_type, metadata, etc.
    """
    from retrieval.qdrant_service import (
        search_images,
        search_text_dense,
        search_text_keyword,
    )
    from retrieval.embeddings import get_image_embedder, get_text_embedder

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
    if domain:
        text_filters["domain"] = domain
        image_filters["domain"] = domain

    # Detect intent
    image_intent = detect_image_intent(query)
    should_search_images = include_images or image_intent["requires_image"]

    # ── 1. Dense text search ──────────────────────────────────────────────────
    try:
        query_vec = get_text_embedder().embed(query)
        dense_results = search_text_dense(query_vec, text_filters or None, top_k=top_k * 2)
    except Exception as exc:
        logger.warning("Dense text search failed: %s", exc)
        dense_results = []

    # ── 2. BM25 keyword search ────────────────────────────────────────────────
    try:
        keyword_results = search_text_keyword(query, text_filters or None, top_k=top_k * 2)
    except Exception as exc:
        logger.warning("BM25 keyword search failed: %s", exc)
        keyword_results = []

    # ── 3. Image search ───────────────────────────────────────────────────────
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

    # ── 4. RRF fusion ─────────────────────────────────────────────────────────
    text_fused = _rrf_fuse([dense_results, keyword_results])
    all_results = _merge_text_and_image(text_fused, image_results, image_intent)

    # ── 5. Reranking ──────────────────────────────────────────────────────────
    # Only rerank text results (image results keep their scores)
    text_only = [r for r in all_results if r.get("source_type") == "text"]
    image_only = [r for r in all_results if r.get("source_type") == "image"]

    if len(text_only) > 1:
        text_only = _rerank(query, text_only)

    # Interleave: top text + top images, then slice to top_k
    final = _interleave(text_only, image_only, top_k)

    return final


def detect_image_intent(query: str) -> dict:
    """
    Detect whether a query requires image results.

    Returns:
        {requires_image: bool, detected_types: list[str]}
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
    Result dicts must have an 'id' field.
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
    Falls back to original order if reranker unavailable.
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
    Images are appended after text results; total capped at top_k.
    """
    final: list[dict] = []
    final.extend(text_results[: max(top_k - len(image_results[:3]), top_k // 2)])
    final.extend(image_results[:3])
    return final[:top_k]
