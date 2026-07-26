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
import math
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

# ── Reranker status ────────────────────────────────────────────────────────────
def reranker_status() -> dict:
    from config import VOYAGE_API_KEY, VOYAGE_RERANK_MODEL
    return {
        "loaded": VOYAGE_API_KEY is not None and len(VOYAGE_API_KEY) > 0,
        "model": VOYAGE_RERANK_MODEL
    }

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
    if domain and domain == "excavator":
        text_filters["domain"] = domain
        image_filters["domain"] = domain

    # Resolve dynamic collection names for excavator
    target_text_col = None
    target_img_col = None

    if domain == "excavator":
        target_text_col = "im_manuals_text"
        target_img_col = "im_manuals_images"

        # Apply exact DTC code and component tag pre-filtering from the query router
        try:
            from chatbot.query_router import classify_im_query
            routing_info = classify_im_query(query)
            dtc_list = [c.strip() for c in routing_info.get("dtc_codes", []) if c.strip()]
            comp_list = [c.strip() for c in routing_info.get("components", []) if c.strip()]
            if dtc_list:
                text_filters["dtc_codes"] = dtc_list
                image_filters["dtc_codes"] = dtc_list
            if comp_list:
                text_filters["component_tags"] = comp_list
                image_filters["component_tags"] = comp_list
        except Exception as e:
            logger.warning("Query routing classification failed in retriever: %s", e)

        if document_id:
            try:
                from storage.mongo_client import load_projects
                projects = load_projects(document_id=document_id, domain=domain)
                if projects:
                    meta = projects[0].get("metadata", {})
                    doc_type = meta.get("doc_type")
                    if doc_type == "manuals":
                        target_text_col = "im_manuals_text"
                    elif doc_type == "service_bulletins":
                        target_text_col = "im_service_bulletins"
                    elif doc_type == "maintenance_logs":
                        target_text_col = "im_maintenance_logs"
                    elif doc_type == "parts_catalog":
                        target_text_col = "im_parts_catalog"
                    elif doc_type == "field_reports":
                        target_text_col = "im_field_reports"
            except Exception as e:
                logger.warning("Failed to resolve excavator collection name: %s", e)

    # Detect intent
    image_intent = detect_image_intent(query)
    should_search_images = include_images or image_intent["requires_image"]
    # Type filtering prevents semantically weak full-page/filler images from
    # outranking the diagram, plan, or table the user explicitly requested.
    if image_intent["detected_types"]:
        image_filters["image_type"] = image_intent["detected_types"]

    # 1. Multi-Query Expansion
    queries = expand_query(query)
    logger.info("Expanded query: '%s' into %d variations: %s", query, len(queries), queries)

    all_dense_results = []
    all_keyword_results = []

    # 2. Run Dense + Sparse Search for each query variation (Retrieve Top 50 candidates)
    text_embedder = get_text_embedder()
    has_strict_filters = bool(text_filters.get("dtc_codes") or text_filters.get("component_tags"))
    
    for q_var in queries:
        # A. Dense text search
        try:
            query_vec = text_embedder.embed(q_var, input_type="query")
            dense_res = search_text_dense(query_vec, text_filters or None, top_k=50, collection_name=target_text_col)
            
            # Fallback if strict filter returned nothing
            if not dense_res and has_strict_filters:
                backup_filters = {k: v for k, v in text_filters.items() if k not in ("dtc_codes", "component_tags")}
                dense_res = search_text_dense(query_vec, backup_filters or None, top_k=50, collection_name=target_text_col)
                
            all_dense_results.append(dense_res)
        except Exception as exc:
            logger.warning("Dense text search failed for variation '%s': %s", q_var, exc)

        # B. BM25 keyword search
        try:
            keyword_res = search_text_keyword(q_var, text_filters or None, top_k=50, collection_name=target_text_col)
            
            # Fallback if strict filter returned nothing
            if not keyword_res and has_strict_filters:
                backup_filters = {k: v for k, v in text_filters.items() if k not in ("dtc_codes", "component_tags")}
                keyword_res = search_text_keyword(q_var, backup_filters or None, top_k=50, collection_name=target_text_col)
                
            all_keyword_results.append(keyword_res)
        except Exception as exc:
            logger.warning("BM25 keyword search failed for variation '%s': %s", q_var, exc)

    # 3. Image search
    image_results: list[dict] = []
    if should_search_images:
        try:
            img_embedder = get_image_embedder()
            img_query_vec = img_embedder.embed_text(query, input_type="query")
            image_res = search_images(
                img_query_vec,
                image_filters or None,
                top_k=6,
                collection_name=target_img_col
            )
            
            # Fallback if strict filter returned nothing
            if not image_res and has_strict_filters:
                backup_filters = {k: v for k, v in image_filters.items() if k not in ("dtc_codes", "component_tags")}
                image_res = search_images(
                    img_query_vec,
                    backup_filters or None,
                    top_k=6,
                    collection_name=target_img_col
                )
            # Never return arbitrary page renders alongside a specific visual
            # result. Full pages remain a last-resort option for scanned PDFs.
            if image_intent["detected_types"]:
                specific = [
                    item for item in image_res
                    if item.get("payload", {}).get("image_type") != "full_page"
                ]
                image_results = specific or image_res
            else:
                image_results = image_res
        except Exception as exc:
            logger.warning("Image search failed: %s", exc)

    # 4. RRF fusion across all variations and dense/sparse lists
    all_lists = all_dense_results + all_keyword_results
    text_fused = _rrf_fuse(all_lists)
    
    from config import RERANK_TOP_K, FINAL_TOP_K
    
    # Keep Top RERANK_TOP_K candidates for reranking
    text_fused = text_fused[:RERANK_TOP_K]
    
    all_results = _merge_text_and_image(text_fused, image_results, image_intent)

    # 5. Reranking against the original query
    text_only = [r for r in all_results if r.get("source_type") == "text"]
    image_only = [r for r in all_results if r.get("source_type") == "image"]

    target_top_k = top_k or FINAL_TOP_K

    if len(text_only) > 1:
        text_only = _rerank(query, text_only, top_k=target_top_k)

    # Rerank target_top_k
    text_only = text_only[:target_top_k]

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

    if not confident_text and text_only:
        logger.warning(
            "No text chunks passed the confidence threshold; falling back to top %d text results.",
            top_k,
        )
        confident_text = text_only[:top_k]

    # ── 7. Parent-Child Context Window Expansion ────────────────────────────
    # For top-3 text results, fetch their prev/next chunk neighbours from Qdrant
    # and append them as context-only entries (score=0, source_type=context_window)
    # For image results, fetch parent chunk and nearby context automatically
    try:
        from retrieval.qdrant_service import get_chunks_by_ids
        context_window_chunks: list[dict] = []
        seen_ids: set[str] = {r["id"] for r in confident_text}

        # A. Text chunk context expansion (prev/next neighbours)
        for r in confident_text[:3]:
            meta = r.get("metadata", {})
            neighbour_ids = []
            if meta.get("prev_chunk_id"):
                neighbour_ids.append(meta["prev_chunk_id"])
            if meta.get("next_chunk_id"):
                neighbour_ids.append(meta["next_chunk_id"])

            if neighbour_ids:
                neighbours = get_chunks_by_ids(neighbour_ids, collection_name=target_text_col)
                for n in neighbours:
                    if n["id"] not in seen_ids:
                        seen_ids.add(n["id"])
                        n_payload = n.get("payload", {})
                        context_window_chunks.append({
                            "id": n["id"],
                            "score": 0.0,
                            "source_type": "context_window",
                            "content": n.get("content", ""),
                            "document_id": n_payload.get("document_id", r.get("document_id", "")),
                            "source_file": n_payload.get("source_file", ""),
                            "page_number": n_payload.get("page_number"),
                            "image_path": None,
                            "image_url": None,
                            "image_id": None,
                            "image_type": None,
                            "caption": None,
                            "ocr_used": n_payload.get("ocr_used", False),
                            "citation_id": f"ctx_{n['id'][:8]}",
                            "metadata": n_payload,
                            "confidence_score": 0.0,
                        })

        confident_text = confident_text + context_window_chunks

        # B. Image parent-child context expansion
        # For each retrieved image, fetch its parent chunk and nearby context
        image_context_chunks: list[dict] = []
        for img_result in image_only[:3]:
            img_meta = img_result.get("metadata", {})
            parent_chunk_id = img_meta.get("parent_chunk_id")
            
            if parent_chunk_id and parent_chunk_id not in seen_ids:
                try:
                    # Fetch the parent chunk (page/section containing the image)
                    parent_chunks = get_chunks_by_ids([parent_chunk_id], collection_name=target_text_col)
                    for parent in parent_chunks:
                        if parent["id"] not in seen_ids:
                            seen_ids.add(parent["id"])
                            parent_payload = parent.get("payload", {})
                            
                            # Add parent chunk as context
                            image_context_chunks.append({
                                "id": parent["id"],
                                "score": 0.0,
                                "source_type": "image_parent_context",
                                "content": parent.get("content", ""),
                                "document_id": parent_payload.get("document_id", img_result.get("document_id", "")),
                                "source_file": parent_payload.get("source_file", ""),
                                "page_number": parent_payload.get("page_number"),
                                "image_path": img_result.get("image_path"),
                                "image_url": img_result.get("image_url"),
                                "image_id": img_result.get("image_id"),
                                "image_type": img_result.get("image_type"),
                                "caption": img_result.get("caption"),
                                "ocr_used": parent_payload.get("ocr_used", False),
                                "citation_id": f"img_parent_{parent['id'][:8]}",
                                "metadata": parent_payload,
                                "confidence_score": 0.0,
                                "linked_image_id": img_result.get("image_id"),
                            })
                            
                            # Also fetch parent's neighbours for additional context
                            parent_meta = parent_payload
                            neighbour_ids = []
                            if parent_meta.get("prev_chunk_id"):
                                neighbour_ids.append(parent_meta["prev_chunk_id"])
                            if parent_meta.get("next_chunk_id"):
                                neighbour_ids.append(parent_meta["next_chunk_id"])
                            
                            if neighbour_ids:
                                neighbours = get_chunks_by_ids(neighbour_ids, collection_name=target_text_col)
                                for n in neighbours:
                                    if n["id"] not in seen_ids:
                                        seen_ids.add(n["id"])
                                        n_payload = n.get("payload", {})
                                        image_context_chunks.append({
                                            "id": n["id"],
                                            "score": 0.0,
                                            "source_type": "image_nearby_context",
                                            "content": n.get("content", ""),
                                            "document_id": n_payload.get("document_id", img_result.get("document_id", "")),
                                            "source_file": n_payload.get("source_file", ""),
                                            "page_number": n_payload.get("page_number"),
                                            "image_path": img_result.get("image_path"),
                                            "image_url": img_result.get("image_url"),
                                            "image_id": img_result.get("image_id"),
                                            "image_type": img_result.get("image_type"),
                                            "caption": img_result.get("caption"),
                                            "ocr_used": n_payload.get("ocr_used", False),
                                            "citation_id": f"img_nearby_{n['id'][:8]}",
                                            "metadata": n_payload,
                                            "confidence_score": 0.0,
                                            "linked_image_id": img_result.get("image_id"),
                                        })
                except Exception as parent_exc:
                    logger.warning("Failed to fetch parent context for image %s: %s", 
                                 img_result.get("image_id"), parent_exc)

        # Add image context chunks to the results
        confident_text = confident_text + image_context_chunks
        
        logger.info("Parent-child expansion: added %d text context chunks, %d image parent context chunks", 
                    len(context_window_chunks), len(image_context_chunks))
                    
    except Exception as exc:
        logger.warning("Context window expansion failed: %s", exc)

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

    if re.search(r"\b(?:figure|fig\.?|diagram|drawing|schematic|graphic|illustration)\b", q):
        if "diagram" not in detected_types:
            detected_types.append("diagram")

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


def _rerank(query: str, results: list[dict], top_k: int = 10) -> list[dict]:
    """
    Rerank results using Voyage Hosted Reranker.
    """
    from services.reranker import rerank_sync
    return rerank_sync(query, results, top_k=top_k)


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
