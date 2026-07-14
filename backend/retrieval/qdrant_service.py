"""
Qdrant service layer for the Hybrid Multimodal RAG pipeline.

Provides helper functions to:
  - Create text and image vector collections
  - Upsert / search text chunks (supports dynamic collections)
  - Upsert / search image chunks (supports dynamic collections)
  - Delete vectors by document_id
  - Health check
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from config import (
    IMAGE_VECTOR_DIM,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_IMAGES,
    QDRANT_COLLECTION_TEXT,
    QDRANT_URL,
    TEXT_VECTOR_DIM,
)

logger = logging.getLogger(__name__)

# ── Lazy Qdrant client ─────────────────────────────────────────────────────────
_qdrant_client = None


def get_qdrant_client():
    """Return the shared Qdrant client from qdrant_client."""
    from retrieval.qdrant_client import client
    return client


# ── Collection management ──────────────────────────────────────────────────────

def create_collections() -> dict[str, str]:
    """
    Create text and image Qdrant collections if they do not already exist,
    or recreate them if their vector dimensions do not match target configurations.
    """
    from qdrant_client.models import Distance, VectorParams

    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    results: dict[str, str] = {}

    def verify_and_create(col_name: str, target_dim: int) -> str:
        recreate = False
        if col_name in existing:
            try:
                col_info = client.get_collection(col_name)
                params = col_info.config.params.vectors
                current_size = -1
                if hasattr(params, "size"):
                    current_size = params.size
                elif isinstance(params, dict) and "size" in params:
                    current_size = params["size"]
                
                if current_size != target_dim:
                    logger.warning(
                        "Collection %s exists with dimension %s but target is %d. Recreating...",
                        col_name, str(current_size), target_dim
                    )
                    client.delete_collection(col_name)
                    recreate = True
            except Exception as e:
                logger.warning("Failed to check collection config for %s: %s. Recreating...", col_name, e)
                try:
                    client.delete_collection(col_name)
                except Exception:
                    pass
                recreate = True

        if col_name not in existing or recreate:
            client.create_collection(
                collection_name=col_name,
                vectors_config=VectorParams(size=target_dim, distance=Distance.COSINE),
            )
            logger.info("Created collection: %s (dim=%d)", col_name, target_dim)
            status = "created"
        else:
            status = "exists"
            
        try:
            from qdrant_client.models import PayloadSchemaType
            client.create_payload_index(
                collection_name=col_name,
                field_name="document_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Ensured payload index for document_id exists on collection: %s", col_name)
        except Exception as e:
            logger.warning("Failed to ensure payload index for document_id on %s: %s", col_name, e)
            
        return status

    results[QDRANT_COLLECTION_TEXT] = verify_and_create(QDRANT_COLLECTION_TEXT, TEXT_VECTOR_DIM)
    results[QDRANT_COLLECTION_IMAGES] = verify_and_create(QDRANT_COLLECTION_IMAGES, IMAGE_VECTOR_DIM)

    return results


# ── Text chunks ────────────────────────────────────────────────────────────────

def upsert_text_chunks(chunks: list[dict], collection_name: str | None = None) -> int:
    """
    Upsert a list of text chunk dicts into the target collection.

    Each chunk must have:
      - ``vector``   : list[float] — dense embedding
      - ``content``  : str         — chunk text
      - ``metadata`` : dict        — arbitrary payload fields
      - ``chunk_id`` : str (optional) — stable UUID; generated if missing
    """
    from qdrant_client.models import PointStruct

    if not chunks:
        return 0

    target = collection_name or QDRANT_COLLECTION_TEXT
    client = get_qdrant_client()
    points: list[PointStruct] = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id") or uuid.uuid4().hex
        payload = {
            "content": chunk.get("content", ""),
            **chunk.get("metadata", {}),
        }
        points.append(
            PointStruct(
                id=_str_to_uuid(chunk_id),
                vector=chunk["vector"],
                payload=payload,
            )
        )

    client.upsert(collection_name=target, points=points)
    logger.info("Upserted %d text chunks into Qdrant collection '%s'", len(points), target)
    return len(points)


def upsert_image_chunks(chunks: list[dict], collection_name: str | None = None) -> int:
    """
    Upsert image metadata records into the target image collection.

    Each chunk must have:
      - ``vector``   : list[float] — image/caption embedding
      - ``image_id`` : str
      - ``metadata`` : dict        — image metadata payload
    """
    from qdrant_client.models import PointStruct

    if not chunks:
        return 0

    target = collection_name or QDRANT_COLLECTION_IMAGES
    client = get_qdrant_client()
    points: list[PointStruct] = []

    for chunk in chunks:
        image_id = chunk.get("image_id") or uuid.uuid4().hex
        payload = {
            "image_id": image_id,
            **chunk.get("metadata", {}),
        }
        points.append(
            PointStruct(
                id=_str_to_uuid(image_id),
                vector=chunk["vector"],
                payload=payload,
            )
        )

    client.upsert(collection_name=target, points=points)
    logger.info("Upserted %d image chunks into Qdrant collection '%s'", len(points), target)
    return len(points)


# ── Search ─────────────────────────────────────────────────────────────────────

def search_text_dense(
    query_vector: list[float],
    filters: dict | None = None,
    top_k: int = 10,
    collection_name: str | None = None,
) -> list[dict]:
    """Dense cosine similarity search over the target collection."""
    from qdrant_client.models import Filter

    target = collection_name or QDRANT_COLLECTION_TEXT
    client = get_qdrant_client()
    qdrant_filter = _build_filter(filters)

    try:
        # Modern Qdrant client query interface
        res = client.query_points(
            collection_name=target,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
        )
        results = res.points
    except (AttributeError, TypeError):
        # Legacy fallback
        results = client.search(
            collection_name=target,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
    return _format_results(results)


def search_text_keyword(
    query_text: str,
    filters: dict | None = None,
    top_k: int = 10,
    collection_name: str | None = None,
) -> list[dict]:
    """
    Keyword-based search over the target collection using Qdrant payload scroll
    + BM25 re-scoring over the returned payloads.
    """
    from rank_bm25 import BM25Okapi

    target = collection_name or QDRANT_COLLECTION_TEXT
    client = get_qdrant_client()
    qdrant_filter = _build_filter(filters)

    scroll_result, _ = client.scroll(
        collection_name=target,
        scroll_filter=qdrant_filter,
        limit=500,
        with_payload=True,
        with_vectors=False,
    )

    if not scroll_result:
        return []

    corpus = [_point_text(point) for point in scroll_result]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_text.lower().split())

    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for idx, score in scored:
        if score <= 0:
            continue
        point = scroll_result[idx]
        payload = dict(point.payload or {})
        results.append(
            {
                "id": str(point.id),
                "score": float(score),
                "content": payload.get("content", ""),
                "payload": payload,
            }
        )
    return results


def search_images(
    query_vector: list[float],
    filters: dict | None = None,
    top_k: int = 6,
    collection_name: str | None = None,
) -> list[dict]:
    """Dense cosine similarity search over the image collection."""
    target = collection_name or QDRANT_COLLECTION_IMAGES
    client = get_qdrant_client()
    qdrant_filter = _build_filter(filters)

    try:
        # Modern Qdrant client query interface
        res = client.query_points(
            collection_name=target,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
        )
        results = res.points
    except (AttributeError, TypeError):
        # Legacy fallback
        results = client.search(
            collection_name=target,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
    return _format_results(results)


# ── Delete ─────────────────────────────────────────────────────────────────────

def delete_vectors_by_document_id(
    document_id: str,
    collection_names: list[str] | None = None,
) -> dict[str, int]:
    """Delete all Qdrant vectors associated with a document_id across specified collections."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = get_qdrant_client()
    doc_filter = Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
    )

    targets = collection_names or [QDRANT_COLLECTION_TEXT, QDRANT_COLLECTION_IMAGES]
    results = {}

    for col in targets:
        try:
            delete_res = client.delete(
                collection_name=col,
                points_selector=doc_filter,
            )
            results[f"{col}_deleted"] = 1
        except Exception as exc:
            logger.warning("Failed to delete points from collection %s: %s", col, exc)
            results[f"{col}_deleted"] = 0

    logger.info("Deleted Qdrant vectors for document_id=%s across %s: %s", document_id, targets, results)
    return results


# ── Health check ───────────────────────────────────────────────────────────────

def health_check() -> dict[str, Any]:
    """Return Qdrant connection + collection stats."""
    is_cloud = "cloud.qdrant.io" in QDRANT_URL
    try:
        client = get_qdrant_client()
        collections = {c.name: c for c in client.get_collections().collections}

        text_info = None
        image_info = None

        if QDRANT_COLLECTION_TEXT in collections:
            info = client.get_collection(QDRANT_COLLECTION_TEXT)
            text_info = {
                "status": str(getattr(info, "status", "unknown")),
                "vectors_count": getattr(info, "vectors_count", None),
                "points_count": getattr(info, "points_count", None),
            }

        if QDRANT_COLLECTION_IMAGES in collections:
            info = client.get_collection(QDRANT_COLLECTION_IMAGES)
            image_info = {
                "status": str(getattr(info, "status", "unknown")),
                "vectors_count": getattr(info, "vectors_count", None),
                "points_count": getattr(info, "points_count", None),
            }

        return {
            "qdrant_connected": True,
            "qdrant_url": QDRANT_URL,
            "qdrant_cloud": is_cloud,
            "api_key_configured": bool(QDRANT_API_KEY),
            "text_collection": text_info or "not_found",
            "image_collection": image_info or "not_found",
        }
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        return {
            "qdrant_connected": False,
            "qdrant_url": QDRANT_URL,
            "qdrant_cloud": is_cloud,
            "api_key_configured": bool(QDRANT_API_KEY),
            "error": str(exc),
        }


# ── Private helpers ────────────────────────────────────────────────────────────

def _str_to_uuid(value: str) -> str:
    """Convert an arbitrary string to a stable UUID string for Qdrant point IDs."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, value))


def _build_filter(filters: dict | None):
    """Convert a filter dict supporting lists (e.g. for dtc_codes / tags) to a Qdrant Filter object."""
    if not filters:
        return None

    from qdrant_client.models import FieldCondition, Filter, MatchValue, MatchAny

    conditions = []
    for k, v in filters.items():
        if v is None:
            continue
        if isinstance(v, list):
            # Support matching any values in the payload array
            conditions.append(FieldCondition(key=k, match=MatchAny(any=v)))
        else:
            conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))

    return Filter(must=conditions) if conditions else None


def _point_text(point) -> str:
    payload = dict(point.payload or {})
    return " ".join(str(v) for v in payload.values() if isinstance(v, str))


def _format_results(results) -> list[dict]:
    formatted = []
    for result in results:
        payload = dict(result.payload or {})
        formatted.append(
            {
                "id": str(result.id),
                "score": float(result.score),
                "content": payload.get("content", ""),
                "payload": payload,
            }
        )
    return formatted
