"""
Qdrant service layer for the Hybrid Multimodal RAG pipeline.

Provides helper functions to:
  - Create text and image vector collections (with indexing threshold control)
  - Upsert text / image chunks in batches of N (default 100)
  - Disable/restore HNSW indexing during bulk ingestion for maximum write speed
  - Search text (dense + BM25) and images
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
from logging_and_resilience import LogContext, RetryConfig, retry

logger = logging.getLogger(__name__)


# ── Qdrant client ──────────────────────────────────────────────────────────────

def get_qdrant_client():
    """Return the shared Qdrant client (prefer_grpc=True for faster bulk writes)."""
    from retrieval.qdrant_client import client
    return client


def get_async_qdrant_client():
    """Return the shared AsyncQdrantClient for high-throughput ingestion writes."""
    from retrieval.qdrant_client import async_client
    return async_client


# ── Collection management ──────────────────────────────────────────────────────

@retry(RetryConfig(max_attempts=3, initial_delay_ms=100, max_delay_ms=1000))
def create_collections() -> dict[str, str]:
    """
    Create text and image Qdrant collections if they do not already exist,
    or recreate them if their vector dimensions do not match target configurations.
    """
    from qdrant_client.models import Distance, VectorParams

    with LogContext(stage="qdrant.create_collections"):
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
                        "Collection %s exists with dim=%s but target=%d. Recreating...",
                        col_name, current_size, target_dim,
                    )
                    client.delete_collection(col_name)
                    recreate = True
            except Exception as e:
                logger.warning("Failed to check collection %s: %s. Recreating...", col_name, e)
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

        # Ensure payload index for fast document_id filtering
        try:
            from qdrant_client.models import PayloadSchemaType
            client.create_payload_index(
                collection_name=col_name,
                field_name="document_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            logger.debug("Payload index for document_id on %s: %s", col_name, e)

        return status

    results[QDRANT_COLLECTION_TEXT] = verify_and_create(QDRANT_COLLECTION_TEXT, TEXT_VECTOR_DIM)
    results[QDRANT_COLLECTION_IMAGES] = verify_and_create(QDRANT_COLLECTION_IMAGES, IMAGE_VECTOR_DIM)
    return results


# ── Indexing threshold control ─────────────────────────────────────────────────

@retry(RetryConfig(max_attempts=3, initial_delay_ms=100, max_delay_ms=1000))
def set_indexing_threshold(
    threshold: int,
    collection_names: list[str] | None = None,
) -> None:
    """
    Set the HNSW indexing threshold on all target collections.

    Set threshold=0  to DISABLE indexing during bulk ingestion (maximum write speed).
    Set threshold=20000 to RE-ENABLE indexing after ingestion completes.

    Qdrant will build/update the HNSW index lazily once the threshold is restored.
    """
    from qdrant_client.models import OptimizersConfigDiff

    with LogContext(stage="qdrant.set_indexing_threshold"):
        client = get_qdrant_client()
        targets = collection_names or [QDRANT_COLLECTION_TEXT, QDRANT_COLLECTION_IMAGES]

        for col in targets:
            try:
                client.update_collection(
                    collection_name=col,
                    optimizer_config=OptimizersConfigDiff(indexing_threshold=threshold),
                )
                if threshold == 0:
                    logger.info("HNSW indexing DISABLED on collection '%s' for bulk write.", col)
                else:
                    logger.info("HNSW indexing RESTORED (threshold=%d) on collection '%s'.", threshold, col)
            except Exception as exc:
                logger.warning("Could not set indexing_threshold=%d on %s: %s", threshold, col, exc)


async def set_indexing_threshold_async(
    threshold: int,
    collection_names: list[str] | None = None,
) -> None:
    """Async counterpart used by the background ingestion pipeline."""
    from qdrant_client.models import OptimizersConfigDiff

    with LogContext(stage="qdrant.set_indexing_threshold_async"):
        client = get_async_qdrant_client()
        targets = collection_names or [QDRANT_COLLECTION_TEXT, QDRANT_COLLECTION_IMAGES]
        for col in targets:
            try:
                await client.update_collection(
                    collection_name=col,
                    optimizer_config=OptimizersConfigDiff(indexing_threshold=threshold),
                )
                logger.info("Async indexing threshold=%d applied to %s", threshold, col)
            except Exception as exc:
                logger.warning("Could not set async indexing_threshold=%d on %s: %s", threshold, col, exc)


async def create_collections_async() -> dict[str, str]:
    """Create target collections through AsyncQdrantClient for bulk ingestion."""
    from qdrant_client.models import Distance, OptimizersConfigDiff, PayloadSchemaType, VectorParams

    with LogContext(stage="qdrant.create_collections_async"):
        client = get_async_qdrant_client()
        results: dict[str, str] = {}
        for name, dimension in ((QDRANT_COLLECTION_TEXT, TEXT_VECTOR_DIM), (QDRANT_COLLECTION_IMAGES, IMAGE_VECTOR_DIM)):
            status = "exists"
            try:
                info = await client.get_collection(collection_name=name)
                vectors = info.config.params.vectors
                current = getattr(vectors, "size", None)
                if isinstance(vectors, dict):
                    current = vectors.get("size")
                if current != dimension:
                    await client.delete_collection(collection_name=name)
                    raise RuntimeError("dimension mismatch")
            except Exception:
                await client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
                    optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
                )
                status = "created"
            try:
                await client.create_payload_index(
                    collection_name=name,
                    field_name="document_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass
            results[name] = status
    return results


# ── Batched text upsert ────────────────────────────────────────────────────────

@retry(RetryConfig(max_attempts=3, initial_delay_ms=100, max_delay_ms=1000))
def upsert_text_chunks_batched(
    chunks: list[dict],
    collection_name: str | None = None,
    batch_size: int = 100,
) -> int:
    """
    Upsert text chunk dicts in batches of `batch_size`.
    Logs progress every batch.
    Returns total number of points upserted.
    """
    from qdrant_client.models import PointStruct

    with LogContext(stage="qdrant.upsert_text_chunks"):
        if not chunks:
            return 0

        target = collection_name or QDRANT_COLLECTION_TEXT
        client = get_qdrant_client()
        total = 0
        num_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch = chunks[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        points: list[PointStruct] = []

        for chunk in batch:
            vec = chunk.get("vector")
            if not vec:
                continue  # Skip chunks with no vector (embedding failed)
            chunk_id = chunk.get("chunk_id") or uuid.uuid4().hex
            payload = {
                "content": chunk.get("content", ""),
                **chunk.get("metadata", {}),
            }
            points.append(
                PointStruct(
                    id=_str_to_uuid(chunk_id),
                    vector=vec,
                    payload=payload,
                )
            )

        if points:
            client.upsert(collection_name=target, points=points)
            total += len(points)
            logger.info(
                "Qdrant text upsert: batch %d/%d — %d points (total=%d) → '%s'",
                batch_idx + 1, num_batches, len(points), total, target,
            )

    return total


# ── Batched image upsert ───────────────────────────────────────────────────────

@retry(RetryConfig(max_attempts=3, initial_delay_ms=100, max_delay_ms=1000))
def upsert_image_chunks_batched(
    chunks: list[dict],
    collection_name: str | None = None,
    batch_size: int = 100,
) -> int:
    """
    Upsert image metadata records in batches of `batch_size`.
    Returns total number of points upserted.
    """
    from qdrant_client.models import PointStruct

    with LogContext(stage="qdrant.upsert_image_chunks"):
        if not chunks:
            return 0

        target = collection_name or QDRANT_COLLECTION_IMAGES
        client = get_qdrant_client()
        total = 0
        num_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch = chunks[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        points: list[PointStruct] = []

        for chunk in batch:
            vec = chunk.get("vector")
            if not vec:
                continue
            image_id = chunk.get("image_id") or uuid.uuid4().hex
            payload = {
                "image_id": image_id,
                **chunk.get("metadata", {}),
            }
            points.append(
                PointStruct(
                    id=_str_to_uuid(image_id),
                    vector=vec,
                    payload=payload,
                )
            )

        if points:
            client.upsert(collection_name=target, points=points)
            total += len(points)
            logger.info(
                "Qdrant image upsert: batch %d/%d — %d points (total=%d) → '%s'",
                batch_idx + 1, num_batches, len(points), total, target,
            )

    return total


async def upsert_text_chunks_batched_async(
    chunks: list[dict],
    collection_name: str | None = None,
    batch_size: int = 100,
) -> int:
    """Async batched text upsert using AsyncQdrantClient."""
    from qdrant_client.models import PointStruct

    with LogContext(stage="qdrant.upsert_text_chunks_async"):
        if not chunks:
            return 0

        target = collection_name or QDRANT_COLLECTION_TEXT
        async_client = get_async_qdrant_client()
        total = 0
        num_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch = chunks[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        points: list[PointStruct] = []
        for chunk in batch:
            vec = chunk.get("vector")
            if not vec:
                continue
            chunk_id = chunk.get("chunk_id") or uuid.uuid4().hex
            payload = {
                "content": chunk.get("content", ""),
                **chunk.get("metadata", {}),
            }
            points.append(
                PointStruct(
                    id=_str_to_uuid(chunk_id),
                    vector=vec,
                    payload=payload,
                )
            )

        if points:
            await async_client.upsert(collection_name=target, points=points)
            total += len(points)
            logger.info(
                "Async Qdrant text upsert: batch %d/%d — %d points (total=%d) → '%s'",
                batch_idx + 1, num_batches, len(points), total, target,
            )

    return total


async def upsert_image_chunks_batched_async(
    chunks: list[dict],
    collection_name: str | None = None,
    batch_size: int = 100,
) -> int:
    """Async batched image upsert using AsyncQdrantClient."""
    from qdrant_client.models import PointStruct

    with LogContext(stage="qdrant.upsert_image_chunks_async"):
        if not chunks:
            return 0

        target = collection_name or QDRANT_COLLECTION_IMAGES
        async_client = get_async_qdrant_client()
        total = 0
        num_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch = chunks[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        points: list[PointStruct] = []
        for chunk in batch:
            vec = chunk.get("vector")
            if not vec:
                continue
            image_id = chunk.get("image_id") or uuid.uuid4().hex
            payload = {
                "image_id": image_id,
                **chunk.get("metadata", {}),
            }
            points.append(
                PointStruct(
                    id=_str_to_uuid(image_id),
                    vector=vec,
                    payload=payload,
                )
            )

        if points:
            await async_client.upsert(collection_name=target, points=points)
            total += len(points)
            logger.info(
                "Async Qdrant image upsert: batch %d/%d — %d points (total=%d) → '%s'",
                batch_idx + 1, num_batches, len(points), total, target,
            )

    return total


# ── Legacy single-call upsert (kept for compatibility) ────────────────────────

def upsert_text_chunks(chunks: list[dict], collection_name: str | None = None) -> int:
    """Legacy wrapper — delegates to batched upsert."""
    return upsert_text_chunks_batched(chunks, collection_name=collection_name)


def upsert_image_chunks(chunks: list[dict], collection_name: str | None = None) -> int:
    """Legacy wrapper — delegates to batched upsert."""
    return upsert_image_chunks_batched(chunks, collection_name=collection_name)


# ── Search ─────────────────────────────────────────────────────────────────────

@retry(RetryConfig(max_attempts=3, initial_delay_ms=100, max_delay_ms=500))
def search_text_dense(
    query_vector: list[float],
    filters: dict | None = None,
    top_k: int = 10,
    collection_name: str | None = None,
) -> list[dict]:
    """Dense cosine similarity search over the target collection."""
    target = collection_name or QDRANT_COLLECTION_TEXT
    client = get_qdrant_client()
    qdrant_filter = _build_filter(filters)

    with LogContext(stage="qdrant.search_text_dense"):
        try:
            res = client.query_points(
                collection_name=target,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
            results = res.points
        except (AttributeError, TypeError):
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
    Keyword-based BM25 search over the target collection.
    Scrolls payload and applies BM25 scoring client-side.
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


@retry(RetryConfig(max_attempts=3, initial_delay_ms=100, max_delay_ms=500))
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

    with LogContext(stage="qdrant.search_images"):
        try:
            res = client.query_points(
                collection_name=target,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
            results = res.points
        except (AttributeError, TypeError):
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
            client.delete(collection_name=col, points_selector=doc_filter)
            results[f"{col}_deleted"] = 1
            logger.info("Deleted vectors for document_id=%s from '%s'", document_id, col)
        except Exception as exc:
            logger.warning("Failed to delete from collection %s: %s", col, exc)
            results[f"{col}_deleted"] = 0

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
    """Convert a filter dict to a Qdrant Filter object."""
    if not filters:
        return None

    from qdrant_client.models import FieldCondition, Filter, MatchValue, MatchAny, Range

    conditions = []
    for k, v in filters.items():
        if v is None:
            continue
        # Public search API uses these ergonomic hard-constraint names.  Do
        # not turn them into equality matches (which silently loses results).
        if k in {"max_price", "price_lte"}:
            conditions.append(FieldCondition(key="price", range=Range(lte=float(v))))
            continue
        if k in {"min_price", "price_gte"}:
            conditions.append(FieldCondition(key="price", range=Range(gte=float(v))))
            continue
        if k in {"min_bedrooms", "bedrooms_gte"}:
            conditions.append(FieldCondition(key="bedrooms", range=Range(gte=float(v))))
            continue
        if isinstance(v, list):
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


def get_chunks_by_ids(
    chunk_ids: list[str],
    collection_name: str | None = None,
) -> list[dict]:
    """
    Fetch Qdrant points by their deterministic UUIDs (converted from chunk_id strings).
    Used for parent-child context window expansion in the hybrid retriever.

    Returns a list of {id, content, payload} dicts for each found point.
    Missing points are silently skipped.
    """
    if not chunk_ids:
        return []

    col = collection_name or QDRANT_COLLECTION_TEXT
    client = get_qdrant_client()

    # Convert string chunk_ids to the same deterministic UUIDs used during upsert
    uuids = [_to_uuid(cid) for cid in chunk_ids]

    try:
        points = client.retrieve(
            collection_name=col,
            ids=uuids,
            with_payload=True,
            with_vectors=False,
        )
        result = []
        for pt in points:
            payload = dict(pt.payload or {})
            result.append({
                "id": str(pt.id),
                "content": payload.get("content", ""),
                "payload": payload,
                "score": 0.0,
            })
        return result
    except Exception as exc:
        logger.warning("get_chunks_by_ids failed for collection=%s: %s", col, exc)
        return []
