"""
Optimized RAG Background Worker — async pipeline with batching, image compression,
and Qdrant indexing-threshold management.

Pipeline stages:
  1. extracting_text    — PyMuPDF per-page text + table extraction
  2. extracting_images  — embedded images / page renders
  3. chunking           — sliding-window (zero API calls)
  4. embedding_text     — voyage-3 in batches of 128 (1M TPM)
  5. embedding_images   — voyage-multimodal-3.5, compressed JPEG ≤ 768px
  6. indexing_qdrant    — batched upsert (100/batch), indexing disabled during write
  7. ready              — restore indexing, update status
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time

logger = logging.getLogger(__name__)

# ── Image compression settings ─────────────────────────────────────────────────
_MAX_IMAGE_SIDE  = 768   # px — max width or height before Voyage embedding
_JPEG_QUALITY    = 85    # JPEG compression quality (85 is virtually lossless for embeddings)
_MIN_IMAGE_SIDE  = 120   # px — skip images smaller than this

# ── Qdrant batch size ──────────────────────────────────────────────────────────
_QDRANT_BATCH_SIZE = 100


# ── Image compression helper ───────────────────────────────────────────────────

def _compress_image(src_path: str, dst_path: str) -> bool:
    """
    Open an image, resize so the longest side ≤ _MAX_IMAGE_SIDE, save as JPEG.
    Returns True on success, False if the image is too small to embed.
    """
    try:
        from PIL import Image
        img = Image.open(src_path).convert("RGB")
        w, h = img.size

        if w < _MIN_IMAGE_SIDE or h < _MIN_IMAGE_SIDE:
            return False  # Skip tiny images

        if max(w, h) > _MAX_IMAGE_SIDE:
            ratio = _MAX_IMAGE_SIDE / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        img.save(dst_path, "JPEG", quality=_JPEG_QUALITY, optimize=True)
        return True
    except Exception as exc:
        logger.warning("Image compression failed for %s: %s", src_path, exc)
        return False


# ── Main worker ────────────────────────────────────────────────────────────────

def run_rag_indexing(
    *,
    document_id: str,
    pdf_bytes: bytes,
    source_file: str,
    doc_metadata: dict,
) -> None:
    """
    Full RAG indexing pipeline — runs in FastAPI BackgroundTask thread.
    Internally uses asyncio.run() for I/O concurrency.
    """
    try:
        asyncio.run(
            _run_pipeline(
                document_id=document_id,
                pdf_bytes=pdf_bytes,
                source_file=source_file,
                doc_metadata=doc_metadata,
            )
        )
    except Exception as exc:
        # asyncio.run already logged inside; catch here to prevent thread crash
        logger.error("[%s] Pipeline runner exited with error: %s", document_id, exc)


async def _run_pipeline(
    *,
    document_id: str,
    pdf_bytes: bytes,
    source_file: str,
    doc_metadata: dict,
) -> None:
    """Async implementation of the full RAG pipeline."""
    from storage.doc_status import set_status
    from ingestion.pdf_processor import process_pdf
    from ingestion.chunker import chunk_text_pages
    from ingestion.image_processor import build_image_embedding_text, process_images
    from retrieval.embeddings import get_image_embedder, get_text_embedder
    from retrieval.qdrant_service import (
        create_collections_async,
        upsert_text_chunks_batched_async,
        upsert_image_chunks_batched_async,
        set_indexing_threshold_async,
    )

    text_chunks_indexed = 0
    images_indexed = 0
    total_pages = 0
    t_start = time.perf_counter()

    logger.info("=== RAG WORKER START | doc=%s | file=%s ===", document_id, source_file)

    try:
        with tempfile.TemporaryDirectory(prefix=f"rag_{document_id}_") as temp_dir:
            # ── Write PDF to disk ──────────────────────────────────────────────
            pdf_path = os.path.join(temp_dir, source_file)
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            logger.info("[%s] PDF written (%d bytes)", document_id, len(pdf_bytes))

            image_folder = os.path.join(temp_dir, "images")
            compressed_folder = os.path.join(temp_dir, "compressed")
            os.makedirs(compressed_folder, exist_ok=True)
            image_base_path = f"documents/{document_id}/images"

            # ── Stage 1: Extract text ─────────────────────────────────────────
            set_status(document_id, "extracting_text")
            # Run CPU-bound PDF parsing in a thread so we don't block the event loop
            processed = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: process_pdf(
                    pdf_path=pdf_path,
                    document_id=document_id,
                    source_file=source_file,
                    output_folder=image_folder,
                    image_base_path=image_base_path,
                ),
            )
            pages = processed.get("pages", [])
            total_pages = processed.get("total_pages", len(pages))
            logger.info("[%s] Text extracted: %d pages", document_id, total_pages)

            if not pages:
                raise ValueError("PDF produced no pages — file may be empty or corrupt.")

            # ── Stage 2: Collect image paths ───────────────────────────────────
            set_status(document_id, "extracting_images", total_pages=total_pages)
            all_image_paths = [
                img.get("local_path")
                for page in pages
                for img in page.get("images", [])
                if img.get("local_path")
            ]
            logger.info("[%s] Raw image files: %d", document_id, len(all_image_paths))

            # ── Stage 3: Chunk text ────────────────────────────────────────────
            set_status(document_id, "chunking", total_pages=total_pages)
            text_chunks = chunk_text_pages(pages, document_id, doc_metadata)
            logger.info("[%s] Text chunks: %d", document_id, len(text_chunks))

            # ── Stage 4 + 5: Embed text AND compress+embed images concurrently ─
            set_status(document_id, "embedding_text", total_pages=total_pages)
            image_records = process_images(pages, document_id, source_file, doc_metadata)

            text_task = asyncio.get_event_loop().run_in_executor(
                None, _embed_texts, text_chunks
            )
            image_task = asyncio.get_event_loop().run_in_executor(
                None, _compress_and_embed_images, image_records, compressed_folder
            )

            # Wait for both to finish simultaneously
            await asyncio.gather(text_task, image_task)

            logger.info(
                "[%s] Embedding complete — text=%d vectors, images=%d vectors",
                document_id,
                sum(1 for c in text_chunks if c.get("vector")),
                sum(1 for r in image_records if r.get("vector")),
            )

            # ── Stage 6: Index into Qdrant ─────────────────────────────────────
            set_status(document_id, "indexing_qdrant", total_pages=total_pages)
            logger.info("[%s] Creating Qdrant collections...", document_id)
            await create_collections_async()

            # Disable HNSW indexing during bulk write for maximum write throughput
            await set_indexing_threshold_async(threshold=0)

            try:
                if text_chunks:
                    text_chunks_indexed = await upsert_text_chunks_batched_async(
                        text_chunks, batch_size=_QDRANT_BATCH_SIZE
                    )
                    logger.info("[%s] Qdrant: %d text chunks indexed", document_id, text_chunks_indexed)

                if image_records:
                    images_indexed = await upsert_image_chunks_batched_async(
                        image_records, batch_size=_QDRANT_BATCH_SIZE
                    )
                    logger.info("[%s] Qdrant: %d image chunks indexed", document_id, images_indexed)
            finally:
                # Always restore indexing, even on error
                await set_indexing_threshold_async(threshold=20_000)

        # temp_dir cleaned up; all data is in Qdrant

        elapsed = time.perf_counter() - t_start
        logger.info(
            "=== RAG WORKER DONE | doc=%s | pages=%d | text=%d | images=%d | elapsed=%.1fs ===",
            document_id, total_pages, text_chunks_indexed, images_indexed, elapsed,
        )

        set_status(
            document_id,
            "ready",
            text_chunks_indexed=text_chunks_indexed,
            images_indexed=images_indexed,
            total_pages=total_pages,
            ocr_used=processed.get("ocr_used", False),
        )

    except Exception as exc:
        elapsed = time.perf_counter() - t_start
        logger.error(
            "=== RAG WORKER FAILED | doc=%s | elapsed=%.1fs | error=%s ===",
            document_id, elapsed, exc, exc_info=True,
        )
        try:
            set_status(
                document_id,
                "failed",
                error=str(exc)[:400],
                text_chunks_indexed=text_chunks_indexed,
                images_indexed=images_indexed,
                total_pages=total_pages,
            )
        except Exception as inner:
            logger.error("[%s] Failed to write failed status: %s", document_id, inner)


# ── Sync helpers (run in executor threads) ────────────────────────────────────

def _embed_texts(text_chunks: list[dict]) -> None:
    """Embed all text chunks in batches of 128. Mutates chunk['vector'] in-place."""
    if not text_chunks:
        return
    from retrieval.embeddings import get_text_embedder
    t0 = time.perf_counter()
    embedder = get_text_embedder()
    texts = [c["content"] for c in text_chunks]
    vectors = embedder.embed_batch(texts, batch_size=128)
    for chunk, vec in zip(text_chunks, vectors):
        chunk["vector"] = vec
    logger.info(
        "Text embedding: %d chunks in %.2fs (batch_size=128)",
        len(text_chunks), time.perf_counter() - t0,
    )


def _compress_and_embed_images(image_records: list[dict], compressed_folder: str) -> None:
    """
    For each image record:
      1. Compress PNG → JPEG ≤ 768px
      2. Embed the compressed file via Voyage multimodal API
      3. Fall back to text-only embedding if image fails
    Mutates record['vector'] in-place.
    """
    if not image_records:
        return
    from retrieval.embeddings import get_image_embedder
    from ingestion.image_processor import compress_image_for_voyage
    img_embedder = get_image_embedder()
    t0 = time.perf_counter()
    embedded = 0
    text_fallback = 0
    valid_records: list[dict] = []
    compressed_paths: list[str] = []
    for record in image_records:
        local_path = record.get("local_path")
        compressed_path = compress_image_for_voyage(local_path, compressed_folder) if local_path else None
        if compressed_path:
            valid_records.append(record)
            compressed_paths.append(compressed_path)

    vectors: list[list[float]] = []
    if compressed_paths:
        try:
            vectors = img_embedder.embed_image_paths(compressed_paths, batch_size=8)
        except Exception as exc:
            logger.warning("Batched image embedding failed: %s", exc)
    vector_by_record = {id(record): vector for record, vector in zip(valid_records, vectors)}
    embedded = len(vector_by_record)
    for record in image_records:
        vec = vector_by_record.get(id(record))
        if vec is None:
            try:
                vec = img_embedder.embed_text(_build_embed_text(record))
                text_fallback += 1
            except Exception as exc:
                logger.warning("Image text-fallback embed failed: %s", exc)
                vec = []
        record["vector"] = vec or []

    logger.info(
        "Image embedding: %d images (visual=%d, text_fallback=%d) in %.2fs",
        len(image_records), embedded, text_fallback, time.perf_counter() - t0,
    )


def _build_embed_text(record: dict) -> str:
    """Build embedding text for an image record (caption + type + context)."""
    parts = [
        record.get("caption", ""),
        (record.get("image_type") or "").replace("_", " "),
        (record.get("nearby_page_text") or "")[:300],
    ]
    return " ".join(p for p in parts if p).strip()
