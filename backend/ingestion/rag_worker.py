"""
RAG Background Worker — runs the full indexing pipeline after a PDF is uploaded.

Called from app.py via FastAPI BackgroundTasks so /upload returns immediately.

Pipeline stages:
  1. extracting_text    — PyMuPDF per-page text extraction
  2. extracting_images  — PDF page renders as PNG
  3. chunking           — token-aware text chunking
  4. embedding_text     — bge-m3 dense vectors
  5. embedding_images   — jina-clip-v2 / CLIP vectors
  6. indexing_qdrant    — upsert to Qdrant Cloud
  7. ready              — update status to "ready"
"""

from __future__ import annotations

import logging
import os
import tempfile

logger = logging.getLogger(__name__)


def run_rag_indexing(
    *,
    document_id: str,
    pdf_bytes: bytes,
    source_file: str,
    doc_metadata: dict,
) -> None:
    """
    Full RAG indexing pipeline — runs in background after upload.

    Parameters
    ----------
    document_id:  Unique document identifier (already saved in MongoDB).
    pdf_bytes:    Raw PDF bytes (kept in memory from the upload request).
    source_file:  Original filename for metadata.
    doc_metadata: Dict with project_name, builder, document_type, source_file.
    """
    from storage.doc_status import set_status
    from ingestion.pdf_processor import process_pdf
    from ingestion.chunker import chunk_text_pages
    from ingestion.image_processor import build_image_embedding_text, process_images
    from retrieval.embeddings import get_image_embedder, get_text_embedder
    from retrieval.qdrant_service import (
        create_collections,
        upsert_image_chunks,
        upsert_text_chunks,
    )

    text_chunks_indexed = 0
    images_indexed = 0
    total_pages = 0

    logger.info("=== RAG WORKER START | doc=%s | file=%s ===", document_id, source_file)

    try:
        with tempfile.TemporaryDirectory(prefix=f"rag_{document_id}_") as temp_dir:
            # Write PDF bytes to temp file
            pdf_path = os.path.join(temp_dir, source_file)
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            logger.info("[%s] PDF written to temp: %s (%d bytes)", document_id, pdf_path, len(pdf_bytes))

            image_folder = os.path.join(temp_dir, "images")
            image_base_path = f"documents/{document_id}/images"

            # ── Stage 1: Extract text (also renders page images) ─────────────
            set_status(document_id, "extracting_text")
            processed = process_pdf(
                pdf_path=pdf_path,
                document_id=document_id,
                source_file=source_file,
                output_folder=image_folder,
                image_base_path=image_base_path,
            )
            pages = processed.get("pages", [])
            total_pages = processed.get("total_pages", len(pages))
            full_text_len = len(processed.get("full_text", ""))
            logger.info(
                "[%s] Text extracted: %d pages, %d total chars",
                document_id, total_pages, full_text_len,
            )

            if not pages:
                raise ValueError("PDF produced no pages — file may be empty or corrupt.")

            # ── Stage 2: Images extracted (done inside process_pdf above) ────
            set_status(document_id, "extracting_images", total_pages=total_pages)
            all_image_paths = [
                img.get("local_path")
                for page in pages
                for img in page.get("images", [])
                if img.get("local_path")
            ]
            logger.info("[%s] Page renders available: %d", document_id, len(all_image_paths))

            # ── Stage 3: Chunk text ──────────────────────────────────────────
            set_status(document_id, "chunking", total_pages=total_pages)
            text_chunks = chunk_text_pages(pages, document_id, doc_metadata)
            logger.info("[%s] Text chunks created: %d", document_id, len(text_chunks))

            # ── Stage 4: Embed text ──────────────────────────────────────────
            set_status(document_id, "embedding_text", total_pages=total_pages)
            if text_chunks:
                logger.info("[%s] Loading text embedder (bge-m3)…", document_id)
                text_embedder = get_text_embedder()
                chunk_texts = [c["content"] for c in text_chunks]
                text_vectors = text_embedder.embed_batch(chunk_texts)
                for chunk, vec in zip(text_chunks, text_vectors):
                    chunk["vector"] = vec
                logger.info("[%s] Text vectors generated: %d", document_id, len(text_chunks))
            else:
                logger.warning("[%s] No text chunks to embed — PDF may have no text layer", document_id)

            # ── Stage 5: Embed images ────────────────────────────────────────
            set_status(document_id, "embedding_images", total_pages=total_pages)
            image_records = process_images(pages, document_id, source_file, doc_metadata)
            logger.info("[%s] Image records created: %d", document_id, len(image_records))

            if image_records:
                logger.info("[%s] Loading image embedder (jina-clip-v2)…", document_id)
                img_embedder = get_image_embedder()
                for record in image_records:
                    embed_text = build_image_embedding_text(record)
                    local_path = record.get("local_path")
                    vec = None
                    if local_path and os.path.exists(local_path):
                        try:
                            vec = img_embedder.embed_image_file(local_path)
                        except Exception as img_exc:
                            logger.warning(
                                "[%s] Image embed failed for %s, using text fallback: %s",
                                document_id, local_path, img_exc,
                            )
                    if vec is None:
                        vec = img_embedder.embed_text(embed_text)
                    record["vector"] = vec
                logger.info("[%s] Image vectors generated: %d", document_id, len(image_records))

            # ── Stage 6: Index into Qdrant Cloud ─────────────────────────────
            set_status(document_id, "indexing_qdrant", total_pages=total_pages)
            logger.info("[%s] Connecting to Qdrant Cloud…", document_id)
            create_collections()

            if text_chunks:
                text_chunks_indexed = upsert_text_chunks(text_chunks)
                logger.info("[%s] Qdrant: %d text chunks indexed", document_id, text_chunks_indexed)

            if image_records:
                images_indexed = upsert_image_chunks(image_records)
                logger.info("[%s] Qdrant: %d image records indexed", document_id, images_indexed)

        # temp_dir cleaned up here — all data already in Qdrant

        # ── Stage 7: Ready ───────────────────────────────────────────────────
        set_status(
            document_id,
            "ready",
            text_chunks_indexed=text_chunks_indexed,
            images_indexed=images_indexed,
            total_pages=total_pages,
            ocr_used=processed.get("ocr_used", False),
        )
        logger.info(
            "=== RAG WORKER DONE | doc=%s | pages=%d | text=%d | images=%d ===",
            document_id, total_pages, text_chunks_indexed, images_indexed,
        )

    except Exception as exc:
        logger.error(
            "=== RAG WORKER FAILED | doc=%s | error=%s ===",
            document_id, exc, exc_info=True,
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
            logger.error("Failed to write failed status: %s", inner)
