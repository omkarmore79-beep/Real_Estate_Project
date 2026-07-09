import os
import json
import logging
import tempfile
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def _clean_llm_output(text: str) -> str:
    """Strip ```json fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last ``` lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text

def process_document_background(document_id: str, file_path: str, metadata: dict) -> None:
    """
    Entire backend document ingestion, RAG vector embedding, and Qdrant indexing pipeline.
    Runs asynchronously and persists progress at each stage.
    """
    from storage.doc_status import set_status
    from ingestion.extractor import extract_document
    from ingestion.cleaner import clean_text
    from formatter.llm_formatter import format_with_llm
    from formatter.project_normalizer import normalize_project_data
    from ingestion.image_extractor import extract_images_from_pdf
    from ingestion.image_analyzer import analyze_images, page_metadata_from_images
    from storage.mongo_store import save_file_to_mongo, save_project_to_mongo
    
    from ingestion.pdf_processor import process_pdf
    from ingestion.chunker import chunk_text_pages
    from ingestion.image_processor import build_image_embedding_text, process_images
    from retrieval.embeddings import get_image_embedder, get_text_embedder
    from retrieval.qdrant_service import (
        create_collections,
        upsert_image_chunks,
        upsert_text_chunks,
    )
    
    logger.info("=== BACKGROUND PROCESSOR START | doc=%s | file=%s ===", document_id, file_path)
    
    text_chunks_indexed = 0
    images_indexed = 0
    total_pages = 0

    try:
        source_file = metadata.get("source_file", "document.pdf")
        content_type = metadata.get("content_type", "application/pdf")
        title = metadata.get("title", "")
        builder = metadata.get("builder", "")
        project = metadata.get("project", "")
        document_type = metadata.get("document_type", "")
        description = metadata.get("description", "")
        tags = metadata.get("tags", "")

        # ── 1. Legacy Text Extraction ──────────────────────────────────────────
        set_status(document_id, "extracting_text")
        try:
            raw_text = extract_document(file_path)
            cleaned_text = clean_text(raw_text)
            logger.info("[%s] Text extracted: %d chars", document_id, len(cleaned_text))
        except Exception as exc:
            logger.warning("[%s] Text extraction error: %s", document_id, exc)
            cleaned_text = ""

        # ── 2. LLM metadata formatter ────────────────────────────────────────
        try:
            formatted_output = format_with_llm(cleaned_text)
            cleaned_output = _clean_llm_output(formatted_output)
            json_data = json.loads(cleaned_output)
            logger.info("[%s] LLM formatter succeeded", document_id)
        except Exception as exc:
            logger.warning("[%s] LLM formatter error: %s", document_id, exc)
            json_data = {}

        json_data = normalize_project_data(json_data, cleaned_text)
        
        metadata_project = (project or "").strip()
        metadata_builder = (builder or "").strip()

        if metadata_project:
            json_data["project_name"] = metadata_project
        if metadata_builder:
            json_data["developer"] = metadata_builder

        # ── 3. Legacy Image Extraction ─────────────────────────────────────────
        set_status(document_id, "extracting_images")
        
        # Create temp folder for image extraction
        with tempfile.TemporaryDirectory(prefix=f"proc_{document_id}_") as temp_dir:
            image_folder = os.path.join(temp_dir, "images")
            image_metadata = []
            try:
                extracted_images = extract_images_from_pdf(
                    file_path,
                    output_folder=image_folder,
                    image_base_path=f"documents/{document_id}/images",
                )
                image_metadata = analyze_images(extracted_images, cleaned_text)
                logger.info("[%s] Legacy images extracted: %d", document_id, len(image_metadata))
            except Exception as exc:
                logger.warning("[%s] Legacy image extraction error: %s", document_id, exc)

            # ── 4. Save PDF to MongoDB GridFS ──────────────────────────────────
            try:
                saved_pdf_id = save_file_to_mongo(
                    document_id=document_id,
                    file_path=file_path,
                    filename=source_file,
                    content_type=content_type,
                    file_kind="pdf",
                )
                logger.info("[%s] PDF saved: id=%s", document_id, saved_pdf_id)
            except Exception as exc:
                logger.error("[%s] PDF save failed: %s", document_id, exc)
                saved_pdf_id = None

            # ── 5. Save images to MongoDB GridFS ───────────────────────────────
            for image in image_metadata:
                image_id = image.get("image_id")
                local_path = image.pop("local_path", None)
                if not image_id or not local_path:
                    continue
                try:
                    save_file_to_mongo(
                        document_id=document_id,
                        file_path=local_path,
                        filename=f"{image_id}.png",
                        content_type="image/png",
                        file_kind="image",
                        image_id=image_id,
                    )
                except Exception as exc:
                    logger.warning("[%s] Image save failed for %s: %s", document_id, image_id, exc)
                image["image_path"] = f"documents/{document_id}/images/{image_id}"

            # ── 6. Assemble project document & Save ───────────────────────────
            json_data.update({
                "images": image_metadata,
                "pages": page_metadata_from_images(image_metadata),
                "document_id": document_id,
                "source_file": source_file,
                "stored_file": os.path.basename(file_path),
                "raw_text": cleaned_text,
                "pdf_path": f"documents/{document_id}/pdf",
                "rag_status": "processing",
                "metadata": {
                    "title": title or source_file,
                    "builder": metadata_builder,
                    "project": metadata_project,
                    "document_type": document_type or "",
                    "description": description or "",
                    "tags": [t.strip() for t in (tags or "").split(",") if t.strip()],
                },
            })

            # Save project metadata
            try:
                saved_document = save_project_to_mongo(json_data)
                if saved_document:
                    logger.info("[%s] Project metadata saved successfully", document_id)
            except Exception as exc:
                logger.error("[%s] Project save failed: %s", document_id, exc)

        # ── 7. Multimodal Hybrid RAG vector indexing ─────────────────────────
        # Note: We keep a persistent image folder location for RAG image processing
        rag_images_dir = os.path.join(os.path.dirname(file_path), "rag_images")
        
        # ── Stage 1: Extract text and run OCR where needed ──────────────────────
        set_status(document_id, "extracting_text")
        processed = process_pdf(
            pdf_path=file_path,
            document_id=document_id,
            source_file=source_file,
            output_folder=rag_images_dir,
            image_base_path=f"documents/{document_id}/images",
        )
        pages = processed.get("pages", [])
        total_pages = processed.get("total_pages", len(pages))
        
        # ── Stage 2: Chunk text ──────────────────────────────────────────
        set_status(document_id, "chunking", total_pages=total_pages)
        doc_metadata = {
            "project_name": metadata_project,
            "builder": metadata_builder,
            "document_type": document_type,
            "source_file": source_file,
        }
        text_chunks = chunk_text_pages(pages, document_id, doc_metadata)
        logger.info("[%s] Text chunks created: %d", document_id, len(text_chunks))

        # ── Stage 3: Embed text ──────────────────────────────────────────
        set_status(document_id, "embedding_text", total_pages=total_pages)
        if text_chunks:
            logger.info("[%s] Generating text embeddings...", document_id)
            text_embedder = get_text_embedder()
            chunk_texts = [c["content"] for c in text_chunks]
            text_vectors = text_embedder.embed_batch(chunk_texts)
            for chunk, vec in zip(text_chunks, text_vectors):
                chunk["vector"] = vec
        else:
            logger.warning("[%s] No text chunks to embed", document_id)

        # ── Stage 4: Embed images ────────────────────────────────────────
        set_status(document_id, "embedding_images", total_pages=total_pages)
        image_records = process_images(pages, document_id, source_file, doc_metadata)
        logger.info("[%s] Image records created: %d", document_id, len(image_records))

        if image_records:
            logger.info("[%s] Generating image embeddings...", document_id)
            img_embedder = get_image_embedder()
            for record in image_records:
                embed_text = build_image_embedding_text(record)
                local_path = record.get("local_path")
                vec = None
                if local_path and os.path.exists(local_path):
                    try:
                        vec = img_embedder.embed_image_file(local_path)
                    except Exception as img_exc:
                        logger.warning("[%s] Image embedding failed: %s. Using text fallback.", document_id, img_exc)
                if vec is None:
                    vec = img_embedder.embed_text(embed_text)
                record["vector"] = vec

        # ── Stage 5: Index Qdrant ─────────────────────────────────────────
        set_status(document_id, "indexing_qdrant", total_pages=total_pages)
        create_collections()

        if text_chunks:
            text_chunks_indexed = upsert_text_chunks(text_chunks)
        if image_records:
            images_indexed = upsert_image_chunks(image_records)

        # ── Stage 6: Ready! ──────────────────────────────────────────────
        set_status(
            document_id,
            "ready",
            text_chunks_indexed=text_chunks_indexed,
            images_indexed=images_indexed,
            total_pages=total_pages,
        )
        logger.info("=== BACKGROUND PROCESSOR SUCCESS | doc=%s ===", document_id)
        
    except Exception as e:
        logger.error("=== BACKGROUND PROCESSOR FAILED | doc=%s | error=%s ===", document_id, e, exc_info=True)
        set_status(
            document_id,
            "failed",
            error=str(e)[:400]
        )
