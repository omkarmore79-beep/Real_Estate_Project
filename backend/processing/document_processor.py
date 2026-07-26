import os
import json
import logging
import tempfile
import hashlib
from datetime import datetime, timezone
import fitz # PyMuPDF

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

def validate_and_hash_file(file_path: str) -> str:
    """
    Validate file uploads for security and compute hash.
    Rejects Executable files, Oversized files, and Encrypted/Corrupted PDFs.
    """
    # 1. Size Check (50MB Limit)
    max_bytes = 50 * 1024 * 1024
    file_size = os.path.getsize(file_path)
    if file_size > max_bytes:
        raise ValueError(f"File size exceeds 50MB limit: {file_size} bytes")
    if file_size == 0:
        raise ValueError("Uploaded file is empty.")

    # 2. Executable / Malicious Check (MZ or ELF magic bytes)
    with open(file_path, "rb") as f:
        magic_bytes = f.read(4)
        if magic_bytes.startswith(b"MZ") or magic_bytes.startswith(b"\x7fELF"):
            raise ValueError("Executable files are not permitted for security reasons.")

    # 3. PDF validation
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        try:
            doc = fitz.open(file_path)
            if doc.is_encrypted:
                doc.close()
                raise ValueError("Encrypted PDFs are not supported.")
            if doc.page_count == 0:
                doc.close()
                raise ValueError("PDF has no pages.")
            doc.close()
        except Exception as exc:
            if "encrypted" in str(exc).lower():
                raise ValueError("Encrypted PDFs are not supported.")
            raise ValueError(f"Corrupted or invalid PDF file: {exc}")

    # 4. Compute SHA-256 Hash
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def process_document_background(document_id: str, file_path: str, metadata: dict) -> None:
    """
    Entire backend document ingestion, RAG vector embedding, and Qdrant indexing pipeline.
    Runs asynchronously and persists progress at each stage.
    """
    from storage.doc_status import set_status
    from ingestion.cleaner import clean_text
    from formatter.llm_formatter import format_with_llm
    from formatter.project_normalizer import normalize_project_data
    from storage.mongo_store import (
        save_file_to_mongo, 
        save_project_to_mongo, 
        _get_collection, 
        _load_projects_locally
    )
    
    from ingestion.multiformat_parser import parse_document
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

        # Excavator specific fields
        domain = metadata.get("domain", "real_estate")
        doc_type = metadata.get("doc_type", "")
        revision_date = metadata.get("revision_date", "")
        section_path = metadata.get("section_path", "")
        component_tags = metadata.get("component_tags", "")
        dtc_codes = metadata.get("dtc_codes", "")
        supersedes_doc_id = metadata.get("supersedes_doc_id", "")

        # ── 0. Security Validation & Hash computation ───────────────────────────
        set_status(document_id, "uploaded", message="Validating file security...")
        file_hash = validate_and_hash_file(file_path)
        logger.info("[%s] File hash: %s", document_id, file_hash)

        # ── 0.5 Deduplication check ──────────────────────────────────────────────
        set_status(document_id, "uploaded", message="Checking for duplicates...")
        existing_doc = None
        
        # Check MongoDB
        try:
            col = _get_collection()
            if col is not None:
                existing_doc = col.find_one({"hash": file_hash})
        except Exception as exc:
            logger.warning("[%s] Failed to query Mongo for hash: %s", document_id, exc)

        # Check local storage fallback
        if not existing_doc:
            try:
                local_projects = _load_projects_locally(include_raw_text=True)
                for p in local_projects:
                    if p.get("hash") == file_hash:
                        existing_doc = p
                        break
            except Exception as exc:
                logger.warning("[%s] Failed to query local fallback for hash: %s", document_id, exc)

        if existing_doc:
            logger.info("[%s] Duplicate document detected! Reusing data from document_id: %s", 
                        document_id, existing_doc.get("document_id"))
            
            # Reuse MongoDB / Local data
            reused_doc = dict(existing_doc)
            reused_doc["document_id"] = document_id
            reused_doc["source_file"] = source_file
            reused_doc["stored_file"] = os.path.basename(file_path)
            
            # Overwrite metadata values if new values are provided
            reused_metadata = reused_doc.get("metadata", {})
            if title: reused_metadata["title"] = title
            if builder: reused_metadata["builder"] = builder
            if project: reused_metadata["project"] = project
            if document_type: reused_metadata["document_type"] = document_type
            if description: reused_metadata["description"] = description
            if tags: reused_metadata["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
            
            # Excavator specific
            if doc_type: reused_metadata["doc_type"] = doc_type
            if revision_date: reused_metadata["revision_date"] = revision_date
            if section_path: reused_metadata["section_path"] = section_path
            if component_tags: reused_metadata["component_tags"] = [t.strip() for t in component_tags.split(",") if t.strip()]
            if dtc_codes: reused_metadata["dtc_codes"] = [t.strip() for t in dtc_codes.split(",") if t.strip()]
            if supersedes_doc_id: reused_metadata["supersedes_doc_id"] = supersedes_doc_id
            
            reused_doc["metadata"] = reused_metadata
            reused_doc["domain"] = domain

            save_project_to_mongo(reused_doc)
            
            # Update status to ready
            set_status(
                document_id,
                "ready",
                text_chunks_indexed=reused_doc.get("text_chunks_indexed", 0),
                images_indexed=reused_doc.get("images_indexed", 0),
                total_pages=len(reused_doc.get("pages", [])),
                message="Duplicate document detected. Reused previous embeddings and metadata successfully."
            )
            logger.info("=== BACKGROUND PROCESSOR SUCCESS (DEDUPLICATED) | doc=%s ===", document_id)
            return

        # ── 1. Text Extraction using Multiformat Parser ─────────────────────────
        set_status(document_id, "layout_parsing", message="Parsing document layout and hierarchy...")
        rag_images_dir = os.path.join(os.path.dirname(file_path), "rag_images")
        
        start_parse = datetime.now(timezone.utc)
        processed = parse_document(
            file_path=file_path,
            document_id=document_id,
            source_file=source_file,
            output_folder=rag_images_dir,
            image_base_path=f"documents/{document_id}/images",
        )
        end_parse = datetime.now(timezone.utc)
        logger.info("[%s] Ingestion step: PARSING | Elapsed: %.2fs", document_id, (end_parse - start_parse).total_seconds())

        pages = processed.get("pages", [])
        total_pages = processed.get("total_pages", len(pages))
        cleaned_text = clean_text(processed.get("full_text", ""))
        logger.info("[%s] Text extracted: %d chars, %d pages", document_id, len(cleaned_text), total_pages)

        # ── Incremental Versioning ───────────────────────────────────────────
        from storage.mongo_store import _get_collection, _load_projects_locally
        prev_doc = None
        try:
            col = _get_collection()
            if col is not None:
                prev_doc = col.find_one({"source_file": source_file}, sort=[("metadata.version", -1)])
            if not prev_doc:
                local_projects = _load_projects_locally(include_raw_text=True)
                for p in local_projects:
                    if p.get("source_file") == source_file:
                        if not prev_doc or float(p.get("metadata", {}).get("version", "1.0")) > float(prev_doc.get("metadata", {}).get("version", "1.0")):
                            prev_doc = p
        except Exception as exc:
            logger.warning("[%s] Failed to query previous version for versioning: %s", document_id, exc)

        doc_version = "1.0"
        if prev_doc:
            prev_hash = prev_doc.get("hash")
            if prev_hash != file_hash:
                try:
                    prev_ver_str = prev_doc.get("metadata", {}).get("version", "1.0")
                    doc_version = f"{float(prev_ver_str) + 0.1:.1f}"
                    logger.info("[%s] New version of %s detected. Incrementing version to %s", document_id, source_file, doc_version)
                except Exception:
                    doc_version = "1.1"
            else:
                doc_version = prev_doc.get("metadata", {}).get("version", "1.0")

        # ── 2. LLM metadata formatter ────────────────────────────────────────
        set_status(document_id, "extracting_text", message="Extracting structural text blocks...")
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

        # ── 4. Save original file to MongoDB GridFS ───────────────────────────
        try:
            saved_file_id = save_file_to_mongo(
                document_id=document_id,
                file_path=file_path,
                filename=source_file,
                content_type=content_type,
                file_kind="pdf" if source_file.lower().endswith(".pdf") else "other",
            )
            logger.info("[%s] Document file saved: id=%s", document_id, saved_file_id)
        except Exception as exc:
            logger.error("[%s] Document file save failed: %s", document_id, exc)
            saved_file_id = None

        # ── 5. Save page images to MongoDB GridFS if PDF or Scanned ───────────
        set_status(document_id, "extracting_images", message="Saving extracted page images...")
        image_metadata = []
        for page in pages:
            for image in page.get("images", []):
                image_id = image.get("image_id")
                local_path = image.get("local_path")
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
                    image["image_path"] = f"documents/{document_id}/images/{image_id}"
                    image_metadata.append(image)
                except Exception as exc:
                    logger.warning("[%s] Image save failed for %s: %s", document_id, image_id, exc)

        # ── 6. Assemble project document & Save ───────────────────────────
        json_data.update({
            "images": image_metadata,
            "pages": pages,
            "parents": processed.get("parents", {}),  # Store parents mapping!
            "document_id": document_id,
            "source_file": source_file,
            "stored_file": os.path.basename(file_path),
            "raw_text": cleaned_text,
            "pdf_path": f"documents/{document_id}/pdf",
            "rag_status": "processing",
            "hash": file_hash,  # Store hash for deduplication
            "domain": domain,
            "metadata": {
                "title": title or source_file,
                "builder": metadata_builder,
                "project": metadata_project,
                "document_type": document_type or "",
                "description": description or "",
                "tags": [t.strip() for t in (tags or "").split(",") if t.strip()],
                # Excavator fields
                "doc_type": doc_type or "",
                "revision_date": revision_date or "",
                "section_path": section_path or "",
                "component_tags": [t.strip() for t in (component_tags or "").split(",") if t.strip()],
                "dtc_codes": [t.strip() for t in (dtc_codes or "").split(",") if t.strip()],
                "supersedes_doc_id": supersedes_doc_id or "",
                "version": doc_version,
            },
        })

        # ── Stage 2: Chunk text ──────────────────────────────────────────
        set_status(document_id, "generating_chunks", total_pages=total_pages, message="Generating structure-aware chunks...")
        
        # Calculate dynamic confidence weight for excavator
        if domain == "excavator":
            weights = {
                "manuals": 1.0,
                "parts_catalog": 0.9,
                "service_bulletins": 0.85,
                "field_reports": 0.7,
                "maintenance_logs": 0.6,
            }
            confidence_weight = weights.get(doc_type.lower(), 0.5)
            
            doc_metadata = {
                "domain": "excavator",
                "doc_id": document_id,
                "doc_type": doc_type,
                "title": title or source_file,
                "source_file": source_file,
                "revision_date": revision_date,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "section_path": section_path,
                "machine_model": "R215L",
                "component_tags": [t.strip() for t in (component_tags or "").split(",") if t.strip()],
                "dtc_codes": [t.strip() for t in (dtc_codes or "").split(",") if t.strip()],
                "supersedes_doc_id": supersedes_doc_id,
                "confidence_weight": confidence_weight,
                "version": doc_version,
            }
        else:
            doc_metadata = {
                "domain": "real_estate",
                "project_name": metadata_project,
                "builder": metadata_builder,
                "document_type": document_type,
                "source_file": source_file,
                "hash": file_hash,
                "version": doc_version,
            }
            
        start_chunk = datetime.now(timezone.utc)
        text_chunks = chunk_text_pages(pages, document_id, doc_metadata)
        end_chunk = datetime.now(timezone.utc)
        logger.info("[%s] Ingestion step: CHUNKING | Elapsed: %.2fs", document_id, (end_chunk - start_chunk).total_seconds())
        logger.info("[%s] Text chunks created: %d", document_id, len(text_chunks))

        # ── Stage 3: Embed text ──────────────────────────────────────────
        set_status(document_id, "generating_embeddings", total_pages=total_pages, message="Generating Voyage embeddings...")
        
        start_embed_text = datetime.now(timezone.utc)
        if text_chunks:
            logger.info("[%s] Generating text embeddings...", document_id)
            text_embedder = get_text_embedder()
            chunk_texts = [c["content"] for c in text_chunks]
            text_vectors = text_embedder.embed_batch(chunk_texts, document_id=document_id)
            for chunk, vec in zip(text_chunks, text_vectors):
                chunk["vector"] = vec
        else:
            logger.warning("[%s] No text chunks to embed", document_id)
        end_embed_text = datetime.now(timezone.utc)
        logger.info("[%s] Ingestion step: TEXT_EMBEDDING | Elapsed: %.2fs", document_id, (end_embed_text - start_embed_text).total_seconds())

        # ── Stage 4: Embed images ────────────────────────────────────────
        image_records = process_images(pages, document_id, source_file, doc_metadata)
        logger.info("[%s] Image records created: %d", document_id, len(image_records))

        start_embed_img = datetime.now(timezone.utc)
        if image_records:
            logger.info("[%s] Generating image embeddings...", document_id)
            img_embedder = get_image_embedder()
            embedding_inputs = [
                (build_image_embedding_text(record), record.get("local_path"))
                for record in image_records
            ]
            image_vectors = img_embedder.embed_interleaved_batch(embedding_inputs, document_id=document_id)
            for record, vec in zip(image_records, image_vectors):
                record["vector"] = vec
        end_embed_img = datetime.now(timezone.utc)
        logger.info("[%s] Ingestion step: IMAGE_EMBEDDING | Elapsed: %.2fs", document_id, (end_embed_img - start_embed_img).total_seconds())

        # ── Stage 5: Index Qdrant ─────────────────────────────────────────
        set_status(document_id, "indexing", total_pages=total_pages, message="Indexing vectors into Qdrant Cloud...")
        create_collections()

        # Determine target Qdrant collections
        target_text_collection = None
        target_image_collection = None
        
        if domain == "excavator":
            target_image_collection = "im_manuals_images"
            if doc_type == "manuals":
                target_text_collection = "im_manuals_text"
            elif doc_type == "service_bulletins":
                target_text_collection = "im_service_bulletins"
            elif doc_type == "maintenance_logs":
                target_text_collection = "im_maintenance_logs"
            elif doc_type == "parts_catalog":
                target_text_collection = "im_parts_catalog"
            elif doc_type == "field_reports":
                target_text_collection = "im_field_reports"

        from retrieval.qdrant_service import set_indexing_threshold
        set_indexing_threshold(0)
        
        start_index = datetime.now(timezone.utc)
        try:
            if text_chunks:
                text_chunks_indexed = upsert_text_chunks(text_chunks, collection_name=target_text_collection)
            if image_records:
                images_indexed = upsert_image_chunks(image_records, collection_name=target_image_collection)
        finally:
            set_indexing_threshold(20000)
        end_index = datetime.now(timezone.utc)
        logger.info("[%s] Ingestion step: INDEXING | Elapsed: %.2fs", document_id, (end_index - start_index).total_seconds())

        # Save metadata to MongoDB/Local Fallback
        json_data["text_chunks_indexed"] = text_chunks_indexed
        json_data["images_indexed"] = images_indexed
        json_data["rag_status"] = "ready"
        save_project_to_mongo(json_data)

        # ── Stage 6: Ready! ──────────────────────────────────────────────
        set_status(
            document_id,
            "ready",
            text_chunks_indexed=text_chunks_indexed,
            images_indexed=images_indexed,
            total_pages=total_pages,
            message="Completed"
        )
        logger.info("=== BACKGROUND PROCESSOR SUCCESS | doc=%s ===", document_id)
        
    except Exception as e:
        logger.error("=== BACKGROUND PROCESSOR FAILED | doc=%s | error=%s ===", document_id, e, exc_info=True)
        set_status(
            document_id,
            "failed",
            error=str(e)[:400]
        )
