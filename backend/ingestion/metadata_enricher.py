"""
Metadata Enrichment Module for Industrial Multimodal RAG.

Ensures every indexed object contains comprehensive metadata:
- document_id
- page_number
- section
- subsection
- chunk_type
- figure_id
- table_id
- caption
- machine_model
- document_version
- ingestion_timestamp
- builder
- project
- previous_chunk
- next_chunk
- ocr_confidence
- layout_confidence
- image_type
- parent_chunk
- parent_page
- parent_section
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# Required metadata fields for different object types
_REQUIRED_TEXT_CHUNK_FIELDS = [
    "document_id",
    "page_number",
    "section",
    "subsection",
    "chunk_type",
    "chunk_index",
    "source_type",
    "ocr_used",
    "ocr_confidence",
    "confidence_score",
    "word_count",
    "char_count",
    "ingestion_timestamp",
    "previous_chunk_id",
    "next_chunk_id",
    "parent_chunk",
    "parent_page",
    "parent_section",
    "parent_document",
]

_REQUIRED_IMAGE_FIELDS = [
    "document_id",
    "page_number",
    "image_id",
    "image_type",
    "caption",
    "figure_number",
    "type_confidence",
    "image_confidence",
    "section_title",
    "ingestion_timestamp",
    "parent_chunk_id",
    "parent_section",
    "parent_page",
    "parent_document",
    "domain",
    "machine_model",
]

_REQUIRED_TABLE_FIELDS = [
    "document_id",
    "page_number",
    "table_id",
    "table_number",
    "title",
    "row_count",
    "column_count",
    "ingestion_timestamp",
    "domain",
    "machine_model",
]

_REQUIRED_FIGURE_FIELDS = [
    "document_id",
    "page_number",
    "figure_id",
    "figure_number",
    "figure_title",
    "caption",
    "figure_type",
    "parent_section",
    "parent_chunk",
    "parent_page",
    "parent_document",
    "ingestion_timestamp",
]


def enrich_text_chunk_metadata(
    chunk: dict,
    metadata: dict | None = None,
) -> dict:
    """
    Enrich a text chunk with comprehensive metadata.
    
    Ensures all required fields are present with sensible defaults.
    """
    meta = metadata or {}
    chunk_metadata = chunk.get("metadata", {})
    
    # Ensure basic fields
    enriched = {
        "document_id": chunk_metadata.get("document_id") or meta.get("document_id", ""),
        "page_number": chunk_metadata.get("page_number", 0),
        "section": chunk_metadata.get("section", "General"),
        "subsection": chunk_metadata.get("subsection", ""),
        "chunk_type": chunk_metadata.get("chunk_type", "paragraph"),
        "chunk_index": chunk_metadata.get("chunk_index", 0),
        "source_type": chunk_metadata.get("source_type", "pdf_text"),
        "ocr_used": chunk_metadata.get("ocr_used", False),
        "ocr_confidence": chunk_metadata.get("ocr_confidence", 1.0),
        "confidence_score": chunk_metadata.get("confidence_score", 1.0),
        "word_count": chunk_metadata.get("word_count", 0),
        "char_count": chunk_metadata.get("char_count", 0),
        "ingestion_timestamp": chunk_metadata.get("ingestion_timestamp") or datetime.now(timezone.utc).isoformat(),
        "previous_chunk_id": chunk_metadata.get("previous_chunk_id"),
        "next_chunk_id": chunk_metadata.get("next_chunk_id"),
        # Parent-child relationships
        "parent_chunk": chunk_metadata.get("parent_chunk") or chunk.get("chunk_id"),
        "parent_page": chunk_metadata.get("parent_page", chunk_metadata.get("page_number", 0)),
        "parent_section": chunk_metadata.get("parent_section", chunk_metadata.get("section", "General")),
        "parent_document": chunk_metadata.get("parent_document") or chunk_metadata.get("document_id", ""),
        # Additional fields
        "project": meta.get("project_name", meta.get("project", "")),
        "builder": meta.get("builder", ""),
        "document_type": meta.get("document_type", ""),
        "source_file": meta.get("source_file", ""),
        "domain": meta.get("domain", "generic"),
        "machine_model": meta.get("machine_model", ""),
        "document_version": meta.get("version", ""),
        # Layout confidence (estimated from source type)
        "layout_confidence": _estimate_layout_confidence(chunk_metadata),
    }
    
    # Preserve any additional metadata
    for key, value in chunk_metadata.items():
        if key not in enriched:
            enriched[key] = value
    
    # Update chunk metadata
    chunk["metadata"] = enriched
    
    return chunk


def enrich_image_metadata(
    image: dict,
    metadata: dict | None = None,
) -> dict:
    """
    Enrich an image record with comprehensive metadata.
    """
    meta = metadata or {}
    image_metadata = image.get("metadata", {})
    
    enriched = {
        "document_id": image_metadata.get("document_id") or meta.get("document_id", ""),
        "page_number": image_metadata.get("page_number", 0),
        "image_id": image_metadata.get("image_id", ""),
        "image_type": image_metadata.get("image_type", "other"),
        "caption": image_metadata.get("caption", ""),
        "figure_number": image_metadata.get("figure_number"),
        "type_confidence": image_metadata.get("type_confidence", 0.5),
        "image_confidence": image_metadata.get("image_confidence", 0.5),
        "section_title": image_metadata.get("section_title", "General"),
        "ingestion_timestamp": image_metadata.get("ingestion_timestamp") or datetime.now(timezone.utc).isoformat(),
        # Parent-child relationships
        "parent_chunk_id": image_metadata.get("parent_chunk_id"),
        "parent_section": image_metadata.get("parent_section", "General"),
        "parent_page": image_metadata.get("parent_page", 0),
        "parent_document": image_metadata.get("parent_document") or image_metadata.get("document_id", ""),
        # Additional fields
        "project": meta.get("project_name", meta.get("project", "")),
        "builder": meta.get("builder", ""),
        "document_type": meta.get("document_type", ""),
        "source_file": meta.get("source_file", ""),
        "domain": meta.get("domain", "generic"),
        "machine_model": meta.get("machine_model", ""),
        "multimodal_description": image_metadata.get("multimodal_description", ""),
    }
    
    # Preserve any additional metadata
    for key, value in image_metadata.items():
        if key not in enriched:
            enriched[key] = value
    
    # Update image metadata
    image["metadata"] = enriched
    
    return image


def enrich_table_metadata(
    table: dict,
    metadata: dict | None = None,
) -> dict:
    """
    Enrich a table record with comprehensive metadata.
    """
    meta = metadata or {}
    table_metadata = table.get("metadata", {})
    
    enriched = {
        "document_id": table_metadata.get("document_id") or meta.get("document_id", ""),
        "page_number": table_metadata.get("page_number", 0),
        "table_id": table_metadata.get("table_id", ""),
        "table_number": table_metadata.get("table_number", ""),
        "title": table_metadata.get("title", ""),
        "row_count": table_metadata.get("row_count", 0),
        "column_count": table_metadata.get("column_count", 0),
        "ingestion_timestamp": table_metadata.get("ingestion_timestamp") or datetime.now(timezone.utc).isoformat(),
        # Additional fields
        "project": meta.get("project_name", meta.get("project", "")),
        "builder": meta.get("builder", ""),
        "document_type": meta.get("document_type", ""),
        "source_file": meta.get("source_file", ""),
        "domain": meta.get("domain", "generic"),
        "machine_model": meta.get("machine_model", ""),
        "column_headers": table_metadata.get("column_headers", []),
        "row_headers": table_metadata.get("row_headers", []),
        "units": table_metadata.get("units", {}),
    }
    
    # Preserve any additional metadata
    for key, value in table_metadata.items():
        if key not in enriched:
            enriched[key] = value
    
    # Update table metadata
    table["metadata"] = enriched
    
    return table


def enrich_figure_metadata(
    figure: dict,
    metadata: dict | None = None,
) -> dict:
    """
    Enrich a figure record with comprehensive metadata.
    """
    meta = metadata or {}
    figure_metadata = figure.get("metadata", {})
    
    enriched = {
        "document_id": figure_metadata.get("document_id") or meta.get("document_id", ""),
        "page_number": figure_metadata.get("page_number", 0),
        "figure_id": figure_metadata.get("figure_id", ""),
        "figure_number": figure_metadata.get("figure_number"),
        "figure_title": figure_metadata.get("figure_title", ""),
        "caption": figure_metadata.get("caption", ""),
        "figure_type": figure_metadata.get("figure_type", "other"),
        "parent_section": figure_metadata.get("parent_section", "General"),
        "parent_chunk": figure_metadata.get("parent_chunk"),
        "parent_chunk_id": figure_metadata.get("parent_chunk_id"),
        "parent_page": figure_metadata.get("parent_page", 0),
        "parent_document": figure_metadata.get("parent_document") or figure_metadata.get("document_id", ""),
        "ingestion_timestamp": figure_metadata.get("ingestion_timestamp") or datetime.now(timezone.utc).isoformat(),
        # Additional fields
        "project": meta.get("project_name", meta.get("project", "")),
        "builder": meta.get("builder", ""),
        "document_type": meta.get("document_type", ""),
        "source_file": meta.get("source_file", ""),
        "domain": meta.get("domain", "generic"),
        "machine_model": meta.get("machine_model", ""),
    }
    
    # Preserve any additional metadata
    for key, value in figure_metadata.items():
        if key not in enriched:
            enriched[key] = value
    
    # Update figure metadata
    figure["metadata"] = enriched
    
    return figure


def _estimate_layout_confidence(metadata: dict) -> float:
    """
    Estimate layout confidence based on source type and OCR confidence.
    """
    source_type = metadata.get("source_type", "pdf_text")
    ocr_confidence = metadata.get("ocr_confidence", 1.0)
    
    if source_type == "pdf_text":
        return 0.95
    elif source_type == "ocr":
        return ocr_confidence
    elif source_type == "mixed":
        return (0.95 + ocr_confidence) / 2
    
    return 0.8


def validate_metadata_completeness(
    obj: dict,
    obj_type: str,
) -> dict:
    """
    Validate that an object has all required metadata fields.
    
    Returns:
        Dictionary with:
        - is_valid: Boolean
        - missing_fields: List of missing fields
        - warnings: List of warnings
    """
    if obj_type == "text_chunk":
        required = _REQUIRED_TEXT_CHUNK_FIELDS
    elif obj_type == "image":
        required = _REQUIRED_IMAGE_FIELDS
    elif obj_type == "table":
        required = _REQUIRED_TABLE_FIELDS
    elif obj_type == "figure":
        required = _REQUIRED_FIGURE_FIELDS
    else:
        return {"is_valid": False, "missing_fields": [], "warnings": ["Unknown object type"]}
    
    metadata = obj.get("metadata", {})
    missing_fields = []
    warnings = []
    
    for field in required:
        if field not in metadata or metadata[field] is None or metadata[field] == "":
            missing_fields.append(field)
    
    # Check for low confidence values
    if "confidence_score" in metadata and metadata["confidence_score"] < 0.3:
        warnings.append(f"Low confidence score: {metadata['confidence_score']}")
    
    if "ocr_confidence" in metadata and metadata["ocr_confidence"] < 0.5:
        warnings.append(f"Low OCR confidence: {metadata['ocr_confidence']}")
    
    return {
        "is_valid": len(missing_fields) == 0,
        "missing_fields": missing_fields,
        "warnings": warnings,
    }


def enrich_all_objects(
    chunks: list[dict],
    images: list[dict],
    tables: list[dict],
    figures: list[dict],
    metadata: dict | None = None,
) -> dict:
    """
    Enrich all object types with comprehensive metadata.
    
    Returns summary of enrichment results.
    """
    meta = metadata or {}
    
    # Enrich chunks
    enriched_chunks = [enrich_text_chunk_metadata(chunk, meta) for chunk in chunks]
    
    # Enrich images
    enriched_images = [enrich_image_metadata(img, meta) for img in images]
    
    # Enrich tables
    enriched_tables = [enrich_table_metadata(tbl, meta) for tbl in tables]
    
    # Enrich figures
    enriched_figures = [enrich_figure_metadata(fig, meta) for fig in figures]
    
    # Validate completeness
    validation_results = {
        "chunks": [validate_metadata_completeness(c, "text_chunk") for c in enriched_chunks],
        "images": [validate_metadata_completeness(i, "image") for i in enriched_images],
        "tables": [validate_metadata_completeness(t, "table") for t in enriched_tables],
        "figures": [validate_metadata_completeness(f, "figure") for f in enriched_figures],
    }
    
    # Count valid objects
    valid_counts = {
        "chunks": sum(1 for v in validation_results["chunks"] if v["is_valid"]),
        "images": sum(1 for v in validation_results["images"] if v["is_valid"]),
        "tables": sum(1 for v in validation_results["tables"] if v["is_valid"]),
        "figures": sum(1 for v in validation_results["figures"] if v["is_valid"]),
    }
    
    logger.info(
        "Metadata enrichment complete: %d/%d chunks, %d/%d images, %d/%d tables, %d/%d figures valid",
        valid_counts["chunks"], len(enriched_chunks),
        valid_counts["images"], len(enriched_images),
        valid_counts["tables"], len(enriched_tables),
        valid_counts["figures"], len(enriched_figures),
    )
    
    return {
        "chunks": enriched_chunks,
        "images": enriched_images,
        "tables": enriched_tables,
        "figures": enriched_figures,
        "validation": validation_results,
        "valid_counts": valid_counts,
    }
