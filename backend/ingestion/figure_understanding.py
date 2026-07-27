"""
Figure Understanding Module for Industrial Multimodal RAG.

Treats every figure as an independent retrieval object.
Stores:
- figure_id
- figure_title
- caption
- page
- parent_section
- parent_chunk
- figure_type
- figure_number

Links figures to their parent context for retrieval.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# Pattern for detecting figure references
_FIGURE_PATTERN = re.compile(
    r"(?:figure|fig\.?|diagram|img\.?|image|exhibit|attachment|plate|chart)\s*[\#\-]?\s*([A-Za-z0-9\.\-_]+)?",
    re.IGNORECASE
)

_FIGURE_CAPTION_PATTERN = re.compile(
    r"(?:^|[\.:\-])\s*(figure|fig\.?|source|caption|description|note|illustration)\s*[\#\-]?\s*([A-Za-z0-9\.\-_]*)[\s\:]*(.+?)(?=[\.!?]|$)",
    re.IGNORECASE | re.MULTILINE
)


def extract_figures_from_page(
    page: dict,
    document_id: str,
    metadata: dict | None = None,
) -> list[dict]:
    """
    Extract all figures from a page as independent retrieval objects.
    
    Args:
        page: Page dictionary from PDF processor
        document_id: Document identifier
        metadata: Additional metadata
    
    Returns:
        List of figure records for indexing
    """
    meta = metadata or {}
    figure_records = []
    
    page_number = page.get("page_number", 0)
    page_text = (page.get("text") or "").strip()
    section = page.get("section", "General")
    
    # Extract images from page
    images = page.get("images", [])
    
    for img_idx, img in enumerate(images):
        image_id = img.get("image_id") or f"page_{page_number}_img_{img_idx}"
        image_path = img.get("image_path", "")
        local_path = img.get("local_path")
        
        # Determine if this image is a figure (has caption or figure number)
        figure_number = extract_figure_number_from_context(page_text, image_id)
        caption = extract_caption_from_context(page_text, image_id)
        
        # If it has figure number or caption, treat as figure
        if figure_number or caption:
            figure_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_figure_{figure_number}_{page_number}"))
            
            # Determine figure type from image metadata
            figure_type = img.get("image_type", "other")
            
            # Generate figure title
            figure_title = generate_figure_title(figure_number, caption, figure_type)
            
            # Create parent chunk reference
            parent_chunk_id = f"{document_id}_page_{page_number}"
            
            # Build figure record
            figure_record = {
                "figure_id": figure_id,
                "document_id": document_id,
                "page_number": page_number,
                "figure_number": figure_number,
                "figure_title": figure_title,
                "caption": caption,
                "figure_type": figure_type,
                "image_path": image_path,
                "local_path": local_path,
                "image_id": image_id,
                "vector": [],  # filled by embedding service
                # Parent-child relationships
                "parent_section": section,
                "parent_chunk": parent_chunk_id,
                "parent_page": page_number,
                "parent_document": document_id,
                # Metadata
                "metadata": {
                    "document_id": document_id,
                    "page_number": page_number,
                    "figure_id": figure_id,
                    "figure_number": figure_number,
                    "figure_title": figure_title,
                    "caption": caption,
                    "figure_type": figure_type,
                    "image_path": image_path,
                    "image_id": image_id,
                    "parent_section": section,
                    "parent_chunk_id": parent_chunk_id,
                    "parent_page": page_number,
                    "parent_document": document_id,
                    "source_file": meta.get("source_file", ""),
                    "project": meta.get("project_name", meta.get("project", "")),
                    "builder": meta.get("builder", ""),
                    "document_type": meta.get("document_type", ""),
                    "domain": meta.get("domain", "generic"),
                    "machine_model": meta.get("machine_model", ""),
                    "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }
            
            figure_records.append(figure_record)
    
    return figure_records


def extract_figure_number_from_context(page_text: str, image_id: str = "") -> Optional[str]:
    """
    Extract figure number from page text.
    
    Looks for patterns like "Figure 3.2", "Fig. A1", "Figure 1-5".
    """
    match = _FIGURE_PATTERN.search(page_text)
    if match and match.group(1):
        return match.group(1).strip()
    
    # Try to extract from image_id if it contains figure info
    if image_id:
        fig_match = re.search(r"fig(?:ure)?\s*[\#\-]?\s*([A-Za-z0-9\.\-_]+)", image_id, re.IGNORECASE)
        if fig_match:
            return fig_match.group(1).strip()
    
    return None


def extract_caption_from_context(page_text: str, image_id: str = "") -> Optional[str]:
    """
    Extract caption from page text.
    
    Looks for explicitly marked captions near figure references.
    """
    # Try explicit caption pattern
    caption_match = _FIGURE_CAPTION_PATTERN.search(page_text)
    if caption_match and caption_match.group(3):
        caption = caption_match.group(3).strip()
        if len(caption) > 10:
            return caption
    
    # Try to find text after figure reference
    fig_match = _FIGURE_PATTERN.search(page_text)
    if fig_match:
        # Get text after the figure reference
        after_fig = page_text[fig_match.end():].strip()
        if after_fig:
            # Take first sentence or up to 200 chars
            first_sentence = re.split(r'[\.!?]', after_fig)[0].strip()
            if len(first_sentence) > 10:
                return first_sentence[:200]
    
    return None


def generate_figure_title(
    figure_number: Optional[str],
    caption: Optional[str],
    figure_type: str,
) -> str:
    """
    Generate a title for the figure.
    
    Uses figure number and caption to create a descriptive title.
    """
    if caption and len(caption) > 10:
        if figure_number:
            return f"Figure {figure_number}: {caption}"
        return caption
    
    if figure_number:
        type_label = _get_figure_type_label(figure_type)
        return f"Figure {figure_number}: {type_label}"
    
    return _get_figure_type_label(figure_type)


def _get_figure_type_label(figure_type: str) -> str:
    """Get a human-readable label for figure type."""
    type_labels = {
        "engineering_diagram": "Engineering Diagram",
        "exploded_view": "Exploded View",
        "hydraulic_diagram": "Hydraulic Diagram",
        "electrical_diagram": "Electrical Diagram",
        "flowchart": "Flowchart",
        "lifting_chart": "Lifting Chart",
        "maintenance_image": "Maintenance Procedure Image",
        "troubleshooting_image": "Troubleshooting Image",
        "warning_label": "Warning Label",
        "floor_plan": "Floor Plan",
        "master_plan": "Master Plan",
        "location_map": "Location Map",
        "amenity": "Amenity",
        "exterior": "Exterior View",
        "interior": "Interior View",
        "diagram": "Diagram",
        "chart": "Chart",
        "table": "Table",
        "photograph": "Photograph",
        "other": "Figure",
    }
    return type_labels.get(figure_type, "Figure")


def generate_figure_embedding_text(figure_record: dict) -> str:
    """
    Generate text for figure embedding.
    
    Combines figure number, title, caption, and type.
    """
    parts = []
    
    figure_number = figure_record.get("figure_number")
    if figure_number:
        parts.append(f"Figure {figure_number}")
    
    figure_title = figure_record.get("figure_title")
    if figure_title:
        parts.append(figure_title)
    
    caption = figure_record.get("caption")
    if caption and caption != figure_title:
        parts.append(caption)
    
    figure_type = figure_record.get("figure_type")
    if figure_type:
        type_label = _get_figure_type_label(figure_type)
        parts.append(f"Type: {type_label}")
    
    page_number = figure_record.get("page_number")
    if page_number:
        parts.append(f"Page {page_number}")
    
    parent_section = figure_record.get("parent_section")
    if parent_section and parent_section != "General":
        parts.append(f"Section: {parent_section}")
    
    combined = ". ".join(parts)
    return combined if combined else "Document figure"


def process_figures_from_pages(
    pages: list[dict],
    document_id: str,
    metadata: dict | None = None,
) -> list[dict]:
    """
    Process all figures from PDF pages.
    
    Args:
        pages: List of page dictionaries from PDF processor
        document_id: Document identifier
        metadata: Additional metadata
    
    Returns:
        List of figure records for indexing
    """
    all_figures = []
    
    for page in pages:
        figures = extract_figures_from_page(page, document_id, metadata)
        all_figures.extend(figures)
    
    logger.info(
        "Processed %d figures for document_id=%s",
        len(all_figures), document_id
    )
    
    # Log figure type distribution
    type_dist = {}
    for fig in all_figures:
        ftype = fig.get("figure_type", "unknown")
        type_dist[ftype] = type_dist.get(ftype, 0) + 1
    
    for ftype, count in type_dist.items():
        logger.info(f"  Figure type '{ftype}': {count} figures")
    
    return all_figures


def link_figures_to_chunks(
    figures: list[dict],
    chunks: list[dict],
) -> list[dict]:
    """
    Link figures to their parent chunks.
    
    Updates figure records with parent_chunk_id based on page and section.
    """
    # Create mapping of (page_number, section) -> chunk_id
    chunk_map = {}
    for chunk in chunks:
        chunk_meta = chunk.get("metadata", {})
        page_num = chunk_meta.get("page_number")
        section = chunk_meta.get("section", "General")
        chunk_id = chunk.get("chunk_id")
        
        if page_num is not None and chunk_id:
            chunk_map[(page_num, section)] = chunk_id
    
    # Update figures with parent chunk references
    for figure in figures:
        page_num = figure.get("page_number")
        section = figure.get("parent_section", "General")
        
        if page_num is not None:
            chunk_key = (page_num, section)
            if chunk_key in chunk_map:
                figure["parent_chunk_id"] = chunk_map[chunk_key]
                figure["metadata"]["parent_chunk_id"] = chunk_map[chunk_key]
    
    logger.info("Linked %d figures to parent chunks", len(figures))
    return figures


def get_figure_by_number(
    figures: list[dict],
    figure_number: str,
) -> Optional[dict]:
    """
    Retrieve a figure by its figure number.
    
    Useful for answering queries like "What does Figure 3.2 show?".
    """
    for fig in figures:
        if fig.get("figure_number") == figure_number:
            return fig
    return None


def get_figures_by_type(
    figures: list[dict],
    figure_type: str,
) -> list[dict]:
    """
    Retrieve all figures of a specific type.
    
    Useful for queries like "Show me all exploded views".
    """
    return [fig for fig in figures if fig.get("figure_type") == figure_type]


def get_figures_by_section(
    figures: list[dict],
    section: str,
) -> list[dict]:
    """
    Retrieve all figures from a specific section.
    
    Useful for queries like "What diagrams are in the Maintenance section?".
    """
    return [fig for fig in figures if fig.get("parent_section") == section]
