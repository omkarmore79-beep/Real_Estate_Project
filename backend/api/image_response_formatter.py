"""
Image Response Formatter for Enhanced UI Display.

Formats retrieved images for frontend display including:
- Retrieved image with caption
- Page
- Section
- Supporting explanation
- Exact retrieved figure (not unrelated page images)
- Confidence score
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def format_image_for_ui(image_result: dict) -> dict:
    """
    Format a retrieved image result for UI display.
    
    Returns a dictionary with all UI-relevant fields.
    """
    metadata = image_result.get("metadata", {})
    
    ui_image = {
        "image_id": metadata.get("image_id", ""),
        "image_url": image_result.get("image_url") or metadata.get("image_url", ""),
        "image_path": image_result.get("image_path") or metadata.get("image_path", ""),
        "local_path": image_result.get("local_path"),
        "image_type": metadata.get("image_type", "other"),
        "caption": metadata.get("caption", ""),
        "figure_number": metadata.get("figure_number"),
        "page_number": metadata.get("page_number"),
        "section": metadata.get("section", "General"),
        "subsection": metadata.get("subsection", ""),
        "confidence": image_result.get("combined_confidence", image_result.get("score", 0.0)),
        "confidence_level": image_result.get("confidence_level", "unknown"),
        # Supporting context
        "supporting_explanation": metadata.get("multimodal_description", ""),
        "nearby_paragraph": metadata.get("nearby_paragraph", ""),
        "section_title": metadata.get("section_title", ""),
        # Parent context
        "parent_chunk_id": metadata.get("parent_chunk_id"),
        "parent_section": metadata.get("parent_section"),
        "parent_page": metadata.get("parent_page"),
        # Additional metadata
        "components": metadata.get("components", []),
        "diagram_subtype": metadata.get("diagram_subtype"),
        "spatial_description": metadata.get("spatial_description"),
    }
    
    return ui_image


def format_images_for_ui(results: list[dict]) -> list[dict]:
    """
    Format all retrieved images for UI display.
    
    Filters out unrelated page images and prioritizes exact figures.
    """
    # Separate images from other result types
    image_results = [r for r in results if r.get("source_type") == "image"]
    
    # Filter out full_page images (unrelated renders)
    filtered_images = [
        img for img in image_results
        if img.get("metadata", {}).get("image_type") != "full_page"
    ]
    
    # If no specific images, fall back to all images
    if not filtered_images and image_results:
        filtered_images = image_results
    
    # Format for UI
    ui_images = [format_image_for_ui(img) for img in filtered_images]
    
    # Sort by confidence
    ui_images.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    
    return ui_images


def build_image_gallery(images: list[dict], max_images: int = 6) -> dict:
    """
    Build an image gallery structure for the UI.
    
    Returns:
        Dictionary with:
        - total_images: Total number of images
        - displayed_images: Number of images to display
        - images: List of formatted images
        - has_diagrams: Whether any images are diagrams
        - has_charts: Whether any images are charts
        - has_photos: Whether any images are photos
    """
    ui_images = format_images_for_ui(images)
    displayed = ui_images[:max_images]
    
    # Classify image types
    has_diagrams = any(img.get("image_type") in ["engineering_diagram", "exploded_view", "hydraulic_diagram", "electrical_diagram", "flowchart"] for img in displayed)
    has_charts = any(img.get("image_type") in ["lifting_chart", "chart", "graph"] for img in displayed)
    has_photos = any(img.get("image_type") in ["photograph", "exterior", "interior"] for img in displayed)
    
    return {
        "total_images": len(ui_images),
        "displayed_images": len(displayed),
        "images": displayed,
        "has_diagrams": has_diagrams,
        "has_charts": has_charts,
        "has_photos": has_photos,
    }


def format_single_image_response(image_result: dict, context_chunks: list[dict]) -> dict:
    """
    Format a single image with its surrounding context.
    
    Used when the query specifically asks for one image/figure.
    """
    ui_image = format_image_for_ui(image_result)
    
    # Add context from nearby chunks
    context_text = []
    for chunk in context_chunks[:3]:
        content = chunk.get("content", "")
        if content and chunk.get("source_type") != "context_window":
            context_text.append(content[:300])
    
    return {
        "image": ui_image,
        "context": "\n\n".join(context_text),
        "has_context": len(context_text) > 0,
    }


def get_image_caption_with_context(image_result: dict) -> str:
    """
    Generate a rich caption for an image with context.
    
    Combines figure number, caption, section, and nearby paragraph.
    """
    metadata = image_result.get("metadata", {})
    
    parts = []
    
    figure_number = metadata.get("figure_number")
    if figure_number:
        parts.append(f"Figure {figure_number}")
    
    caption = metadata.get("caption", "")
    if caption:
        parts.append(caption)
    
    section = metadata.get("section")
    if section and section != "General":
        parts.append(f"in {section}")
    
    page = metadata.get("page_number")
    if page is not None:
        parts.append(f"on page {page}")
    
    nearby_paragraph = metadata.get("nearby_paragraph")
    if nearby_paragraph and len(nearby_paragraph) > 50:
        first_sentence = nearby_paragraph.split(".")[0][:150]
        parts.append(f"Context: {first_sentence}")
    
    return ". ".join(parts)


def format_answer_with_images(
    answer: str,
    results: list[dict],
    max_inline_images: int = 3,
) -> dict:
    """
    Format an answer with embedded images.
    
    Returns:
        Dictionary with:
        - answer: The text answer
        - images: List of images to display
        - image_gallery: Structured gallery
        - citations: Formatted citations
    """
    # Build image gallery
    image_gallery = build_image_gallery(results, max_inline_images)
    
    # Format citations
    from retrieval.citation_builder import format_citations_for_answer
    citations = format_citations_for_answer(results)
    
    return {
        "answer": answer,
        "images": image_gallery["images"],
        "image_gallery": image_gallery,
        "citations": citations,
        "has_images": len(image_gallery["images"]) > 0,
    }


def filter_relevant_images(query: str, images: list[dict]) -> list[dict]:
    """
    Filter images to only show those relevant to the query.
    
    Uses intent detection to prioritize matching image types.
    """
    from retrieval.intent_detector import detect_query_intent
    
    intent_info = detect_query_intent(query)
    detected_types = intent_info.get("detected_types", [])
    
    if detected_types:
        # Filter to only matching types
        filtered = [
            img for img in images
            if img.get("image_type") in detected_types
        ]
        if filtered:
            return filtered
    
    # No specific types or no matches, return all sorted by confidence
    return images


def get_image_display_priority(image_type: str) -> int:
    """
    Get display priority for an image type.
    
    Lower number = higher priority (displayed first).
    """
    priorities = {
        "engineering_diagram": 1,
        "exploded_view": 2,
        "hydraulic_diagram": 3,
        "electrical_diagram": 4,
        "lifting_chart": 5,
        "flowchart": 6,
        "diagram": 7,
        "chart": 8,
        "table": 9,
        "floor_plan": 10,
        "master_plan": 11,
        "location_map": 12,
        "photograph": 13,
        "exterior": 14,
        "interior": 15,
        "other": 20,
        "full_page": 99,
    }
    
    return priorities.get(image_type, 20)


def sort_images_by_relevance(images: list[dict]) -> list[dict]:
    """
    Sort images by relevance (type priority + confidence).
    """
    def sort_key(img):
        image_type = img.get("image_type", "other")
        priority = get_image_display_priority(image_type)
        confidence = img.get("confidence", 0)
        return (priority, -confidence)  # Lower priority number first, higher confidence first
    
    return sorted(images, key=sort_key)


def build_image_tooltip(image: dict) -> str:
    """
    Build a tooltip text for an image on hover.
    """
    parts = []
    
    image_type = image.get("image_type", "").replace("_", " ").title()
    if image_type:
        parts.append(f"Type: {image_type}")
    
    caption = image.get("caption", "")
    if caption:
        parts.append(f"Caption: {caption[:80]}")
    
    figure_number = image.get("figure_number")
    if figure_number:
        parts.append(f"Figure: {figure_number}")
    
    page = image.get("page_number")
    if page is not None:
        parts.append(f"Page: {page}")
    
    section = image.get("section")
    if section and section != "General":
        parts.append(f"Section: {section}")
    
    confidence = image.get("confidence", 0)
    if confidence > 0:
        parts.append(f"Confidence: {confidence:.2f}")
    
    return " | ".join(parts)
