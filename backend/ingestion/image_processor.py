"""
Image Processor — text-context-based image metadata creator.

Does NOT run OCR or call a vision LLM.
Classifies images purely from:
  - Page text (nearby context)
  - Image filename / image_id patterns

Reuses the text-fallback classification logic already present in
``ingestion/image_analyzer.py`` (_fallback_page_metadata) to keep
classification consistent.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Attempt to use the enhanced image analyzer from repository root.
_IMAGE_ANALYZER_ENHANCED_AVAILABLE = False
try:
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    from image_analyzer_enhanced import (
        build_image_embedding_text_enhanced,
        process_images_enhanced,
    )
    _IMAGE_ANALYZER_ENHANCED_AVAILABLE = True
except Exception:
    build_image_embedding_text_enhanced = None
    process_images_enhanced = None

from PIL import Image

logger = logging.getLogger(__name__)

MAX_VOYAGE_IMAGE_SIDE = 768
VOYAGE_JPEG_QUALITY = 85


def compress_image_for_voyage(
    path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
    *,
    max_side: int = MAX_VOYAGE_IMAGE_SIDE,
    quality: int = VOYAGE_JPEG_QUALITY,
) -> str | None:
    """Resize and JPEG-compress an image for Voyage multimodal embedding.

    The returned file is deliberately caller-owned: callers should retain the
    output directory until the API request has completed.  ``None`` is returned
    for unreadable or uninformative (tiny) images.
    """
    source = Path(path)
    if not source.is_file():
        return None
    destination_dir = Path(output_dir) if output_dir else source.parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{source.stem}.voyage.jpg"
    try:
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            if min(width, height) < 120:
                return None
            if max(width, height) > max_side:
                scale = max_side / max(width, height)
                image = image.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    Image.Resampling.LANCZOS,
                )
            image.save(destination, format="JPEG", quality=quality, optimize=True)
        return str(destination)
    except (OSError, ValueError) as exc:
        logger.warning("Could not preprocess image %s: %s", source, exc)
        return None

# ── Image type labels ──────────────────────────────────────────────────────────
IMAGE_TYPES = (
    "floor_plan",
    "master_plan",
    "location_map",
    "amenity",
    "exterior",
    "interior",
    "logo",
    "table",
    "diagram",
    "safety_label",
    "full_page",
    "other",
)

# ── Classification rules: (image_type, phrases, keywords) ─────────────────────
_CLASSIFICATION_RULES: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    (
        "floor_plan",
        ("floor plan", "unit plan", "apartment layout", "carpet area", "super built-up"),
        ("bedroom", "kitchen", "balcony", "bhk", "living", "toilet", "dimensions"),
    ),
    (
        "master_plan",
        ("master plan", "master layout plan", "site layout", "township", "tower layout"),
        ("tower", "towers", "internal roads", "landscape", "entry", "podium"),
    ),
    (
        "location_map",
        ("location plan", "location map", "connectivity", "nearby landmarks", "map not to scale"),
        ("airport", "metro", "railway", "hospital", "school", "highway", "landmark", "distance"),
    ),
    (
        "amenity",
        ("amenities", "clubhouse", "gymnasium", "swimming pool", "jogging track", "indoor games"),
        ("gym", "pool", "kids", "play", "jogging", "games", "garden", "sports"),
    ),
    (
        "exterior",
        ("elevation", "building view", "tower view", "exterior", "project view"),
        ("elevation", "exterior", "render", "view", "facade"),
    ),
    (
        "interior",
        ("interior", "inside view", "room view", "lobby", "living space"),
        ("interior", "room", "lobby", "lounge"),
    ),
    (
        "logo",
        ("logo", "brand"),
        ("logo", "brand"),
    ),
    (
        "table",
        ("payment plan", "payment chart", "price chart", "cost sheet", "schedule"),
        ("payment", "installment", "schedule", "amount", "milestone"),
    ),
]

# ── Filename-based hints ───────────────────────────────────────────────────────
_FILENAME_HINTS: list[tuple[str, re.Pattern]] = [
    ("floor_plan", re.compile(r"floor|unit|layout|bhk", re.IGNORECASE)),
    ("master_plan", re.compile(r"master|site|township", re.IGNORECASE)),
    ("location_map", re.compile(r"location|map|connectivity", re.IGNORECASE)),
    ("amenity", re.compile(r"amenity|amenities|clubhouse|gym|pool", re.IGNORECASE)),
    ("exterior", re.compile(r"elevation|exterior|view|render|facade", re.IGNORECASE)),
    ("interior", re.compile(r"interior|room|lobby", re.IGNORECASE)),
    ("logo", re.compile(r"logo|brand", re.IGNORECASE)),
    ("table", re.compile(r"payment|price|chart|table|schedule", re.IGNORECASE)),
]


# ── Public API ─────────────────────────────────────────────────────────────────

def process_images(
    pages: list[dict],
    document_id: str,
    source_file: str,
    metadata: dict | None = None,
) -> list[dict]:
    """
    Build image metadata records from processed PDF pages, extracting proximal captions,
    figure numbers, and OCR labels.
    """
    if _IMAGE_ANALYZER_ENHANCED_AVAILABLE and process_images_enhanced is not None:
        try:
            return process_images_enhanced(pages, document_id, source_file, metadata)
        except Exception as exc:
            logger.warning(
                "Enhanced image processor failed for document_id=%s; falling back. %s",
                document_id,
                exc,
            )

    meta = metadata or {}
    records: list[dict] = []

    for page in pages:
        page_number = page.get("page_number", 0)
        page_text = (page.get("text") or "").strip()
        ocr_text = (page.get("ocr_text") or "").strip()

        for img in page.get("images", []):
            image_id = img.get("image_id") or f"page_{page_number}"
            image_path = img.get("image_path", "")
            local_path = img.get("local_path")
            image_url = image_path
            figure_number = img.get("figure_number", "Page Image")
            ocr_labels = img.get("ocr_labels", "")

            image_type = img.get("image_type") or classify_image_type(
                "\n".join(filter(None, (img.get("caption", ""), img.get("surrounding_explanation", ""), page_text))),
                image_id,
            )
            caption = img.get("caption") or generate_caption(image_type, page_text, page_number)
            nearby_text = (img.get("surrounding_explanation") or page_text)[:1000]
            section = img.get("section") or page.get("section", "General")

            if meta.get("domain") == "excavator":
                payload_metadata = {
                    "doc_id": meta.get("doc_id", document_id),
                    "document_id": document_id,
                    "doc_type": meta.get("doc_type", ""),
                    "title": meta.get("title", ""),
                    "source_file": source_file,
                    "revision_date": meta.get("revision_date", ""),
                    "ingested_at": meta.get("ingested_at", ""),
                    "page_number": page_number,
                    "section_path": meta.get("section_path") or section,
                    "section": section,
                    "machine_model": meta.get("machine_model", "R215L"),
                    "component_tags": meta.get("component_tags", []),
                    "dtc_codes": meta.get("dtc_codes", []),
                    "supersedes_doc_id": meta.get("supersedes_doc_id", ""),
                    "confidence_weight": meta.get("confidence_weight", 1.0),
                    "domain": "excavator",
                    "image_id": image_id,
                    "image_path": image_path,
                    "image_url": image_url,
                    "image_type": image_type,
                    "caption": caption,
                    "nearby_text": nearby_text[:500],
                    "ocr_context": ocr_text[:500],
                    "figure_number": figure_number,
                    "ocr_labels": ocr_labels,
                    "parent_id": f"{document_id}_page_{page_number}",
                }
            else:
                 payload_metadata = {
                    "document_id": document_id,
                    "page_number": page_number,
                    "image_id": image_id,
                    "image_path": image_path,
                    "image_url": image_url,
                    "image_type": image_type,
                    "caption": caption,
                    "source_file": source_file,
                    "project": meta.get("project_name", meta.get("project", "")),
                    "builder": meta.get("builder", ""),
                    "document_type": meta.get("document_type", ""),
                    "domain": meta.get("domain", "generic"),
                    "section": section,
                    "nearby_text": nearby_text[:500],
                    "ocr_context": ocr_text[:500],
                    "figure_number": figure_number,
                    "ocr_labels": ocr_labels,
                    "parent_id": f"{document_id}_page_{page_number}",
                }

            records.append(
                {
                    "image_id": image_id,
                    "document_id": document_id,
                    "page_number": page_number,
                    "page": page_number,  # legacy field
                    "image_path": image_path,
                    "image_url": image_url,
                    "local_path": local_path,
                    "nearby_page_text": nearby_text,
                    "nearby_text": nearby_text,
                    "ocr_context": ocr_text[:1000],
                    "ocr_labels": ocr_labels,
                    "source_file": source_file,
                    "section": section,
                    "image_type": image_type,
                    "caption": caption,
                    "vector": [],  # filled by embedding service
                    "metadata": payload_metadata,
                }
            )

    logger.info(
        "Processed %d image records for document_id=%s", len(records), document_id
    )
    return records



def classify_image_type(page_text: str, filename: str = "") -> str:
    """
    Classify image type from page text and filename without OCR.

    Returns one of: floor_plan, master_plan, location_map, amenity,
    exterior, interior, logo, table, other.
    """
    # 1. Try filename hints first
    for image_type, pattern in _FILENAME_HINTS:
        if pattern.search(filename):
            return image_type

    if re.search(r"\b(?:figure|fig\.?|diagram|drawing|schematic|[A-Z]{1,3}\d{4,}[A-Z0-9]*)\b", page_text, re.I):
        return "diagram"

    if not page_text:
        return "other"

    haystack = page_text.lower()
    best_type = "other"
    best_score = 0

    for image_type, phrases, keywords in _CLASSIFICATION_RULES:
        phrase_score = sum(4 for phrase in phrases if phrase in haystack)
        keyword_score = sum(1 for kw in keywords if kw in haystack)
        score = phrase_score + keyword_score
        if score > best_score:
            best_score = score
            best_type = image_type

    return best_type


def generate_caption(image_type: str, page_text: str, page_number: int) -> str:
    """
    Generate a simple text caption from image type and page context.

    Never reads pixels or uses OCR — purely text-derived.
    """
    type_labels = {
        "floor_plan": "Floor plan",
        "master_plan": "Master plan",
        "location_map": "Location map",
        "amenity": "Amenity image",
        "exterior": "Exterior / elevation view",
        "interior": "Interior view",
        "logo": "Developer logo",
        "table": "Pricing / payment table",
        "other": "Document image",
    }
    label = type_labels.get(image_type, "Document image")

    # Extract first 120 chars of meaningful page text for context
    short_context = ""
    if page_text:
        clean = re.sub(r"\s+", " ", page_text[:300]).strip()
        short_context = clean[:120]

    if short_context:
        return f"{label} on page {page_number}. Context: {short_context}"
    return f"{label} on page {page_number}."


def build_image_embedding_text(record: dict) -> str:
    """
    Build a combined structured multimodal text string for image embedding.
    Combines Document, Section, Caption, Nearby Paragraph context, OCR tags, and Image Summary.
    """
    if _IMAGE_ANALYZER_ENHANCED_AVAILABLE and build_image_embedding_text_enhanced is not None:
        try:
            return build_image_embedding_text_enhanced(record)
        except Exception as exc:
            logger.warning(
                "Enhanced image embedding text builder failed for image_id=%s; falling back. %s",
                record.get("image_id"),
                exc,
            )

    meta = record.get("metadata", {})
    parts = []
    
    doc_title = meta.get("title") or record.get("source_file") or ""
    if doc_title:
        parts.append(f"Document: {doc_title}")
        
    section = meta.get("section_path") or record.get("section") or ""
    if section:
        parts.append(f"Section Heading: {section}")
        
    page_num = record.get("page_number") or meta.get("page_number")
    if page_num:
        parts.append(f"Page Number: {page_num}")
        
    caption = record.get("caption") or meta.get("caption") or ""
    if caption:
        parts.append(f"Nearby Caption: {caption}")
        
    nearby_text = record.get("nearby_text") or record.get("nearby_page_text") or ""
    if nearby_text:
        parts.append(f"Nearby Context: {nearby_text[:400]}")
        
    ocr_labels = record.get("ocr_labels") or record.get("ocr_context") or ""
    if ocr_labels:
        parts.append(f"OCR Extracted Text: {ocr_labels[:400]}")
        
    img_summary = record.get("description") or record.get("caption") or ""
    if img_summary:
        parts.append(f"Image Summary: {img_summary}")
        
    return "\n".join(parts)
